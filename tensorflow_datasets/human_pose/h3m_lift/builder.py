# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""2D-3D human pose dimensionality lifting on Human 3.6 Million."""
# TODO(jackd):
# tests

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import itertools
import distutils.version

from absl import logging

import numpy as np
import tensorflow as tf

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core import api_utils
from tensorflow_datasets.core import utils
from tensorflow_datasets.core.download import extractor
from tensorflow_datasets.core.utils import py_utils

from tensorflow_datasets.human_pose import skeleton as base_skeleton
from tensorflow_datasets.human_pose.mpii import skeleton as mpii_skeleton
from tensorflow_datasets.human_pose.h3m_lift import skeleton
from tensorflow_datasets.human_pose.h3m_lift import transform
from tensorflow_datasets.core import lazy_imports
from tensorflow_datasets.core.download import resource as resource_lib


H3M_CITATIONS = ("""\
@article{h36m_pami,
  author = {Ionescu, Catalin and Papava, Dragos and Olaru, Vlad and Sminchisescu, Cristian},
  title = {Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  publisher = {IEEE Computer Society},
  year = {2014}
}""","""\
@inproceedings{IonescuSminchisescu11,
  author = {Catalin Ionescu, Fuxin Li, Cristian Sminchisescu},
  title = {Latent Structured Models for Human Pose Estimation},
  booktitle = {International Conference on Computer Vision},
  year = {2011}
}""")

BASELINE_CITATION = """\
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}"""


CAMERA_IDS = (
    "54138969",
    "55011271",
    "58860488",
    "60457274",
)

class _ContextWrapper(object):
  def __init__(self, wrapped):
    self._wrapped = wrapped

  def __enter__(self):
    return self._wrapped

  def __exit__(self, *args, **kwargs):
    pass


class _SessionWrapper(object):
  def __init__(self, graph, sess_fn):
    self._graph = graph
    self._sess_fn = sess_fn
    self._sess = None

  def __enter__(self):
    assert(self._sess is None)
    self._sess = tf.Session(graph=self._graph)
    return self

  def __exit__(self, *args, **kwargs):
    self._sess.close()
    self._sess = None

  def __call__(self, *args, **kwargs):
    return self._sess_fn(self._sess, *args, **kwargs)



_camera_index = {k: i for i, k in enumerate(CAMERA_IDS)}


TRAIN_SUBJECT_IDS = ('S1', 'S5', 'S6', 'S7', 'S8')
VALIDATION_SUBJECT_IDS = ('S9', 'S11')
SUBJECT_IDS = TRAIN_SUBJECT_IDS + VALIDATION_SUBJECT_IDS

ExtrinsicCameraParams = collections.namedtuple(
    'ExtrinsicCameraParams', ['rotation', 'translation'])
IntrinsicCameraParams = collections.namedtuple(
    'IntrinsicCameraParams',
    ['focal_length', 'center', 'radial_dist_coeff', 'tangential_dist_coeff'])


def _camera_subpath(subject_id, camera_id):
  return "subject%d/camera%d" % (
    int(subject_id[1:]), _camera_index[camera_id]+1)


def load_camera_params(path, subject_ids=SUBJECT_IDS, camera_ids=CAMERA_IDS):
  """Loads the cameras parameters of h36m

  Args:
    path: path to hdf5 file with h3m camera data
    subject_ids: list/tuple of subject ids
    camera_ids: list/tuple of camera_ids

  Returns:
    extrinsics: ExtrinsicCameraParams, named tuple with:
        rotation: Camera rotation matrix
          shape (num_subject, num_cameras, 3, 3)

        translation: Camera translation parameters
          shape (num_subject, num_cameras, 3)

    intrinsics: list of lists of IntrinsicCameraParams, named tuple with:
        focal_length: Camera focal length
          shape (num_subject, num_cameras)
        center: Camera center
          shape (num_subject, num_cameras, 2)
        radial_dist: Camera radial distortion coefficients
          shape (num_subject, num_cameras, 3)
        tangential_dist: Camera tangential distortion coefficients
          shape (num_subject, num_cameras, 2)

    The first two axis correspond to the order of `subject_ids` and `camera_ids`
    inputs.
  """

  num_subjects = len(subject_ids)
  num_cameras = len(camera_ids)

  def _init(*trailing_dims):
    return np.zeros(
      (num_subjects, num_cameras) + trailing_dims, dtype=np.float32)

  rotation = _init(3, 3)
  translation = _init(3)
  focal_length = _init(2)
  center = _init(2)
  radial_dist_coeff = _init(3)
  tangential_dist_coeff = _init(2)

  rest = (
    translation, focal_length, center, radial_dist_coeff, tangential_dist_coeff)
  rest_keys = ('T', 'f', 'c', 'k', 'p')

  with tf.io.gfile.GFile(path, "rb") as fp:
    hf = lazy_imports.h5py.File(fp, "r")  # pylint: disable=no-member
    for i, subject_id in enumerate(subject_ids):
      for j, camera_id in enumerate(camera_ids):
        group = hf[_camera_subpath(subject_id, camera_id)]
        rotation[i, j] = np.array(group['R']).T
        for param, key in zip(rest, rest_keys):
          param[i, j] = group[key][:, 0]

  extr = ExtrinsicCameraParams(rotation, rest[0])
  intr = IntrinsicCameraParams(*rest[1:])
  return extr, intr


class H3mLiftConfig(tfds.core.BuilderConfig):

  @api_utils.disallow_positional_args()
  def __init__(
      self,
      version=tfds.core.Version(0, 0, 1),
      skeleton_2d=skeleton.s14,
      skeleton_3d=skeleton.s14,
      **kwargs):
    self.skeleton_2d = skeleton_2d
    self.skeleton_3d = skeleton_3d
    super(H3mLiftConfig, self).__init__(version=version, **kwargs)

  @property
  def num_joints_2d(self):
    return self.skeleton_2d.num_joints

  @property
  def num_joints_3d(self):
    return self.skeleton_3d.num_joints


def _get_base_resource(manual_dir):
  # dl_manager doesn't like dropbox apparently...
  path = os.path.join(manual_dir, "h36m.zip")
  if not tf.io.gfile.exists(path):
    if not tf.io.gfile.exists(manual_dir):
      tf.io.gfile.makedirs(manual_dir)
    url = "https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip"
    ex = "wget -O %s %s" % (path, url)
    msg = ("Please manually download files from"
             " %s and place it at %s\n e.g.\n%s" % (url, path, ex))
    raise AssertionError(msg)
  return resource_lib.Resource(
    path=path,
    extract_method=resource_lib.ExtractMethod.ZIP,
  )


def _get_finetuned_resource(manual_dir):
  url = "https://drive.google.com/open?id=0BxWzojlLp259S2FuUXJ6aUNxZkE"
  # DL manager fails to get drive data, so using manual fallback
  path = os.path.join(
    manual_dir, "stacked_hourglass_fined_tuned_240.tar.gz")
  if not tf.io.gfile.exists(path):
    raise AssertionError(
      "You must download finetuned dataset files manually from %s and place "
      "them in %s" % (url, path))
  return resource_lib.Resource(
    path=path,
    extract_method=resource_lib.ExtractMethod.TAR_GZ,
  )


def _filename_3d(sequence_id):
  return "%s.h5" % sequence_id.replace('_', ' ')


def _filename_2d(sequence_id, camera_id):
  return "%s.%s.h5" % (sequence_id.replace(' ', '_'), camera_id)


def _p3_subject_dir(h36m_dir, subject_id):
  return os.path.join(
    h36m_dir, subject_id, "MyPoses", "3D_positions")


def _load_p3(h36m_dir, subject_id, sequence_id):
  path = os.path.join(
    _p3_subject_dir(h36m_dir, subject_id), _filename_3d(sequence_id))
  if not tf.io.gfile.exists(path):
      logging.info("No 3d pose data at %s, skipping" % path)
      return None
  with tf.io.gfile.GFile(path, "rb") as fobj:
    data = lazy_imports.h5py.File(fobj, "r")["3D_positions"][:]   # pylint: disable=no-member
    data = np.reshape(data, (32, 3, -1))
    data = np.transpose(data, (2, 0, 1))
    return data.astype(np.float32)


def _ground_truth_loader(h36m_dir, extrinsic_params, intrinsic_params):
  subject_indices = {k: i for i, k in enumerate(SUBJECT_IDS)}
  camera_indices = {k: i for i, k in enumerate(CAMERA_IDS)}

  def get_p2(p3, extr, intr, si, ci):
    assert(len(p3.shape) == 3)
    n_frames = tf.shape(p3)[0]
    p3 = tf.reshape(p3, (-1, 3))
    extr = [e[si, ci] for e in extr]
    intr = [i[si, ci] for i in intr]
    p3_cam = transform.world_to_camera_frame(p3, *extr)
    p2 = transform.project_points(p3_cam, *intr)[0]
    return tf.reshape(p2, (n_frames, 32, 2))

  if tf.executing_eagerly():
    def eager_fn(subject_id, sequence_id, camera_id):
      p3 = _load_p3(h36m_dir, subject_id, sequence_id)
      ci = camera_indices[camera_id]
      si = subject_indices[subject_id]
      with tf.device('/cpu:0'):
        p2 = get_p2(p3, extrinsic_params, intrinsic_params, si, ci)
      return p3, p2.numpy()
    return _ContextWrapper(eager_fn)

  else:
    graph = tf.Graph()
    with graph.as_default():
      # much faster for small operations involved
      with tf.device('/cpu:0'):
        p3_pl = tf.placeholder(
          shape=(None, 32, 3), dtype=tf.float32)
        ci = tf.placeholder(shape=(), dtype=tf.int64)
        si = tf.placeholder(shape=(), dtype=tf.int64)
        extr_tf = [tf.constant(e) for e in extrinsic_params]
        intr_tf = [tf.constant(i) for i in intrinsic_params]
        p2_tf = get_p2(p3_pl, extr_tf, intr_tf, si, ci)

    def sess_fn(sess, subject_id, sequence_id, camera_id):
      p3 = _load_p3(h36m_dir, subject_id, sequence_id)
      p2 = sess.run(p2_tf, feed_dict={
        si: subject_indices[subject_id],
        ci: camera_indices[camera_id],
        p3_pl: p3,
      })
      return p3, p2

    return _SessionWrapper(graph, sess_fn)


def _hourglass_loader(h36m_dir):
  def f(subject_id, sequence_id, camera_id):
    p3 = _load_p3(h36m_dir, subject_id, sequence_id)
    path = os.path.join(
        h36m_dir, subject_id, "StackedHourglass",
        _filename_2d(sequence_id, camera_id))
    if not tf.io.gfile.exists(path):
      logging.info("No 2d pose data at %s, skipping" % path)
      return p3, None
    with tf.io.gfile.GFile(path, "rb") as fobj:
      p2 = lazy_imports.h5py.File(fobj, "r")["poses"][:]  # pylint: disable=no-member
    return p3, p2
  return _ContextWrapper(f)


def _finetuned_loader(h36m_dir, finetuned_dir):
  def f(subject_id, sequence_id, camera_id):
    p3 = _load_p3(h36m_dir, subject_id, sequence_id)
    path = os.path.join(
      finetuned_dir, subject_id, "StackedHourglassFineTuned240",
      _filename_2d(sequence_id, camera_id))
    if not tf.io.gfile.exists(path):
      logging.info("No 2d pose data at %s, skipping" % path)
      return p3, None
    with tf.io.gfile.GFile(path, "rb") as fobj:
      p2 = lazy_imports.h5py.File(fobj, "r")["poses"][:]  # pylint: disable=no-member
    return p3, p2
  return _ContextWrapper(f)


GROUND_TRUTH = H3mLiftConfig(
      name="ground_truth",
      skeleton_2d=skeleton.s16,
      skeleton_3d=skeleton.s16,
      description="pose_2d is from the projected ground truth pose_3d")

HOURGLASS = H3mLiftConfig(
      name="hourglass",
      skeleton_2d=mpii_skeleton.s16,
      skeleton_3d=skeleton.s16,
      description=
      "pose_2d is from a stacked hourglass network trained on mpii")

HOURGLASS_FINETUNED = H3mLiftConfig(
      name="hourglass_finetuned",
      skeleton_2d=mpii_skeleton.s16,
      skeleton_3d=skeleton.s16,
      description=
        "pose_2d is from a stacked hourglass network fine-tuned on h3m")


class H3mLift(tfds.core.GeneratorBasedBuilder):
  """2D - 3D pose lifting task for human pose estimation on human 3.6m.

  `H3mLift.load_camera_params` provides camera parameters. Basic transformations
  and projection operations are available in
  `tfds.human_pose.h3m_lift.transform`.

  For information about the joints used, see
  `tfds.human_pose.h3m_lift.skeleton`

  3D data is provided in world coordinates by default on a 16 joint skeleton.

  2D data is available from 3 configs:
    * `ground_truth`: the 3D poses projected to 2D using camera parameters,
      using the same skeleton as 3D poses (skeleton.s16)
    * `hourglass`: a stacked hourglass network trained on MPII images. Note the
      skeleton here is `skeleton.mpii_s16`, which has the same number of joints,
      but different joints (only 1 joint in the head and an independent pelvis).
    * `hourglass_finetuned`: similar to `hourglass` except the network is
      finetuned on h3m dataset (same source as the 3D poses).
  """

  BUILDER_CONFIGS = [GROUND_TRUTH, HOURGLASS, HOURGLASS_FINETUNED]

  @property
  def skeleton_2d(self):
    return self.builder_config.skeleton_2d

  @property
  def skeleton_3d(self):
    return self.builder_config.skeleton_3d

  @property
  def _camera_path(self):
    return os.path.join(self._data_dir, "cameras.h5")

  def load_camera_params(
      self, subject_ids=SUBJECT_IDS, camera_ids=CAMERA_IDS):
    return load_camera_params(
      self._camera_path, subject_ids=subject_ids, camera_ids=camera_ids)

  def _info(self):
    config = self.builder_config
    return tfds.core.DatasetInfo(
        builder=self,
        description=(
            "2D / 3D human poses for varying number of joints and 2D "
            "sources"),
        features=tfds.features.FeaturesDict({
            "pose_2d": tfds.features.Sequence(tfds.features.Tensor(
                shape=(config.num_joints_2d, 2), dtype=tf.float32)),
            "pose_3d": tfds.features.Sequence(tfds.features.Tensor(
                shape=(config.num_joints_3d, 3), dtype=tf.float32)),
            "camera_id": tfds.features.ClassLabel(names=CAMERA_IDS),
            "subject_id": tfds.features.ClassLabel(names=SUBJECT_IDS),
            "sequence_id": tfds.features.Text(),
        }),
        urls=[
            "http://vision.imar.ro/human3.6m/description.php",    # h3m
            "https://github.com/una-dinosauria/3d-pose-baseline", # baseline
            "https://github.com/princeton-vl/pose-hg-demo",       # hourglass
        ],
        supervised_keys=("pose_2d", "pose_3d"),
        citation="\n".join(H3M_CITATIONS + (BASELINE_CITATION,))
    )

  def _split_generators(self, dl_manager):
    config = self.builder_config
    manual_dir = dl_manager.manual_dir
    download_res = {"base_data": _get_base_resource(manual_dir)}
    if config is HOURGLASS_FINETUNED:
      download_res["finetuned_data"] = _get_finetuned_resource(manual_dir)

    dirs = dl_manager.extract(download_res)
    base_dir = dirs["base_data"]
    h36m_dir = os.path.join(base_dir, "h36m")
    camera_path = self._camera_path

    if not tf.io.gfile.exists(camera_path):
      tf.io.gfile.copy(
        os.path.join(h36m_dir, "cameras.h5"), camera_path)



    p3_indices = base_skeleton.conversion_indices(
      skeleton.s32, config.skeleton_3d)

    if config is GROUND_TRUTH:
      loader = _ground_truth_loader(h36m_dir, *self.load_camera_params())
      p2_indices = base_skeleton.conversion_indices(
        skeleton.s32, config.skeleton_2d)
    else:
      assert(config.skeleton_2d is mpii_skeleton.s16)
      p2_indices = None

      if config is HOURGLASS:
        loader = _hourglass_loader(h36m_dir)
      elif config is HOURGLASS_FINETUNED:
        loader = _finetuned_loader(h36m_dir, dirs["finetuned_data"])
      else:
        raise ValueError("Invalid config '%s'" % config)

    shared_kwargs = dict(
      h36m_dir=h36m_dir,
      loader=loader,
      p2_indices=p2_indices,
      p3_indices=p3_indices,
    )

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=10,
            gen_kwargs=dict(
              subject_ids=TRAIN_SUBJECT_IDS,
              **shared_kwargs
            ),
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            num_shards=4,
            gen_kwargs=dict(
              subject_ids=VALIDATION_SUBJECT_IDS,
              **shared_kwargs
            )
        ),
    ]

  def _generate_examples(
          self, subject_ids, h36m_dir, loader, p2_indices, p3_indices):
    with loader as loader_fn:
      for subject_id in subject_ids:
        subject_dir = _p3_subject_dir(h36m_dir, subject_id)
        if not tf.io.gfile.isdir(subject_dir):
          logging.info("No subject data found for subject %s at %s, skipping"
                      % (subject_id, subject_dir))
          continue
        for fn in tf.io.gfile.listdir(subject_dir):
          sequence_id, ext = fn.split(".")
          assert(ext == "h5")
          for camera_id in CAMERA_IDS:
            p3, p2 = loader_fn(subject_id, sequence_id, camera_id)
            if p2 is None:
              continue
            if p3_indices is not None:
              p3 = p3[:, p3_indices, :]
            if p2_indices is not None:
              p2 = p2[:, p2_indices, :]
            yield dict(
              pose_2d=p2.astype(np.float32),
              pose_3d=p3.astype(np.float32),
              camera_id=camera_id,
              subject_id=subject_id,
              sequence_id=sequence_id,
            )
