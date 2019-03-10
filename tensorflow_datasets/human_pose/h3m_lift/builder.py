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
# TODO(jackd): tests, add configs, check hourglass/finetuned_hourglass works

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections
import itertools
import distutils.version
import zipfile

from absl import logging

import numpy as np
import tensorflow as tf

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core import api_utils
from tensorflow_datasets.core import utils
from tensorflow_datasets.core.download import extractor
from tensorflow_datasets.core.utils import py_utils

from tensorflow_datasets.human_pose.h3m_lift import skeleton
from tensorflow_datasets.human_pose.h3m_lift import transform


def _as_h5py(fp):
  h5py = tfds.core.lazy_imports.h5py
  try:
    return h5py.File(fp, "r")
  except OSError:
    if distutils.version.LooseVersion(h5py.__version__) < "2.9.0":
      py_utils.reraise(
        "Error reading GFile with h5py. Please upgrade to at least 2.9.0")
    else:
      raise


CAMERA_IDS = (
    "54138969",
    "55011271",
    "58860488",
    "60457274",
)

TRAIN_SUBJECT_IDS = ('S1', 'S5', 'S6', 'S7', 'S8')
VALIDATION_SUBJECT_IDS = ('S9', 'S11')
SUBJECT_IDS = TRAIN_SUBJECT_IDS + VALIDATION_SUBJECT_IDS


class Source2D(object):
  GROUND_TRUTH = "ground_truth"
  HOURGLASS = "hourglass"
  HOURGLASS_FINETUNED = "hourglass_finetuned"

  @classmethod
  def all(cls):
    return (
      Source2D.GROUND_TRUTH,
      Source2D.HOURGLASS,
      Source2D.HOURGLASS_FINETUNED,
    )

  @classmethod
  def validate(cls, key):
    return key in cls.all()


ExtrinsicCameraParams = collections.namedtuple(
    'ExtrinsicCameraParams', ['rotation', 'translation'])
IntrinsicCameraParams = collections.namedtuple(
    'IntrinsicCameraParams',
    ['focal_length', 'center', 'radial_dist_coeff', 'tangential_dist_coeff'])


def _load_camera_params(hf, path):
  rotation = np.array(hf[path.format('R')]).T
  translation = np.array(hf[path.format('T')])[:, 0]
  focal_length = np.array(hf[path.format('f')])[:, 0]
  center = np.array(hf[path.format('c')])[:, 0]
  radial_dist_coeff = np.array(hf[path.format('k')])[:, 0]
  tangential_dist_coeff = np.array(hf[path.format('p')][:])[:, 0]

  camera_id = hf[path.format('Name')][:]
  camera_id = "".join([chr(item) for item in camera_id])

  return (
    rotation, translation,
    focal_length, center, radial_dist_coeff, tangential_dist_coeff,
    camera_id)


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

  params = (
    rotation, translation,
    focal_length, center, radial_dist_coeff, tangential_dist_coeff)


  # with h5py.File(path,'r') as hf:
  with tf.io.gfile.GFile(path, "rb") as fp:
    hf = _as_h5py(fp)
    for i, subject_id in enumerate(subject_ids):
      si = int(subject_id[1:])
      for j, camera_id in enumerate(camera_ids):
        example_params = _load_camera_params(
          hf, 'subject%d/camera%d/{0}' % (si,j+1))

        assert(example_params[-1] == camera_id)
        for param, example_param in zip(params, example_params[:-1]):
          param[i, j] = example_param

  return ExtrinsicCameraParams(*params[:2]), IntrinsicCameraParams(*params[2:])


class H3mLiftConfig(tfds.core.BuilderConfig):

  @api_utils.disallow_positional_args()
  def __init__(
      self,
      version=tfds.core.Version(0, 0, 1),
      source_2d=Source2D.GROUND_TRUTH,
      skeleton_2d=skeleton.s14,
      skeleton_3d=skeleton.s14,
      **kwargs):
    self.source_2d = source_2d
    self.skeleton_2d = skeleton_2d
    self.skeleton_3d = skeleton_3d
    super(H3mLiftConfig, self).__init__(version=version, **kwargs)

  @property
  def num_joints_2d(self):
    return self.skeleton_2d.num_joints

  @property
  def num_joints_3d(self):
    return self.skeleton_3d.num_joints


def _get_base_data(base_dir):
  # dl_manager doesn't like dropbox apparently...
  if not tf.io.gfile.exists(base_dir):
    tf.io.gfile.makedirs(base_dir)
  path = os.path.join(base_dir, "h36m.zip")
  if not tf.io.gfile.exists(path):
    url = "https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip"
    ex = "wget -O %s %s" % (path, url)
    msg = ("Please manually download files from"
             " %s and place it at %s\n e.g.\n%s" % (url, path, ex))
    # # wget from command line works, but fails via python...
    # try:
    #   import wget
    #   logging.info("Downloading base h3m data from %s" % url)
    #   wget.download(url, path)
    # except Exception:
    #   py_utils.reraise(msg)
    raise AssertionError(msg)

  return path


def _get_finetuned_data(base_dir):
  url = "https://drive.google.com/open?id=0BxWzojlLp259S2FuUXJ6aUNxZkE"
  # Not sure how to handle this at the moment...
  path = os.path.join(base_dir, "stacked_hourglass_fined_tuned_240.tar.gz")
  if not tf.io.gfile.exists(path):
    raise AssertionError(
      "You must download finetuned dataset files manually from %s and place "
      "them in %s" % (url, path))
  return path


def _filename_3d(sequence_id):
  return "%s.h5" % sequence_id.replace('_', ' ')


def _filename_2d(sequence_id, camera_id):
  return "%s.%s.h5" % (sequence_id.replace(' ', '_'), camera_id)


def _p3_subject_dir(base_dir, subject_id):
  return os.path.join(
    base_dir, "h36m", subject_id, "MyPoses", "3D_positions")


def _load_p3(base_dir, subject_id, sequence_id):
  path = os.path.join(
    _p3_subject_dir(base_dir, subject_id), _filename_3d(sequence_id))
  with tf.io.gfile.GFile(path, "rb") as fobj:
    return np.reshape(
      _as_h5py(fobj)["3D_positions"][:], (-1, 32, 3))


def _load_p2_hourglass(
    base_dir, subject_id, sequence_id, camera_id):
  path = os.path.join(
    base_dir, "h36m", subject_id,
    "StackedHourglass",
    _filename_2d(sequence_id, camera_id))
  with tf.io.gfile.GFile(path, "rb") as fobj:
    return _as_h5py(fobj)["3D_positions"][:]


def _load_p2_finetuned(
    finetuned_dir, subject_id, sequence_id, camera_id):
  path = os.path.join(
    finetuned_dir, subject_id, "StackedHourglassFineTuned240",
    _filename_2d(sequence_id, camera_id))
  with tf.io.gfile.GFile(path, "rb") as fobj:
    return _as_h5py(fobj)["3D_positions"][:]


def _ground_truth_loader(base_dir, extrinsic_params, intrinsic_params):
  subject_indices = {k: i for i, k in enumerate(SUBJECT_IDS)}
  camera_indices = {k: i for i, k in enumerate(CAMERA_IDS)}

  def f(subject_id, sequence_id, camera_id):
    p3 = _load_p3(base_dir, subject_id, sequence_id)
    subject_index = subject_indices[subject_id]
    camera_index = camera_indices[camera_id]
    p3_camera = transform.world_to_camera_frame(
      p3,
      *[e[subject_index][camera_index] for e in extrinsic_params])
    p3_camera = np.reshape(p3_camera, (-1, 3))
    p2 = transform.project_points(
      p3_camera,
      *[i[subject_index][camera_index] for i in intrinsic_params])[0]
    p2 = np.reshape(p2, p3.shape[:-1] + (2,))
    return p3, p2
  return f


def _hourglass_loader(base_dir):
  def f(subject_id, sequence_id, camera_id):
    p3 = _load_p3(base_dir, subject_id, sequence_id)
    p2 = _load_p2_hourglass(base_dir, subject_id, sequence_id, camera_id)
    return p3, p2
  return f


def _finetuned_loader(base_dir, finetuned_dir):
  def f(subject_id, sequence_id, camera_id):
    p3 = _load_p3(base_dir, subject_id, sequence_id)
    p2 = _load_p2_finetuned(finetuned_dir, subject_id, sequence_id, camera_id)
    return p3, p2
  return f


class H3mLift(tfds.core.GeneratorBasedBuilder):
  """
  2D - 3D pose lifting task for human pose estimation on human 3.6m.

  3D data is provided in world coordinates by default. We provide extrinsic
  camera parameters for transforming to world coordinates and intrinsic
  camera coordinates for projecting other 3D points in
  `H3mLift.load_camera_params`.

  Note GROUND_TRUTH 2D poses are available via the "base_sXX" configs, so there
  is no need to transform/project the ground truth 3D poses.
  """

  BUILDER_CONFIGS = [
    H3mLiftConfig(
      name="base_s14",
      source_2d=Source2D.GROUND_TRUTH,
      skeleton_2d=skeleton.s14,
      skeleton_3d=skeleton.s14,
      description="14 joint ground truth points"),
    H3mLiftConfig(
      name="base_s16",
      source_2d=Source2D.GROUND_TRUTH,
      skeleton_2d=skeleton.s16,
      skeleton_3d=skeleton.s16,
      description="16 joint ground truth points"),
  ]

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
        citation="""\
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}"""
    )

  def _split_generators(self, dl_manager):
    config = self.builder_config
    manual_dir = dl_manager._manual_dir
    base_dir = dl_manager.extract(_get_base_data(manual_dir))

    camera_path = self._camera_path
    if not tf.io.gfile.exists(camera_path):
      tf.io.gfile.copy(
        os.path.join(base_dir, "h36m", "cameras.h5"), camera_path)

    p3_indices = skeleton.conversion_indices(skeleton.s32, config.skeleton_3d)

    if config.source_2d == Source2D.GROUND_TRUTH:
      loader = _ground_truth_loader(base_dir, *self.load_camera_params())
      p2_indices = skeleton.conversion_indices(skeleton.s32, config.skeleton_2d)
    else:
      assert(config.skeleton_2d is skeleton.mpii_s16)
      p2_indices = None

      if config.source_2d == Source2D.HOURGLASS:
        loader = _hourglass_loader(base_dir)
      elif config.source_2d == Source2D.HOURGLASS_FINETUNED:
        loader = _finetuned_loader(
          base_dir, dl_manager.extract(_get_finetuned_data(manual_dir)))
      else:
        raise ValueError("Invalid source_2d '%s'" % config.source_2d)

    shared_kwargs = dict(
      base_dir=base_dir,
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
          self, subject_ids, base_dir, loader, p2_indices, p3_indices):
    for subject_id in subject_ids:
      subject_dir = _p3_subject_dir(base_dir, subject_id)
      for fn in tf.io.gfile.listdir(subject_dir):
        sequence_id, ext = fn.split(".")
        assert(ext == "h5")
        for camera_id in CAMERA_IDS:
          p3, p2 = loader(subject_id, sequence_id, camera_id)
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
