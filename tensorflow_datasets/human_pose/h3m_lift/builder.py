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
from tensorflow_datasets.core.utils import py_utils

from tensorflow_datasets.human_pose.h3m_lift import skeleton
from tensorflow_datasets.human_pose.h3m_lift import transform


def _as_h5py(fp):
  h5py = tfds.core.lazy_imports.h5py
  try:
    return h5py.File(fp)
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
VAL_SUBJECT_IDS = ('S9', 'S11')
SUBJECT_IDS = TRAIN_SUBJECT_IDS + VAL_SUBJECT_IDS


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

  R = hf[path.format('R')][:]
  R = R.T

  t = hf[path.format('T')][:]
  f = hf[path.format('f')][:]
  c = hf[path.format('c')][:]
  k = hf[path.format('k')][:]
  p = hf[path.format('p')][:]
  p = p[..., -1::-1]

  camera_id = hf[path.format('Name')][:]
  camera_id = "".join([chr(item) for item in camera_id])

  return (
    ExtrinsicCameraParams(R, t),
    IntrinsicCameraParams(f, c, k, p),
    camera_id,
  )


def load_camera_params(path, subject_ids=SUBJECT_IDS):
  """Loads the cameras parameters of h36m

  Args:
    path: path to hdf5 file with h3m camera data
    subject_ids: list of subject ids

  Returns:
    extrinsics: dictionary mapping
      (subject_id, camera_id) -> ExtrinsicCameraParams, named tuple with:
        rotation: (3, 3) Camera rotation matrix
        translation: (3,) Camera translation parameters
    intrinsics: dictionary mapping
      (subject_id, camera_id) -> IntrinsicCameraParams, named tuple with:
        focal_length: () Camera focal length
        center: (2,) Camera center
        radial_dist: (3,) Camera radial distortion coefficients
        tangential_dist: (2,) Camera tangential distortion coefficients
  """

  extrinsics = {}
  intrinsics = {}

  # with h5py.File(path,'r') as hf:
  with tf.io.gfile.GFile(path, "rb") as fp:
    hf = _as_h5py(fp)
    for subject_id in subject_ids:
      si = int(subject_id[1:])
      for c in range(len(CAMERA_IDS)):
        extr, intr, camera_id = _load_camera_params(
          hf, 'subject%d/camera%d/{0}' % (si,c+1))
        extrinsics[(subject_id, camera_id)] = extr
        intrinsics[(subject_id, camera_id)] = intr

  return extrinsics, intrinsics


def stack_params(namedtuple, params_dict, subject_ids, camera_ids):
  """Stack values in the named tuple element-wise.

  Example usage:
  ```python
  extrinsics_dict, _ = load_camera_params(path, SUBJECT_IDS)
  extrisnics_stack = stack_params(
      ExtrinsicCameraParams, extrinsics, SUBJECT_IDS, CAMERA_IDS)

  subject_index = 2  #
  camera_index = 1   #
  subject_id = SUBJECT_IDS[subject_index]
  camera_id = CAMERA_IDS[camera_index]

  stacked_rotation = extrinsics_stacked.rotation[subject_index, camera_index]
  dict_rotation = extrinsics_dict[(subject_id, camera_id)].rotation
  print(stacked_entry is dict_entry) # True
  ```

  Args:
    namedtuple: output of `collections.namedtuple`
    params_dict: dict mapping (subject_id, camera_id) -> instance of named_tuple
    subject_ids: list/tuple of subject ids
    camera_ids: list/tuple of camera_ids

  Returns:
    instance of `namedtuple` where each element has an additional two leading
      axis of size (len(subject_ids), len(camera_ids)).
  """
  out_fields = []
  for i in namedtuple.count:
    val = [
      [params_dict[(subject_id, camera_id)][i] for camera_id in camera_ids]
      for subject_id in subject_ids]
    out_fields.append(val)

  return namedtuple(*out_fields)


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
  path = os.path.join(base_dir, "h36m.zip")
  if not tf.io.gfile.exists(path):
    url = "https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip"
    try:
      import wget
      logging.info("Downloading base h3m data from %s" % url)
      wget.download(url, path)
    except Exception:
      msg = ("Failed to download base data. Please manually download files from"
             " %s and place it at %s" % (url, path))
      py_utils.reraise(msg)
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

  def load_camera_params(self, subject_ids=SUBJECT_IDS):
    path = os.path.join(
      self._data_dir, "TODO")
    return load_camera_params(path, subject_ids)


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
            "sequence_id": tfds.features.Text(),
            "subject_id": tfds.features.ClassLabel(names=SUBJECT_IDS),
        }),
        urls=[
            "http://vision.imar.ro/human3.6m/description.php",  # h3m
            "https://github.com/sta105/3d-pose-baseline",       # baseline
            "https://github.com/princeton-vl/pose-hg-demo",     # hourglass
        ],
        supervised_keys=("pose2d", "pose3d"),
        citation="""\
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}"""
    )

  def _split_generators(self, dl_manager):
    base_path = os.path.join(dl_manager.manual_dir, "h36m.zip")
    finetuned_path = os.path.join(dl_manager.manual_dir, "h36m.zip")
    _get_base_data(base_path)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=10,
            gen_kwargs=dict(
              base_path=base_path,
              finetuned_path=finetuned_path,
              subject_ids=TRAIN_SUBJECT_IDS,
            ),
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            num_shards=4,
            gen_kwargs=dict(
              base_path=base_path,
              finetuned_path=finetuned_path,
              subject_ids=TRAIN_SUBJECT_IDS,
            )
        ),
    ]

  def _generate_examples(self, base_path, finetuned_path, subject_ids):
    raise NotImplementedError("TODO")
