from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import tensorflow as tf
import numpy as np
from tensorflow_datasets.core import lazy_imports
from tensorflow_datasets.core.utils import py_utils
from tensorflow_datasets.testing import fake_data_utils

from tensorflow_datasets.human_pose.h3m_lift.h3m_lift_test import DL_EXTRACT_RESULT
from tensorflow_datasets.human_pose.h3m_lift.h3m_lift_test import SPLITS
from tensorflow_datasets.human_pose.h3m_lift import builder
from tensorflow_datasets.testing import test_utils


flags.DEFINE_string('tfds_dir', py_utils.tfds_dir(),
                    'Path to tensorflow_datasets directory')
flags.DEFINE_integer('sequence_length', 5, 'length of each generated sequence')

FLAGS = flags.FLAGS


def write_h5py(path, key, shape):
  with lazy_imports.h5py.File(path) as hp:  # pylint: disable=no-member
    hp.create_dataset(key, data=np.random.normal(size=shape))


def main(_):
  root_dir = os.path.join(
    FLAGS.tfds_dir, "testing", "test_data", "fake_examples", "h3m_lift")
  examples = (
    ("S1", "Directions 1", builder.CAMERA_IDS[0]),
    ("S5", "Purchases", builder.CAMERA_IDS[1]),
    ("S9", "Walking", builder.CAMERA_IDS[2]),
  )

  splits = {"train": 0, "validation": 0}
  for subject_id, _, __ in examples:
    if subject_id in builder.TRAIN_SUBJECT_IDS:
      splits["train"] += 1
    elif subject_id in builder.VALIDATION_SUBJECT_IDS:
      splits["validation"] += 1
    else:
      raise ValueError("examples inconsistent with test")
  assert(splits == SPLITS)

  sequence_length = FLAGS.sequence_length

  base_dir = os.path.join(
    root_dir, DL_EXTRACT_RESULT["base_data"], "h36m")
  finetuned_dir = os.path.join(
    root_dir, DL_EXTRACT_RESULT["finetuned_data"])

  def clean():
    test_utils.remake_dir(finetuned_dir)
    # don't remake_dir(base_dir) because there's a copy of camera_params in there.
    if tf.io.gfile.isdir(base_dir):
      for p in tf.io.gfile.listdir(base_dir):
        full_dir = os.path.join(base_dir, p)
        if tf.io.gfile.isdir(full_dir):
          tf.io.gfile.rmtree(full_dir)

  clean()

  # base_data
  for subject_id, sequence_id, camera_id in examples:
    p3_dir = os.path.join(base_dir, subject_id, "MyPoses", "3D_positions")
    tf.io.gfile.makedirs(p3_dir)
    p3_fn = "%s.h5" % sequence_id.replace("_", " ")
    write_h5py(
      os.path.join(p3_dir, p3_fn), "3D_positions", (32*3, sequence_length))

    # p2 data
    p2_fn = "%s.%s.h5" % (sequence_id.replace(" ", "_"), camera_id)
    for p2_dir in (
      os.path.join(base_dir, subject_id, "StackedHourglass"),
      os.path.join(finetuned_dir, subject_id, "StackedHourglassFineTuned240")
        ):
      tf.io.gfile.makedirs(p2_dir)
      write_h5py(
        os.path.join(p2_dir, p2_fn), "poses", (sequence_length, 16, 2))

  # create camera.h5

  # touch archives to stop errors requiring download
  with tf.io.gfile.GFile("%s.zip" % base_dir, "wb") as fp:
    fp.write("Not a real .zip file, just for testing")
  with tf.io.gfile.GFile("%s.tar.gz" % finetuned_dir, "wb") as fp:
    fp.write("Not a real .tar.gz file, just for testing")

if __name__ == '__main__':
  app.run(main)
