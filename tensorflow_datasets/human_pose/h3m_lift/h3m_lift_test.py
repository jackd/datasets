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

"""Tests for h3m_lift dataset module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_datasets import testing
from tensorflow_datasets.human_pose import h3m_lift
from tensorflow_datasets.human_pose.h3m_lift.builder import CAMERA_IDS


DL_EXTRACT_RESULT = {
  "base_data": "./",
  "finetuned_data": "stacked_hourglass_fined_tuned_240",
}

SPLITS = {
    "train": 2,
    "validation": 1,
}


class H3mLiftGrountTruthTest(testing.DatasetBuilderTestCase):
  DATASET_CLASS = h3m_lift.H3mLift
  DL_EXTRACT_RESULT = DL_EXTRACT_RESULT

  BUILDER_CONFIG_NAMES_TO_TEST = ["ground_truth"]
  SPLITS = {k: len(CAMERA_IDS) * v for k, v in SPLITS.items()}


class H3mLiftHouglassTest(testing.DatasetBuilderTestCase):
  DATASET_CLASS = h3m_lift.H3mLift
  DL_EXTRACT_RESULT = DL_EXTRACT_RESULT

  BUILDER_CONFIG_NAMES_TO_TEST = [
    "hourglass",
    "hourglass_finetuned",
  ]
  SPLITS = SPLITS

if __name__ == "__main__":
  testing.test_main()
