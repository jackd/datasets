from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import tensorflow_datasets.core.features.rle_feature.rle.tf_impl as brle
import tensorflow_datasets.core.features.rle_feature.rle.shared_tests as st
from tensorflow_datasets.testing import test_utils
tf.compat.v1.enable_eager_execution()
<<<<<<< HEAD


if brle.is_over_1_12:
=======
major, minor = (int(v) for v in tf.__version__.split('.')[:2])
is_over_1_12 = major >= 1 and minor > 12

if is_over_1_12:
>>>>>>> 474073f64fd0b414fbde5080a478d06a3af387d5
  class BlreTfTest(tf.test.TestCase):
    @property
    def impl(self):
      return brle

    def assert_array_equal(self, x, y, *args, **kwargs):
      if isinstance(x, tf.Tensor):
        x = self.evaluate(x)
      if isinstance(y, tf.Tensor):
        y = self.evaluate(y)
      return self.assertAllEqual(x, y, *args, **kwargs)

    def dense_logical_not(self, x):
      return tf.logical_not(x)

    def dense_length(self, x):
      assert(len(x.shape) == 1)
      return tf.shape(x)[0]

    @test_utils.run_in_graph_and_eager_modes()
    def test_brle_logical_not(self):
      st.test_brle_logical_not(self)

    @test_utils.run_in_graph_and_eager_modes()
    def test_brle_to_dense(self):
      st.test_brle_to_dense(self)

    @test_utils.run_in_graph_and_eager_modes()
    def test_brle_length(self):
      st.test_brle_length(self)

    @test_utils.run_in_graph_and_eager_modes()
    def test_rle_length(self):
      st.test_rle_length(self)

    @test_utils.run_in_graph_and_eager_modes()
    def test_rle_to_dense(self):
      st.test_rle_to_dense(self)


if __name__ == '__main__':
  tf.test.main()
