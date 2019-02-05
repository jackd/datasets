from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

from tensorflow_datasets.core.features.rle_feature.rle import np_impl
import tensorflow_datasets.core.features.rle_feature.rle.shared_tests as st


class BlreNpTest(unittest.TestCase):
  @property
  def impl(self):
    return np_impl

  def assert_array_equal(self, x, y, *args, **kwargs):
    return np.testing.assert_array_equal(x, y, *args, **kwargs)

  def dense_logical_not(self, x):
    return np.logical_not(x)

  def dense_length(self, x):
    assert(len(x.shape) == 1)
    return x.shape[0]

  def evaluate(self, x):
    return x

  # RLE tests
  def test_merge_rle_lengths(self):
    st.test_merge_rle_lengths(self)

  def test_split_long_rle_lengths(self):
    st.test_split_long_rle_lengths(self)

  def test_rle_length(self):
    st.test_rle_length(self)

  def test_rle_to_brle(self):
    st.test_rle_to_brle(self)

  def test_rle_to_dense(self):
    st.test_rle_to_dense(self)

  def test_rle_encode_decode(self):
    st.test_rle_encode_decode(self)

  def test_rle_decode_encode(self):
    st.test_rle_decode_encode(self)

  # BRLE tests
  def test_brle_logical_not(self):
    st.test_brle_logical_not(self)

  def test_brle_length(self):
    st.test_brle_length(self)

  def test_maybe_pad_brle(self):
    st.test_maybe_pad_brle(self)

  def test_merge_brle_lengths(self):
    st.test_merge_brle_lengths(self)

  def test_split_long_brle_lengths(self):
    st.test_split_long_brle_lengths(self)

  def test_brle_split_merge(self):
    st.test_brle_split_merge(self)

  def test_dense_to_brle(self):
    st.test_dense_to_brle(self)

  def test_brle_to_dense(self):
    st.test_brle_to_dense(self)

  def test_brle_to_rle(self):
    st.test_brle_to_rle(self)

  def test_brle_encode_decode(self):
    st.test_brle_encode_decode(self)

  def test_brle_decode_encode(self):
    st.test_brle_decode_encode(self)


if __name__ == '__main__':
  unittest.main()
