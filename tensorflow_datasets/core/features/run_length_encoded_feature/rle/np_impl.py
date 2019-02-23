"""Numpy encode/decode/utility implementations for run length encodings."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def brle_length(brle):
  """Length of dense form of binary run-length-encoded input.

  Efficient implementation of `len(brle_to_dense(brle))`."""
  return np.sum(brle)


def rle_length(rle):
  """Length of dense form of run-length-encoded input.

  Efficient implementation of `len(rle_to_dense(rle))`."""
  return np.sum(rle[1::2])


def rle_to_brle(rle, dtype=None):
  """Convert run length encoded (RLE) value/counts to BRLE.

  RLE data is stored in a rank 1 array with each pair giving (value, count)

  e.g. the RLE encoding of [4, 4, 4, 1, 1, 6] is [4, 3, 1, 2, 6, 1].

  Args:
    rle: run length encoded data

  Returns:
    equivalent binary run length encoding. a list if dtype is None, otherwise
      brle_to_brle is called on that list before returning.

  Raises:
    ValueError if any of the even counts of `rle` are not zero or 1.
  """
  curr_val = 0
  out = [0]
  acc = 0
  for value, count in np.reshape(rle, (-1, 2)):
    acc += count
    if value not in (0, 1):
      raise ValueError("Invalid run length encoding for conversion to BRLE")
    if value == curr_val:
      out[-1] += count
    else:
      out.append(int(count))
      curr_val = value
  if len(out) % 2:
    out.append(0)
  if dtype is not None:
    out = brle_to_brle(out, dtype=dtype)
  out = maybe_pad_brle(out)
  return out


def maybe_pad_brle(lengths, start_value=False):
  """Get a potentially padded version of lengths.

  Args:
    lengths: rank 1 int array
    start_value: bool indicating value corresponding to the first value of
      lengths

  Returns:
    rank 1 array of same dtype as lengths, with an extra zero at the front
      if `start_value`, and an extra zero at the end if the resulting array
      would not have an even number of elements.
  """
  pad_left = int(start_value)
  pad_right = (len(lengths) + pad_left) % 2
  if pad_left + pad_right > 0:
    return np.pad(lengths, [pad_left, pad_right], mode='constant')
  else:
    return lengths


def merge_brle_lengths(lengths):
  """Inverse of split_long_brle_lengths."""
  if len(lengths) == 0:
    return []

  out = [int(lengths[0])]
  accumulating = False
  for length in lengths[1:]:
    if accumulating:
      out[-1] += length
      accumulating = False
    else:
      if length == 0:
        accumulating = True
      else:
        out.append(int(length))
  return maybe_pad_brle(out)


def split_long_brle_lengths(lengths, dtype=np.int64):
  """Split lengths that exceed max dtype value.

  Lengths `l` are converted into [max_val, 0] * l // max_val + [l % max_val]

  e.g. for dtype=np.uint8 (max_value == 255)
  ```
  split_long_brle_lengths([600, 300, 2, 6], np.uint8) == \
       [255, 0, 255, 0, 90, 255, 0, 45, 2, 6]
  ```
  """
  lengths = np.asarray(lengths)
  max_val = np.iinfo(dtype).max
  bad_length_mask = lengths > max_val
  if np.any(bad_length_mask):
    # there are some bad lenghs
    nl = len(lengths)
    repeats = np.asarray(lengths) // max_val
    remainders = (lengths % max_val).astype(dtype)
    lengths = np.empty(shape=(np.sum(repeats)*2 + nl,), dtype=dtype)
    np.concatenate(
      [np.array([max_val, 0] * repeat + [remainder], dtype=dtype)
       for repeat, remainder in zip(repeats, remainders)], out=lengths)
    return lengths
  elif lengths.dtype != dtype:
    return lengths.astype(dtype)
  else:
    return lengths


def dense_to_brle(dense_data, dtype=np.int64):
  """
  Get the binary run length encoding of `dense_data`.

  Args:
    dense_data: rank 1 bool array of data to encode.
    dtype: numpy int type.

  Returns:
    Binary run length encoded rank 1 array of dtype `dtype`.
  """
  if dense_data.dtype != np.bool:
    raise ValueError("`dense_data` must be bool")
  if len(dense_data.shape) != 1:
    raise ValueError("`dense_data` must be rank 1.")
  n = len(dense_data)
  starts = np.r_[0, np.flatnonzero(dense_data[1:] != dense_data[:-1]) + 1]
  lengths = np.diff(np.r_[starts, n])
  lengths = split_long_brle_lengths(lengths, dtype=dtype)
  return maybe_pad_brle(lengths, dense_data[0])


_ft = np.array([False, True], dtype=np.bool)


def dense_to_rle(dense_data, dtype=np.int64):
  """Get run length encoding of the provided dense data."""
  n = len(dense_data)
  starts = np.r_[0, np.flatnonzero(dense_data[1:] != dense_data[:-1]) + 1]
  lengths = np.diff(np.r_[starts, n])
  values = dense_data[starts]
  values, lengths = split_long_rle_lengths(values, lengths, dtype=dtype)
  out = np.stack((values, lengths), axis=1)
  return out.flatten()


def split_long_rle_lengths(values, lengths, dtype=np.int64):
  """Split long lengths in the associated run length encoding.

  e.g.
  ```python
  split_long_rle_lengths([5, 300, 2, 12], np.uint8) == [5, 255, 5, 45, 2, 12]
  ```

  Args:
    values: values column of run length encoding, or `rle[::2]`
    lengths: counts in run length encoding, or `rle[1::2]`
    dtype: numpy data type indicating the maximum value.

  Returns:
    values, lengths associated with the appropriate splits. `lengths` will be
    of type `dtype`, while `values` will be the same as the value passed in.
  """
  max_length = np.iinfo(dtype).max
  lengths = np.asarray(lengths)
  repeats = lengths // max_length
  if np.any(repeats):
    repeats += 1
    remainder = lengths % max_length
    values = np.repeat(values, repeats)
    lengths = np.empty(len(repeats), dtype=dtype)
    lengths.fill(max_length)
    lengths = np.repeat(lengths, repeats)
    lengths[np.cumsum(repeats)-1] = remainder
  elif lengths.dtype != dtype:
    lengths = lengths.astype(dtype)
  return values, lengths


def merge_rle_lengths(values, lengths):
  """Inverse of split_long_rle_lengths except returns normal python lists."""
  ret_values = []
  ret_lengths = []
  curr = None
  for v, l in zip(values, lengths):
    if l == 0:
      continue
    if v == curr:
      ret_lengths[-1] += l
    else:
      curr = v
      ret_lengths.append(int(l))
      ret_values.append(v)
  return ret_values, ret_lengths


def brle_to_rle(brle, dtype=np.int64):
  lengths = brle
  values = np.tile(_ft, len(brle) // 2)
  return rle_to_rle(np.stack((values, lengths), axis=1).flatten(), dtype=dtype)


def brle_to_brle(brle, dtype=np.int64):
  """Almost the identity function.

  Checks for possible merges and required splits.
  """
  return split_long_brle_lengths(merge_brle_lengths(brle), dtype=dtype)


def rle_to_rle(rle, dtype=np.int64):
  """Almost the identity function.

  Checks for possible merges and required splits.
  """
  v, l = np.reshape(rle, (-1, 2)).T
  v, l = merge_rle_lengths(v, l)
  v, l = split_long_rle_lengths(v, l, dtype=dtype)
  return np.stack((v, l), axis=1).flatten()
