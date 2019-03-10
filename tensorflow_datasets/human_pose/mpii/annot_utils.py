"""Utilities for converting mpii annotations file, based on
https://github.com/princeton-vl/pose-hg-train/blob/master/src/misc/mpii.py
"""
import six
import numpy as np
import tensorflow as tf
from tensorflow_datasets.core import lazy_imports


# Part info
_parts = ['rank', 'rkne', 'rhip',
         'lhip', 'lkne', 'lank',
         'pelv', 'thrx', 'neck', 'head',
         'rwri', 'relb', 'rsho',
         'lsho', 'lelb', 'lwri']
_part_indices = {k: i for i, k in enumerate(_parts)}


class Annotations(object):
  def __init__(self, path):
    with tf.io.gfile.GFile(path, "rb") as fp:
      self._annot = lazy_imports.scipy_io.loadmat(fp)['RELEASE']    # pylint: disable=no-member

  @property
  def num_images(self):
    return self._annot['img_train'][0][0][0].shape[0]

  def image_subpath(self, idx):
    filename = str(self._annot['annolist'][0][0][0]['image'][idx][0]['name'][0][0])
    return filename

  def num_people(self, idx):
    # Get number of people present in image
    example = self._annot['annolist'][0][0][0]['annorect'][idx]
    if len(example) > 0:
        return len(example[0])
    else:
        return 0

  def is_train(self, idx):
    # Return true if image is in training set
    return (
      self._annot['img_train'][0][0][0][idx] and
      self._annot['annolist'][0][0][0]['annorect'][idx].size > 0 and
      'annopoints' in self._annot['annolist'][0][0][0]['annorect'][idx].dtype.fields)

# def locations(self, idx, num_people=None):
#   if num_people is None:
#     num_people = self.num_people(idx)
#   centers = np.zeros((num_people, 2), dtype=np.int64)
#   scales = np.zeros((num_people,), dtype=np.float32)
#   example = self._annot['annolist'][0][0][0]['annorect'][idx]
#   example_scale = example['scale'][0]
#   example_objpos = example['objpos'][0]
#   for person in range(num_people):
#     person_scale =
#     if ((not example.dtype.fields is None) and
#       'scale' in example.dtype.fields and
#       example_scale[person].size > 0 and
#       example_objpos[person].size > 0):
#       scale = example_scale[person][0][0]
#       xy = example_objpos[person][0][0]
#       x = xy['x'][0][0]
#       y = xy['y'][0][0]
#       center = (x, y)
#     else:
#       center = [-1, -1]
#       scale = -1
#     centers[person] = center
#     scales[person] = scale
#   return centers, scales

def location(self, idx, person):
  # Return center of person, and scale factor
  example = self._annot['annolist'][0][0][0]['annorect'][idx]
  if ((not example.dtype.fields is None) and
    'scale' in example.dtype.fields and
    example['scale'][0][person].size > 0 and
    example['objpos'][0][person].size > 0):
    scale = example['scale'][0][person][0][0]
    xy = example['objpos'][0][person][0][0]
    x = xy['x'][0][0]
    y = xy['y'][0][0]
    return np.array([x, y]), scale
  else:
    return [-1, -1], -1

def part_info(self, idx, person, part):
  # Part location and visibility
  # This function can take either the part name or the index of the part
  if isinstance(part, six.string_types):
    part = _part_indices[part]

  example = self._annot['annolist'][0][0][0]['annorect'][idx]
  if example['annopoints'][0][person].size > 0:
    parts_info = example['annopoints'][0][person][0][0][0][0]
    for i in range(len(parts_info)):
      if parts_info[i]['id'][0][0] == part:
        if 'is_visible' in parts_info.dtype.fields:
          v = parts_info[i]['is_visible']
          v = v[0][0] if len(v) > 0 else 1
          if isinstance(v, six.string_types):
            v = int(v)
        else:
          v = 1
        return np.array(
          [parts_info[i]['x'][0][0], parts_info[i]['y'][0][0]], int), v
    return np.zeros(2, int), 0
  return -np.ones(2, int), -1

def normalization(self, idx, person):
    # Get head height for distance normalization
    if self.is_train(idx):
        example = self._annot['annolist'][0][0][0]['annorect'][idx]
        x1 = int(example['x1'][0][person][0][0])
        y1 = int(example['y1'][0][person][0][0])
        x2, y2 = int(example['x2'][0][person][0][0]), int(example['y2'][0][person][0][0])
        diff = np.array([y2 - y1, x2 - x1], np.float)
        return np.linalg.norm(diff) * .6
    return -1

def torso_angle(self, idx, person):
    # Get angle from pelvis to thorax, 0 means the torso is up vertically
    pt1 = self._part_info(idx, person, 'pelv')[0]
    pt2 = self._part_info(idx, person, 'thrx')[0]
    if not (pt1[0] == 0 or pt2[0] == 0):
        return 90 + np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) * 180. / np.pi
    else:
        return 0
