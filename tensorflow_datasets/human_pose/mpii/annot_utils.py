from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import numpy as np
import tensorflow as tf
from tensorflow_datasets.core import lazy_imports
from collections import Sequence


class MappedSequence(Sequence):
  def __init__(self, base, map_fn):
    self._base = base
    self._map_fn = map_fn

  def __len__(self):
    return len(self._base)

  def __getitem__(self, index):
    return self._map_fn(self._base[index])


lazy_map = MappedSequence

# Part info
_parts = ['rank', 'rkne', 'rhip',
         'lhip', 'lkne', 'lank',
         'pelv', 'thrx', 'neck', 'head',
         'rwri', 'relb', 'rsho',
         'lsho', 'lelb', 'lwri']
_part_indices = {k: i for i, k in enumerate(_parts)}

NUM_JOINTS = len(_parts)
assert(NUM_JOINTS == 16)


def _xy(part_info):
  return part_info['x'][0][0], part_info['y'][0][0]


class CoordHelper(object):
  def __init__(self, annorect):
    self._annorect = annorect
    self.num_people = len(annorect[0]) if len(annorect) > 0 else 0

    fields = annorect.dtype.fields
    self.is_pointless = not (
        annorect.size > 0 and fields is not None and 'scale' in fields)

    if self.is_pointless:
      self.valid = np.zeros((self.num_people,), dtype=np.bool)
      self.num_valid_people = 0
      self.reorder_indices = np.arange(self.num_people, dtype=np.int64)
    else:
      anno_scale = self._annorect['scale'][0]
      self.valid = np.array([s.size == 1 for s in anno_scale], dtype=np.bool)
      self.num_valid_people = np.sum(self.valid.astype(np.int64))
      x = np.arange(self.num_people, dtype=np.int64)
      self.reorder_indices = np.concatenate(
          [x[self.valid], x[np.logical_not(self.valid)]], axis=0)

  def reindex(self, original_indices):
    assert(len(original_indices.shape) == 1)
    x = np.zeros((self.num_people,), dtype=np.bool)
    x[original_indices] = True
    x = x[self.reorder_indices]
    out = np.where(x)[0]
    return out

  @property
  def scales(self):
    scales = np.zeros((self.num_valid_people,), dtype=np.float32)
    if self.is_pointless:
      return scales
    anno_scale = self._annorect['scale'][0]
    for i, s in enumerate(anno_scale[self.valid]):
      scales[i] = s[0][0]
    return scales

  @property
  def centers(self):
    centers = np.zeros((self.num_valid_people, 2), dtype=np.int64)
    if self.is_pointless:
      return centers

    objpos = self._annorect['objpos'][0]
    for i, pos in enumerate(objpos[self.valid]):
      if pos.size > 0:
        centers[i] = _xy(pos)
    return centers

  @property
  def coordinates(self):
    num_valid_people = self.num_valid_people

    xy = np.zeros((num_valid_people, NUM_JOINTS, 2), dtype=np.int64)
    visible = np.zeros((num_valid_people, NUM_JOINTS), dtype=np.bool)
    if self.is_pointless:
      return xy, visible

    annopoints = self._annorect['annopoints'][0]

    def update_vis(parts_info, vis):
      if 'is_visible' in parts_info.dtype.fields:
        for part_info in parts_info:
          v = part_info['is_visible']
          v = v[0][0] if len(v) > 0 else 1
          if isinstance(v, six.string_types):
            v = int(v)
          j = np.squeeze(part_info['id'])
          vis[j] = bool(v)
      else:
        vis[:] = True

    def update_xy(parts_info, xy):
      for part_info in parts_info:
        j = np.squeeze(part_info['id'])
        xy[j] = _xy(part_info)

    for i, points in enumerate(annopoints[self.valid]):
      parts_info = points[0][0][0][0]
      update_xy(parts_info, xy[i])
      update_vis(parts_info, visible[i])
    return xy, visible

  @property
  def head_boxes(self):
    if self.is_pointless:
      return np.zeros((self.num_people, 4), dtype=np.int64)

    coords = tuple(self._annorect[k][0] for k in ('y1', 'x1', 'y2', 'x2'))

    all_coords = np.array([
      [c[p][0][0] for c in coords] for p in range(self.num_people)],
      dtype=np.int64)
    valid_coords = all_coords[self.valid]
    invalid_coords = all_coords[np.logical_not(self.valid)]
    return np.concatenate([valid_coords, invalid_coords], axis=0)


class Annotations(object):
  def __init__(self, path):
    with tf.io.gfile.GFile(path, "rb") as fp:
      self._annot = lazy_imports.scipy_io.loadmat(fp)['RELEASE']  # pylint: disable=no-member

    self._annolist = self._annot['annolist'][0][0][0]
    self._annorect = self._annolist['annorect']

    self._is_train = np.array(self._annot['img_train'][0][0][0]).astype(np.bool)
    self._num_examples = len(self._is_train)

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def original_separated_indices(self):
    def f(ind):
      ind = ind[0]
      if ind.size == 0:
        return np.zeros((0,), dtype=np.int64)
      else:
        return np.squeeze(np.array(ind, dtype=np.int64), axis=1) - 1

    return lazy_map(
      self._annot['single_person'][0][0], f)

  @property
  def video_indices(self):
    def f(vid):
      vid = np.squeeze(vid)
      return None if vid.size == 0 else vid - 1

    return lazy_map(self._annolist['vididx'], f)

  @property
  def frame_secs(self):
    def f(sec):
      sec = np.squeeze(sec)
      return None if sec.size == 0 else sec.astype(np.int64)
    return lazy_map(self._annolist['frame_sec'], f)

  @property
  def activities(self):
    def fn(act):
      a = act[0]
      names = a['cat_name'], a['act_name']

      cat_name, act_name = (
        None if n is None or len(n) == 0 else n[0] for n in names)
      # return act_name, cat_name
      return cat_name, act_name
    return lazy_map(self._annot['act'][0][0], fn)

  @property
  def filenames(self):
    return lazy_map(self._annolist['image'], lambda v: v[0]['name'][0][0])

  @property
  def is_train(self):
    return self._is_train

  @property
  def youtube_ids(self):
    vl = self._annot['video_list'][0][0][0]
    return lazy_map(vl, lambda x: x[0])

  @property
  def coord_helpers(self):
    return lazy_map(self._annorect, CoordHelper)


def normalizations(head_box):
  # Get head height for distance normalization
  s = head_box.shape
  assert(s[-1]) == 4
  head_box = np.reshape(head_box, s[:-1] + (2, 2))
  return np.linalg.norm(head_box[..., 0, :] - head_box[..., 1, :]) * 0.6


def torso_angle(joints):
  pt1 = joints[..., _part_indices['pelv'], :]
  pt2 = joints[..., _part_indices['thrx'], :]
  if (pt1[0] == 0 or pt2[0] == 0):
    return np.zeros(shape=joints.shape[:-2], dtype=np.float32)
  else:
    diff = pt2 - pt1
    return 90 + np.arctan2(diff[..., 1], diff[..., 0]) * 180. / np.pi


def get_names_lists(annot):
  cat_names, act_names = zip(*annot.activities)
  act_names = [
    (c, a) for (c, a) in zip(cat_names, act_names)
    if c is not None and a is not None]
  cat_names = set(cat_names)
  if None in cat_names:
    cat_names.remove(None)
  # activity names partitioned based on the sorted category names

  act_names = sorted(set(act_names), key=lambda a: '%s~%s' % a)
  act_names = [a[1] for a in act_names]
  cat_names = sorted(set(cat_names))
  cat_names = [c for c in cat_names]
  return cat_names, act_names
