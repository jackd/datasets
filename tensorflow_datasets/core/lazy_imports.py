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

"""Lazy imports for heavy dependencies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib

from tensorflow_datasets.core.utils import py_utils as utils
from distutils import version


def _try_import(module_name, min_version=None):
  """Try importing a module, with an informative error message on failure."""
  try:
    mod = importlib.import_module(module_name)
    if (min_version is not None and
        version.LooseVersion(mod.__version__) < min_version):
      raise ImportError(
        "Installed version of %s does not satisfy minimum version (%s). "
        "See setup.py extras_require. The dataset you are trying to use may "
        "have additional dependencies." % (module_name, min_version))
    return mod
  except ImportError:
    err_msg = ("Tried importing %s but failed. See setup.py extras_require. "
               "The dataset you are trying to use may have additional "
               "dependencies." % module_name)
    utils.reraise(err_msg)


class LazyImporter(object):
  """Lazy importer for heavy dependencies.

  Some datasets require heavy dependencies for data generation. To allow for
  the default installation to remain lean, those heavy depdencies are
  lazily imported here.
  """

  @utils.classproperty
  @classmethod
  def cv2(cls):
    return _try_import("cv2")  # pylint: disable=unreachable

  @utils.classproperty
  @classmethod
  def h5py(cls):
    # issues with h5py.File(fp, "r") for earlier versions
    return _try_import("h5py", min_version="2.9.0")

  @utils.classproperty
  @classmethod
  def pydub(cls):
    return _try_import("pydub")

  @utils.classproperty
  @classmethod
  def matplotlib(cls):
    return _try_import("matplotlib")

  @utils.classproperty
  @classmethod
  def PIL_Image(cls):   # pylint: disable=invalid-name
    # TiffImagePlugin need to be activated explicitly on some systems
    # https://github.com/python-pillow/Pillow/blob/5.4.x/src/PIL/Image.py#L407
    _try_import("PIL.TiffImagePlugin")
    return _try_import("PIL.Image")

  @utils.classproperty
  @classmethod
  def pyplot(cls):
    return _try_import("matplotlib.pyplot")

  @utils.classproperty
  @classmethod
  def scipy(cls):
    return _try_import("scipy")

  @utils.classproperty
  @classmethod
  def scipy_io(cls):
    return _try_import("scipy.io")

  @utils.classproperty
  @classmethod
  def os(cls):
    """For testing purposes only."""
    return _try_import("os")

  @utils.classproperty
  @classmethod
  def test_foo(cls):
    """For testing purposes only."""
    return _try_import("test_foo")


lazy_imports = LazyImporter  # pylint: disable=invalid-name
