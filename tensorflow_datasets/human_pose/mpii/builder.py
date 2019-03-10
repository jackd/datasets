from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.human_pose.mpii.skeleton import s16
from tensorflow_datasets.human_pose.mpii import annot_utils
from tensorflow_datasets.core import lazy_imports
# TODO(jackd) MpiiBaseConfig

NUM_JOINTS = s16.num_joints

DL_IMAGES_URL = "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz"
DL_ANNOT_URL = "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip"

MPII_URL = "http://human-pose.mpi-inf.mpg.de/"

_hg_annot = "https://github.com/princeton-vl/pose-hg-demo/raw/master/annot/%s.h5"
_DL_URLS = {
    "images": DL_IMAGES_URL,
    "hg_train": _hg_annot % "train",
    "hg_test": _hg_annot % "test",
    "hg_validation": _hg_annot % "valid",
    "original_annot": DL_ANNOT_URL,
}

def _download_and_extract(dl_manager):
  paths = dl_manager.download(_DL_URLS)
  images_dir = dl_manager.extract(paths["images"])
  images_dir = os.path.join(images_dir, "images")
  return paths, images_dir


# class MpiiBaseConfig(tfds.core.BuilderConfig):
#   def __init__(self):
#     super(MpiiBaseConfig, self).__init__(
#       name="base",
#       version=tfds.core.Version(0, 0, 1),
#       desciption="Multi-person dataset"
#     )

#   def _info(self, builder):
#     return tfds.core.DatasetInfo(
#       builder=builder,
#       description="Human-annotated 2D human poses on in-the-wild images",
#       features=tfds.features.FeaturesDict({
#         "image": tfds.features.Image(encoding_format="jpeg"),
#         "pose_2d": tfds.features.Tensor(shape=(NUM_JOINTS, 2), dtype=tf.int64),
#         "visible": tfds.features.Tensor(shape=(NUM_JOINTS,), dtype=tf.bool),
#         "center": tfds.features.Tensor(shape=(2,), dtype=tf.int64),
#         "scale": tfds.features.Tensor(shape=(), dtype=tf.float32),
#         "torso_angle": tfds.features.Tensor(shape=(), dtype=tf.float32),
#         "normalize": tfds.features.Tensor(shape=(), dtype=tf.float32),
#         "person_index": tfds.features.Tensor(shape=(), dtype=tf.int64),
#         "filename": tfds.features.Text(),
#       }),
#       urls=[MPII_URL],
#       citation=MPII_CITATION
#     )

#   def _split_generators(self, dl_manager):
#     paths = dl_manager.download(_DL_URLS)
#     images_dir = dl_manager.extract(paths["images"])
#     images_dir = os.path.join(images_dir, "images")

#     paths, images_dir = _download_and_extract(dl_manager)
#     annot=annot_utils.Annotations(paths["original_annotations"])

#     return [
#       tfds.core.SplitGenerator(
#         gen_kwargs=dict(
#           annot=annot,
#           images_dir=images_dir,
#           is_train=True
#         )),
#       tfds.core.SplitGenerator(
#         name=tfds.Split.TEST,
#         num_shards=8,
#         gen_kwargs=dict(
#           annot=annot,
#           images_dir=images_dir,
#           is_train=False
#         ))
#     ]

#   def _generate_examples(self, annot, images, is_train):
#     for idx in annot.num_images:
#       example_is_train = annot.is_train(idx)
#       if example_is_train != is_train:
#         continue

#       # per example
#       video_index = ...

#       example = dict(example_index=idx)


#       # per person
#       num_people = mpii.num_people(idx)
#       center = []
#       scale = []
#       visible = []
#       head_rect = []
#       position = []
#       pose_2d = []
#       visible = []


#       for person in range(num_people):
#         center, scale = mpii.location(idx, person)
#         if center[0] != -1:
#           # valid example

#       example['centers'] = np.array(centers, dtype=np.int64)
#       example['scales'] = np.array(scales, dtype=np.int64)

#       yield example


class MpiiHourglassConfig(tfds.core.BuilderConfig):
  """For conversion details, see
  https://github.com/princeton-vl/pose-hg-train/blob/master/src/misc/convert_annot.py"""
  def __init__(self):
    super(MpiiHourglassConfig, self).__init__(
      name="hourglass",
      version=tfds.core.Version(0, 0, 1),
      description="Single pose per example version as provided by Newell et al."
    )

  def _info(self, builder):
    return tfds.core.DatasetInfo(
      builder=builder,
      description="Human-annotated 2D human poses on in-the-wild images",
      features=tfds.features.FeaturesDict({
        "image": tfds.features.Image(encoding_format="jpeg"),
        "pose_2d": tfds.features.Tensor(shape=(NUM_JOINTS, 2), dtype=tf.int64),
        "visible": tfds.features.Tensor(shape=(NUM_JOINTS,), dtype=tf.bool),
        "center": tfds.features.Tensor(shape=(2,), dtype=tf.int64),
        "scale": tfds.features.Tensor(shape=(), dtype=tf.float32),
        "torso_angle": tfds.features.Tensor(shape=(), dtype=tf.float32),
        "normalize": tfds.features.Tensor(shape=(), dtype=tf.float32),
        "person_index": tfds.features.Tensor(shape=(), dtype=tf.int64),
        "filename": tfds.features.Text(),
      }),
      urls=[
        MPII_URL,
        "http://www-personal.umich.edu/~alnewell/pose/",
      ],
      citation='\n'.join((MPII_CITATION, HOURGLASS_CITATION))
    )

  def _split_generators(self, dl_manager):
    paths, images_dir = _download_and_extract(dl_manager)

    return [
      tfds.core.SplitGenerator(
        name=tfds.Split.TRAIN,
        num_shards=32,
        gen_kwargs=dict(
          annot_path=paths["hg_train"],
          images_dir=images_dir,
        )),
      tfds.core.SplitGenerator(
        name=tfds.Split.VALIDATION,
        num_shards=4,
        gen_kwargs=dict(
          annot_path=paths["hg_validation"],
          images_dir=images_dir,
        )),
        tfds.core.SplitGenerator(
        name=tfds.Split.TEST,
        num_shards=8,
        gen_kwargs=dict(
          annot_path=paths["hg_test"],
          images_dir=images_dir,
        ))
    ]

  def _generate_examples(self, annot_path, images_dir):
    with tf.io.gfile.GFile(annot_path, "rb") as fobj:
      annots = lazy_imports.h5py.File(fobj, "r") # pylint: disable=no-member
      annots = dict(
        pose_2d=np.array(annots["part"], dtype=np.int64),
        visible=np.array(annots["visible"]).astype(np.bool),
        center=np.array(annots["center"], dtype=np.int64),
        scale=np.array(annots["scale"], dtype=np.float32),
        torso_angle=np.array(annots["torsoangle"], dtype=np.float32),
        normalize=np.array(annots["normalize"], dtype=np.float32),
        person_index=np.array(annots["person"], dtype=np.int64),
        filename=list(annots["imgname"]),
      )

    num_examples = annots["pose_2d"].shape[0]

    for i in range(num_examples):
      example = {k: v[i] for k, v in annots.items()}
      example["image"] = os.path.join(images_dir, example["filename"])
      yield example


HOURGLASS_CONFIG = MpiiHourglassConfig()


HOURGLASS_CITATION = """\
  @inproceedings{newell2016stacked,
  title={Stacked hourglass networks for human pose estimation},
  author={Newell, Alejandro and Yang, Kaiyu and Deng, Jia},
  booktitle={European Conference on Computer Vision},
  pages={483--499},
  year={2016},
  organization={Springer}
}"""

MPII_CITATION = """\
@inproceedings{andriluka14cvpr,
        author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt}
        title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2014},
        month = {June}
}"""


class MpiiHumanPose(tfds.core.GeneratorBasedBuilder):
  """Required implementations moved to config.

  This allows reuse of downloaded data (particularly images).
  """
  BUILDER_CONFIGS = [HOURGLASS_CONFIG]

  def _info(self):
    return self.builder_config._info(self)

  def _split_generators(self, dl_manager):
    return self.builder_config._split_generators(dl_manager)

  def _generate_examples(self, **kwargs):
    return self.builder_config._generate_examples(**kwargs)
