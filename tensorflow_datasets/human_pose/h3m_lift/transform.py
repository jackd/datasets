"""
Utilities to deal with the cameras of human3.6m

Based almost entirely on original repository version of same name, with
changes for consistency.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


# def project_point_radial_original(P, f, c, k, p):
#   """
#   Project points from 3d to 2d using camera parameters
#   including radial and tangential distortion

#   Args
#     P: Nx3 points in world coordinates
#     f: 2x1 Camera focal length
#     c: 2x1 Camera center
#     k: 3x1 Camera radial distortion coefficients
#     p: 2x1 Camera tangential distortion coefficients
#   Returns
#     Proj: Nx2 points in pixel space
#     D: 1xN depth of each point in camera space
#     radial: 1xN radial distortion per point
#     tan: 1xN tangential distortion per point
#     r2: 1xN squared radius of the projected points before distortion
#   """

#   # P is a matrix of 3-dimensional points
#   assert(len(P.shape) == 2)
#   assert(P.shape[1] == 3)

#   N = P.shape[0]
#   X = P.T
#   XX = X[:2, :] / X[2, :]
#   r2 = XX[0, :]**2 + XX[1, :]**2

#   # r_pows = np.array([r2, r2**2, r2**3])
#   # r22 = r2*r2
#   # r23 = r22*r2
#   r22 = r2**2
#   r23 = r2**3
#   r_pows = np.array([r2, r22, r23])

#   radial = 1 + np.einsum(
#     'ij,ij->j', np.tile(k, (1, N)), r_pows)
#   tan = p[0]*XX[1, :] + p[1]*XX[0, :]

#   XXX = XX * np.tile(
#     radial + tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)

#   Proj = (f * XXX) + c
#   Proj = Proj.T

#   D = X[2,]

#   return Proj, D, radial, tan, r2


def project_points(
    points, focal_length, center, radial_dist_coeff, tangential_dist_coeff):
  """
  Project points from 3d to 2d using camera parameters with distortion.

  Args
    points: (N, 3) points in world coordinates
    focal_length: (2,) Camera focal length
    center: (2,) Camera center
    radial_dist_coeff: (3,) Camera radial distortion coefficients
    tangential_dist_coeff: (2,) Camera tangential distortion coefficients

  Returns
    proj: (N, 2) points in pixel space
    depth: (N,) depth of each point in camera space
    radial_dist: 1xN radial distortion per point
    tangential_dist: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
  """

  assert(len(points.shape) == 2)
  assert(points.shape[1] == 3)
  points = points.copy()

  # pylint: disable=unbalanced-tuple-unpacking
  XX, depth = np.split(points, (2,), axis=-1)
  XX /= depth
  depth = np.squeeze(depth, axis=-1)
  r2 = np.sum(np.square(XX), axis=-1)

  r22 = r2*r2
  r23 = r22*r2
  r_pows = np.stack((r2, r22, r23), axis=-1)
  radial_dist = 1 + np.sum(radial_dist_coeff[..., -1::-1] * r_pows, axis=-1)
  tangential_dist = np.sum(tangential_dist_coeff*XX, axis=1)

  XXX = XX * (
    np.expand_dims(radial_dist + tangential_dist, axis=-1) +
    np.expand_dims(r2, axis=-1) * np.expand_dims(
      tangential_dist_coeff, axis=-2))

  proj = (focal_length * XXX) + center

  return proj, depth, radial_dist, tangential_dist, r2


def _validate_transform_shapes(points, rotation, translation):
  shape = points.shape
  if len(shape) != 2 or shape[1] != 3:
    raise ValueError("points must have shape (N, 3), got %s" % shape)
  shape = rotation.shape
  if shape != (3, 3):
    raise ValueError("rotation must have shape (3, 3), got %s" % shape)
  shape = translation.shape
  if shape != (3,):
    raise ValueError("translation must have shape (3,), got %s" % shape)


def world_to_camera_frame(points_world, rotation, translation):
  """
  Convert points from world to camera coordinates

  Args
    points_world: (N, 3) 3d points in world coordinates
    rotation: (3, 3) Camera rotation matrix
    translation: (3,) Camera translation parameters
  Returns
    points_cam: (N, 3) 3d points in camera coordinates
  """
  _validate_transform_shapes(points_world, rotation, translation)
  return (points_world - translation).dot(rotation.T)


def camera_to_world_frame(points_camera, rotation, translation):
  """Inverse of world_to_camera_frame

  Args
    points_camera: (N, 3) points in camera coordinates
    rotation: (3, 3) Camera rotation matrix
    translation: (3,) Camera translation parameters
  Returns
    points_world: (N, 3) points in world coordinates
  """
  _validate_transform_shapes(points_camera, rotation, translation)
  return points_camera.dot(rotation) + translation
