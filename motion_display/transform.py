import math

import cv2
import numpy as np
import torch


def apply_transform(pts3d, *, rvec=None, R=None, tvec=None, scale=None):
    assert isinstance(pts3d, np.ndarray)

    if len(pts3d.shape) == 3:  # (B, N, 3)
        B, N, _ = pts3d.shape
        pts3d = pts3d.reshape(-1, 3)
    elif len(pts3d.shape) == 2:
        N, _ = pts3d.shape
        B = None
    else:
        raise TypeError("pts3d must be either 2 or 3 dimensional")

    if scale is not None:
        pts3d = pts3d * scale

    if rvec is not None:
        rvec = rvec.flatten()
        assert rvec.shape == (3,)
        R, _ = cv2.Rodrigues(rvec)  # (3, 3)
        pts3d = (R @ pts3d.T).T
    elif R is not None:
        assert R.shape == (3, 3)
        pts3d = (R @ pts3d.T).T

    if tvec is not None:
        tvec = tvec.flatten()
        assert tvec.shape == (3,)
        pts3d += tvec

    if B is not None:
        pts3d = pts3d.reshape((B, N, 3))
    return pts3d


# Rotation matrix around an arbitrary axis
def get_rotation_matrix(axis, theta):
    # Ensure the axis is a unit vector
    axis = axis / np.linalg.norm(axis)

    # Components of the axis
    x, y, z = axis[0], axis[1], axis[2]

    # Compute the rotation matrix using Rodrigues' rotation formula
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    one_minus_cos_theta = 1 - cos_theta

    # Rotation matrix components
    rotation_matrix = np.array([
        [cos_theta + x * x * one_minus_cos_theta, x * y * one_minus_cos_theta - z * sin_theta,
         x * z * one_minus_cos_theta + y * sin_theta],
        [y * x * one_minus_cos_theta + z * sin_theta, cos_theta + y * y * one_minus_cos_theta,
         y * z * one_minus_cos_theta - x * sin_theta],
        [z * x * one_minus_cos_theta - y * sin_theta, z * y * one_minus_cos_theta + x * sin_theta,
         cos_theta + z * z * one_minus_cos_theta]
    ])

    return rotation_matrix