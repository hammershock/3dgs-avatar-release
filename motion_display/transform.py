import cv2
import numpy as np


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



