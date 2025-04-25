import math
import warnings
from typing import Optional

import cv2
import numpy as np
import torch

from motion_display import CameraStream



def estimatePoseGridBoard(corners, ids, board, cameraMatrix, distCoeffs):
    if ids is None or len(ids) < 4:
        return None, None, False
    objPoints, imgPoints = board.matchImagePoints(corners, ids)
    if objPoints is None or len(objPoints) < 4:
        return None, None, False
    success, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
    return rvec, tvec, success  # rvec, tvec 是 **相机** 在 **Board** 坐标系下的位姿


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


class ChArucoStream(CameraStream):
    GRID_ROWS = 3
    GRID_COLS = 3
    MARKER_LENGTH = 0.05  # 5cm
    MARKER_SEPARATION = 0.007

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    grid_board = cv2.aruco.GridBoard((GRID_COLS, GRID_ROWS), MARKER_LENGTH, MARKER_SEPARATION, aruco_dict)

    @property
    def R(self) -> Optional[np.ndarray]:
        if self.rvec is None:
            return None
        R, _ = cv2.Rodrigues(self.rvec)
        return R

    @property
    def T(self) -> Optional[np.ndarray]:
        if self.tvec is None:
            return None
        return self.tvec.flatten()

    @property
    def world_view_transform(self):
        if self.rvec is None:
            return None
        # return getWorld2View(self.R, self.tvec)
        return torch.tensor(getWorld2View(self.R, self.tvec)).transpose(0, 1).cuda()

    def __init__(self, camera_id, visualize_detections=False):
        super().__init__(camera_id)
        self.tvec, self.rvec = None, None
        self.board_detected = False
        self.visualize_detections = visualize_detections

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            warnings.warn("Error: Failed to capture image")
            raise StopIteration

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markerCorners, markerIds, _ = ChArucoStream.aruco_detector.detectMarkers(gray)
        if markerIds is not None and len(markerIds) > 0:
            self.rvec, self.tvec, self.board_detected = estimatePoseGridBoard(markerCorners, markerIds, self.grid_board, self.K, self.distCoeffs)
            if self.visualize_detections and self.board_detected:
                cv2.drawFrameAxes(frame, self.K, self.distCoeffs, self.rvec, self.tvec, 0.1)
        else:
            self.rvec, self.tvec, self.board_detected = None, None, False
        return frame