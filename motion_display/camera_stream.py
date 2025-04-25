import math
import warnings

import cv2
import numpy as np
import torch


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


# data: {FoVx, FoVy, image_height, image_width, world_view_transform, full_proj_transform, camera_center}
class CameraStream:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.cap = None
        self.width = None
        self.height = None

        # hard-coded Intrinsic parameters
        self.fx, self.fy = 503.69234812, 503.95198528
        self.cx, self.cy = 324.57059016, 295.40637235

        self.zfar = 100.0
        self.znear = 0.01

        self.FovX, self.FovY = None, None

        self.K = np.array([[self.fx, 0., self.cx],
                           [0., self.fy, self.cy],
                           [0, 0, 1]], dtype=np.float32)
        self.distCoeffs = np.array([0.087944, -0.11943704, -0.00071627, -0.00221681, 0.00644727])
        self.camera_center = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera with id {self.camera_id}")
        else:
            print("Sucessfully opened camera with id {}".format(self.camera_id))

        # 获取相机的图像尺寸
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.FovY = focal2fov(self.fy, self.height)
        self.FovX = focal2fov(self.fx, self.width)

        self.camera_center = (self.width // 2, self.height // 2)

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar,
                                                     fovX=self.FovX, fovY=self.FovY).transpose(0,
                                                                                               1).cuda()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            warnings.warn("Error: Failed to capture image")
            raise StopIteration
        return frame


if __name__ == "__main__":
    with CameraStream(0) as camera:
        print(f"Camera resolution: {camera.width}x{camera.height}")  # 输出图像尺寸
        for image in camera:
            cv2.imshow('Camera', image)

            # 按 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
