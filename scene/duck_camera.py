import numpy as np
import torch

from common.utils import estimate_focal_length
from utils.graphics_utils import focal2fov, getProjectionMatrix


def getWorld2View2(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    """get World 2 View Tensor"""
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = Rt.inverse()
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = C2W.inverse()
    return Rt.float()


class Camera:
    def __init__(self, R, T, height=480, width=640, fx=None, fy=None, K=None):
        self.image_height = height  # 480
        self.image_width = width  # 640

        if K is not None:
            self.fx = K[0, 0]
            self.fy = K[1, 1]
            self.cx = K[0, 2]
            self.cy = K[1, 2]
            self.K = K
        else:
            if fx is None or fy is None:
                self.fx = self.fy = estimate_focal_length(height, width)
            else:
                self.fx = fx
                self.fy = fy
            self.cx, self.cy = width / 2, height / 2  # center estimated
            self.K = np.array([[self.fx, 0, self.cx],
                               [0, self.fy, self.cy],
                               [0, 0, 1]], dtype=np.float32)

        self.FoVx = focal2fov(self.fx, self.image_width)
        self.FoVy = focal2fov(self.fy, self.image_height)
        self.zfar = 100.0
        self.znear = 0.01

        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).float().cuda()
        if isinstance(T, np.ndarray):
            T = torch.from_numpy(T).float().cuda()

        self.R = R  # R mat (3, 3)
        self.T = T  # T vec (3, )

        # word-view transform
        Rt = torch.zeros((4, 4), dtype=torch.float32).cuda()
        Rt[:3, :3] = R  # ?
        Rt[3, :3] = T
        Rt[3, 3] = 1.0
        self.world_view_transform = Rt

        # projection matrix
        self.projection_matrix = getProjectionMatrix(znear=self.znear,
                                                     zfar=self.zfar,
                                                     fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        # full projection matrix
        self.full_proj_transform = \
            torch.bmm(self.world_view_transform.unsqueeze(0),
                      self.projection_matrix.unsqueeze(0)).squeeze(0)

        # camera center
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.frame_id = -1

    def update_pose(self, rots, Jtrs, bone_transforms):
        self.rots = rots
        self.Jtrs = Jtrs
        self.bone_transforms = bone_transforms

    def __getattr__(self, name):
        if name in self.data:
            return self.data[name]
        else:
            raise AttributeError(name)
