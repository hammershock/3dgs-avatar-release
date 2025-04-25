import os.path
from typing import NamedTuple

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation


class SMPLParameters(NamedTuple):
    pose: np.ndarray  # 姿态参数
    shape: np.ndarray  # 形状参数
    global_t: np.ndarray  # smpl模型全局平移
    pred_joints: np.ndarray  #
    focal_l: np.ndarray
    Jtrs: torch.Tensor
    rots: torch.Tensor
    bone_transforms: torch.Tensor
    vertices: torch.Tensor
    Jtr_posed: torch.Tensor
    root_orient: torch.Tensor
    bone_transforms_orig: torch.Tensor

    def export(self, out_model_path):
        betas = torch.from_numpy(self.shape).float().cuda().unsqueeze(0)
        body = body_model(betas=betas)  # Get shape vertices
        minimal_shape = body.v.detach().cpu().numpy()[0]  # 1. minimal shape in T pose, given betas
        poses = self.pose[None, ...]
        pose_body = poses[:, 3:66].copy()  #
        pose_hand = poses[:, 66:].copy()  #

        np.savez(out_model_path,
                 minimal_shape=minimal_shape,
                 betas=betas.cpu().detach().numpy(),
                 Jtr_posed=self.Jtr_posed.detach().squeeze().cpu().numpy(),
                 bone_transforms=self.bone_transforms_orig.squeeze().detach().cpu().numpy(),
                 trans=self.global_t,
                 root_orient=self.root_orient[0].detach().cpu().numpy(),
                 pose_body=pose_body[0],
                 pose_hand=pose_hand[0]
                 )


def pose_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T


def matrix_to_pose(T):
    R = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t


def cal_d(rvec1, tvec1, rvec2, tvec2):
    T1 = pose_to_matrix(rvec1, tvec1)
    T2 = pose_to_matrix(rvec2, tvec2)
    dT = np.linalg.inv(T1) @ T2

    dR = dT[:3, :3]
    dt = dT[:3, 3].reshape(3, 1)
    drvec, _ = cv2.Rodrigues(dR)
    dtvec = dt
    return drvec.flatten(), dtvec.flatten()


def add_pose(rvec1, tvec1, drvec, dtvec):
    T1 = pose_to_matrix(rvec1, tvec1)
    dT = pose_to_matrix(drvec, dtvec)

    T2 = T1 @ dT

    rvec2, tvec2 = matrix_to_pose(T2)
    return rvec2.flatten(), tvec2.flatten()


class MotionSeries:
    def __init__(self, filepath, root_orient=None, trans=None, correct_trans=False):
        """

        Args:
            filepath:
            root_orient: root_orient_override
            trans: translation_override
            correct_trans:
        """
        self.data = np.load(filepath)  # npz file path

        self.root_orient = root_orient
        self.trans = trans
        # init transformation in numpy version
        self.init_root_orient = root_orient.squeeze().detach().cpu().numpy() if root_orient is not None else np.zeros(
            (3,), dtype=np.float32)
        self.init_trans = trans.squeeze().detach().cpu().numpy() if trans is not None else np.zeros((3,),
                                                                                                    dtype=np.float32)

        self.correct_trans: bool = correct_trans
        # use original global_t and root_orient instead of set to zero
        self._index = 0

        # check
        for key in ['pose', 'shape', 'global_t', 'pred_joints', 'focal_l']:
            assert key in self.data.keys(), "missing key '{}'".format(key)

        assert self.trans is None or (isinstance(self.trans, torch.Tensor) and self.trans.shape == (1,
                                                                                                    3)), f"{type(self.trans)}_{self.trans.shape}"
        assert self.root_orient is None or (
            isinstance(self.root_orient, torch.Tensor) and self.root_orient.shape == (1, 3),
            f"{type(self.root_orient)}_{self.root_orient.shape}")

    def __len__(self):
        return len(self.data['pose'])

    def __getitem__(self, idx) -> SMPLParameters:
        pose = self.data['pose'][idx]
        shape = self.data['shape'][idx]

        pred_joints = self.data['pred_joints'][idx]
        focal_l = self.data['focal_l'][idx]
        Jtrs = Jtr_norm.float().unsqueeze(0)  # normalized Joints

        # TODO: HANDLE THIS ISSUE
        if not self.correct_trans:
            # override mode
            if self.root_orient is None:
                root_orient = torch.from_numpy(pose[:3]).float().unsqueeze(0)  # (1, 3)
            else:
                root_orient = self.root_orient

            global_t = self.data['global_t'][idx]
            if self.trans is None:
                trans = torch.from_numpy(global_t).float().unsqueeze(0)
            else:
                trans = self.trans

        else:
            # accumulate mode
            root_orient0 = self.data['pose'][0][:3]
            global_t0 = self.data['global_t'][0]
            root_orient_curr = self.data['pose'][idx][:3]
            global_t_curr = self.data['global_t'][idx]
            trans = (self.init_trans + (global_t_curr - global_t0)).flatten()
            # drvec, dtvec = cal_d(root_orient0, global_t0,  # d_transformation to the first frame
            #                      root_orient_curr, global_t_curr)
            #
            # # in this situation, use init transformation
            # r_last, t_last = add_pose(self.init_root_orient, self.init_trans, drvec, dtvec)
            #

            trans = torch.from_numpy(trans).float().unsqueeze(0)
            root_orient = torch.from_numpy(self.init_root_orient.flatten()).float().unsqueeze(0)

            global_t = trans.squeeze(0).detach().cpu().numpy()

        rots, bone_transforms, vertices, Jtr_posed, bone_transforms_orig = parse_params(pose, shape, root_orient, trans)

        params = SMPLParameters(
            pose=pose,  # ()
            shape=shape,
            global_t=global_t,
            pred_joints=pred_joints,
            focal_l=focal_l,
            Jtrs=Jtrs,
            rots=rots,
            bone_transforms=bone_transforms,
            vertices=vertices,
            Jtr_posed=Jtr_posed,
            root_orient=root_orient,
            bone_transforms_orig=bone_transforms_orig
        )
        return params

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
        result = self[self._index]
        self._index += 1
        return result


base_path = os.path.dirname(__file__)
__transform_path = os.path.join(base_path, "../bone_transform_02v.npy")
bone_transform_02v = torch.from_numpy(np.load(__transform_path)).cuda()

Jtr_norm = torch.tensor([
    [0.12727605, -0.07177323, 0.15436138],
    [0.19038345, -0.15399979, 0.14810718],
    [0.06587659, -0.15324871, 0.15048816],
    [0.12500642, 0.02573601, 0.13086761],
    [0.2203844, -0.48889235, 0.14370158],
    [0.03213279, -0.4946489, 0.14248238],
    [0.12969496, 0.14680439, 0.13139906],
    [0.20848866, -0.84309039, 0.104401],
    [0.0460058, -0.84921469, 0.10442484],
    [0.13112752, 0.19406903, 0.15480196],
    [0.23170181, -0.8929688, 0.21106939],
    [0.0238269, -0.8921199, 0.21464352],
    [0.12873973, 0.38599632, 0.11458953],
    [0.20194961, 0.30334232, 0.12352116],
    [0.0578185, 0.30083631, 0.11934796],
    [0.13313888, 0.44409636, 0.16068102],
    [0.2835721, 0.33082552, 0.11565754],
    [-0.02814607, 0.33001713, 0.1108053],
    [0.51536132, 0.31992104, 0.09097455],
    [-0.254716, 0.31851704, 0.09179471],
    [0.73747431, 0.32784373, 0.09006356],
    [-0.48238716, 0.32550399, 0.08713369],
    [0.81253931, 0.32027743, 0.0768725],
    [-0.55798689, 0.31982925, 0.07783393]
])

__body_model_path = os.path.join(base_path, "../body_models/smpl/neutral/model.pkl")
# load basic bodyModel
body_model = None


def parse_params(pose, shape, root_orient, trans):
    """
    Returns:
        rots (1, 24, 9)
        Jtrs (1, 24, 3)
        bone_transforms (1, 24, 4, 4)
    """
    global body_model
    if body_model is None:
        from human_body_prior.body_model.body_model import BodyModel
        body_model = BodyModel(bm_path=__body_model_path, num_betas=10, batch_size=1).cuda()

    # ================================ get poses (rots) ====================================================
    pose = pose.astype(np.float32)  # (72, )

    # root_orient = torch.zeros_like(root_orient)
    # 人物正面是z+，为了让相机观察到人物正面，需要将模型围绕x轴旋转180度
    pose_body = torch.from_numpy(pose[3:66]).float().unsqueeze(0)  # (1, 63)
    pose_hand = torch.from_numpy(pose[66:]).float().unsqueeze(0)  # (1, 6)

    pose_mat = Rotation.from_rotvec(pose.reshape([-1, 3])).as_matrix()  # Rodrigues
    pose_mat[0] = np.eye(3)
    rots = torch.from_numpy(pose_mat.reshape([-1, 9])).float().unsqueeze(0)  # rots as features

    #  ================================ get bone_transforms ================================================
    betas = torch.from_numpy(shape).float().unsqueeze(0)

    body = body_model(root_orient=root_orient.cuda(),  # (1, 3)
                      pose_body=pose_body.cuda(),  # (1, 63)
                      pose_hand=pose_hand.cuda(),  # (1, 6)
                      betas=betas.cuda(),  # (1, 10)
                      trans=trans.cuda()  # (1, 3)
                      )

    bone_transforms_orig = body.bone_transforms.float()  # (1, 24, 4, 4)
    bone_transforms = bone_transforms_orig @ bone_transform_02v.inverse().float()
    bone_transforms[:, :, :3, 3] += trans.flatten().cuda()  # add global offset
    vertices = body.v  # (1, 6890, 3)
    Jtr_posed = body.Jtr  # (1, 24, 3)
    # _, idx = torch.min(Jtr_posed[0, :, 2], 0)  # (24)
    # foot_pos = Jtr_posed[0, idx, :]
    # vertices = vertices.clone() - foot_pos
    # Jtr_posed = Jtr_posed.clone() - foot_pos
    # bone_transforms[:, :, :3, 3] += foot_pos.flatten().cuda()  # add global offset
    return rots, bone_transforms, vertices, Jtr_posed, bone_transforms_orig


if __name__ == '__main__':
    # for example:
    npz_file_path = "your/npz/filepath.npz"  # 使用CLIFF对视频中人体SMPL模型预测的结果文件
    series = MotionSeries(npz_file_path)

    for self in series:
        ...
