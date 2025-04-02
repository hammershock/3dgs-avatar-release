import os
from collections import defaultdict
from typing import Dict

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from tqdm import trange

from scene.cameras import Camera
from utils.dataset_utils import AABB, get_02v_bone_transforms, fetchPly, storePly
from utils.graphics_utils import focal2fov


# ========== 估计 GridBoard 的位姿 ==========
def estimatePoseGridBoard(corners, ids, board, cameraMatrix, distCoeffs):
    if ids is None or len(ids) < 4:
        return None, None, False
    objPoints, imgPoints = board.matchImagePoints(corners, ids)
    if objPoints is None or len(objPoints) < 4:
        return None, None, False
    success, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cameraMatrix, distCoeffs)
    return rvec, tvec, success


class DummyDataset:
    cap = cv2.VideoCapture(0)

    def __init__(self, cfg, split='train', mediapipe_npz_path="gym.npz"):
        self.cfg = cfg
        self.split = split
        assert mediapipe_npz_path is not None

        self.root_dir: str = cfg.root_dir  # "./data/ZJUMoCap"
        assert cfg.refine == False, "refine option not available!"
        self.refine: bool = False

        self.subject: str = cfg.subject  # film series e.g. "CoreView_377"

        self.train_frames = cfg.train_frames  # [0, 570, 1]
        self.train_views = cfg.train_views  # ['1', '2']

        self.val_frames = cfg.val_frames  # [0, 1, 1]
        self.val_cams = cfg.val_views  # ['5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
        self.test_mode = cfg.test_mode  # 'view', 'video' or 'all'
        self.white_bg: bool = cfg.white_background  # background color
        self.H, self.W, _ = self.cap.read()[1].shape  # hardcoded size of ZJU-MoCap original images
        # (480, 640, 3)
        self.h, self.w = cfg.img_hw  # actual size we use to train our model...

        # TODO: load mediapipe data sequence
        pose_data = np.load(mediapipe_npz_path)
        self.keypoints_3d = pose_data["keypoints"]  # (num_frames, 33, 3)
        frame_ids = pose_data["frame_ids"]
        # print('building smpl data')
        self.total_frames_actions = 570 # 570  # self.keypoints_3d.shape[0]
        self.data = [self._mediapipe_to_smpl(self.keypoints_3d[i]) for i in trange(self.total_frames_actions, desc="building smpl data")]

        self.frame_idx = 0

        # ============================== Grid Board Settings =========================================
        GRID_ROWS = 3
        GRID_COLS = 3
        MARKER_LENGTH = 0.05  # 5cm
        MARKER_SEPARATION = 0.008

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h10)
        detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

        self.grid_board = cv2.aruco.GridBoard((GRID_COLS, GRID_ROWS), MARKER_LENGTH, MARKER_SEPARATION, aruco_dict)

        # ============================== Camera Params ========================================================
        self.K = np.array([[503.69234812, 0., 324.57059016],
                                 [0., 503.95198528, 295.40637235],
                                 [0, 0, 1]], dtype=np.float32)
        # self.D = np.array([0.087944, -0.11943704, -0.00071627, -0.00221681, 0.00644727])
        self.D = np.array([0, 0, 0, 0, 0], dtype=np.float32)

        self.faces, self.skinning_weights, self.posedirs, self.J_regressor = self._load_smpl_model()
        self.load_metadata(data_path="./data/ZJUMoCap/CoreView_377/models/000000.npz")
        data = defaultdict(list)
        for d in self.data:
            for k, v in d.items():
                data[k].append(v)
        self.metadata.update(data)
        self.metadata['frames'] = list(range(len(self.data)))
        self.metadata['betas'] = np.array([[-0.25837407, -0.08993837,  0.01356627,  0.07494064,  0.00152014,  0.01397103,   0.00520321,  0.00960219, -0.00686408,  0.01246009]])
        self.last_rvec = np.array([0, 0, 1], dtype=np.float32)
        self.last_tvec = np.array([0, 0, 0], dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def _load_smpl_model(self, base_dir="body_models/misc"):
        faces: np.ndarray = np.load(os.path.join(base_dir, 'faces.npz'))['faces']  # (n, 3)
        skinning_weights: Dict[str, np.ndarray] = dict(np.load(os.path.join(base_dir, 'skinning_weights_all.npz')))
        posedirs: Dict[str, np.ndarray] = dict(np.load(os.path.join(base_dir, 'posedirs_all.npz')))
        J_regressor: Dict[str, np.ndarray] = dict(np.load(os.path.join(base_dir, 'J_regressors.npz')))
        return faces, skinning_weights, posedirs, J_regressor


    def _get_cano_smpl_verts(self, data_path):
        """
        Compute star-posed SMPL body vertices.
        To get a consistent canonical space,
        we do not add pose blend shape
        """
        # compute scale from SMPL body
        model_dict = np.load(data_path)
        gender = 'neutral'

        # 3D models and points
        minimal_shape = model_dict['minimal_shape']
        # Break symmetry if given in float16:
        if minimal_shape.dtype == np.float16:
            minimal_shape = minimal_shape.astype(np.float32)
            minimal_shape += 1e-4 * np.random.randn(*minimal_shape.shape)
        else:
            minimal_shape = minimal_shape.astype(np.float32)

        # Minimally clothed shape
        J_regressor = self.J_regressor[gender]
        Jtr = np.dot(J_regressor, minimal_shape)

        skinning_weights = self.skinning_weights[gender]
        # Get bone transformations that transform a SMPL A-pose mesh
        # to a star-shaped A-pose (i.e. Vitruvian A-pose)
        bone_transforms_02v = get_02v_bone_transforms(Jtr)

        T = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        coord_max = np.max(vertices, axis=0)
        coord_min = np.min(vertices, axis=0)
        padding_ratio = self.cfg.padding
        padding_ratio = np.array(padding_ratio, dtype=np.float)
        padding = (coord_max - coord_min) * padding_ratio
        coord_max += padding
        coord_min -= padding

        cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=self.faces)

        return {
            'gender': gender,
            'smpl_verts': vertices.astype(np.float32),
            'minimal_shape': minimal_shape,
            'Jtr': Jtr,
            'skinning_weights': skinning_weights.astype(np.float32),
            'bone_transforms_02v': bone_transforms_02v,
            'cano_mesh': cano_mesh,

            'coord_min': coord_min,
            'coord_max': coord_max,
            'aabb': AABB(coord_max, coord_min),
        }

    def load_metadata(self, data_path) -> None:
        cano_data = self._get_cano_smpl_verts(data_path)
        if self.split != 'train':
            self.metadata = cano_data
            return

        # start, end, step = self.train_frames
        # frames = list(range(len(data_paths)))
        # if end == 0:
        #     end = len(frames)
        # frame_slice = slice(start, end, step)
        # frames = frames[frame_slice]
        frame_dict = {
            frame: i for i, frame in enumerate(range(len(self.data)))
        }

        self.metadata = {
            'faces': self.faces,
            'posedirs': self.posedirs,
            'J_regressor': self.J_regressor,
            'cameras_extent': 3.469298553466797,
            # hardcoded, used to scale the threshold for scaling/image-space gradient
            'frame_dict': frame_dict,
        }

        self.metadata.update(cano_data)
        # if self.cfg.train_smpl:
        #     self.metadata.update(self.load_smpl_data())

    def load_smpl_data(self) -> dict:
        # load all smpl fitting of the training sequence
        # assert self.split == 'train', "you can only access the ground truth human smpl model in train mode!"
        if self.split != 'train':
            return {}

        from collections import defaultdict
        smpl_data = defaultdict(list)

        for idx, (frame, model_file) in enumerate(zip(self.frames, self.model_files_list)):
            model_dict = np.load(model_file)
            if idx == 0:
                smpl_data['betas'] = model_dict['betas'].astype(np.float32)  # betas is the common value

            smpl_data['frames'].append(frame)
            smpl_data['root_orient'].append(model_dict['root_orient'].astype(np.float32))
            smpl_data['pose_body'].append(model_dict['pose_body'].astype(np.float32))
            smpl_data['pose_hand'].append(model_dict['pose_hand'].astype(np.float32))
            smpl_data['trans'].append(model_dict['trans'].astype(np.float32))

        return smpl_data

    def _get_camera(self, idx, data_dict=None):
        """
        body point -> train camera
        body point @ R + t

        world point(ArUco Marker) -> my camera

        Place the body on Aruco Marker...
        body point @ R + t

        Args:
            idx:
            data_dict:

        Returns:

        """
        assert self.cap.isOpened()
        ret, frame = self.cap.read()
        assert ret
        data_dict = self.data[idx]
        cam_name = '0'
        frame_idx = self.frame_idx
        self.frame_idx += 1

        K = self.K.copy()
        dist = self.D.copy().ravel()

        # 处理图像 & 识别 ArUco 标记
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        markerCorners, markerIds, _ = self.aruco_detector.detectMarkers(gray)

        rvec = None
        if markerIds is not None and len(markerIds) > 0:
            (a, b, c) = estimatePoseGridBoard(markerCorners, markerIds, self.grid_board, self.K, self.D)
            if a is not None:
                rvec, tvec, board_detected = a, b, c
                cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec, 0.1)

        if rvec is not None:
            self.last_rvec = rvec
            self.last_tvec = tvec
        else:
            # 用上一frame的数据
            rvec = self.last_rvec
            tvec = self.last_tvec

        rvec = rvec.reshape((3, 1))

        # camera RT
        R, _ = cv2.Rodrigues(rvec)
        # fx, fy = K[0, 0], K[1, 1]
        # cx, cy = K[0, 2], K[1, 2]

        # M = np.eye(3)
        # M[0, 2] = (K[0, 2] - self.W / 2) / K[0, 0]
        # M[1, 2] = (K[1, 2] - self.H / 2) / K[1, 1]
        # K[0, 2] = self.W / 2
        # K[1, 2] = self.H / 2
        # R = M @ R
        # T = M @ T
        # #
        # T = -np.transpose(R) @ tvec
        # T = tvec.flatten()

        R = np.transpose(R)
        # # T = T[:, 0]
        # T = T.flatten()
        # T = tvec.reshape((1, 3))
        T = np.array([-1.0, 1.0, 4.0], dtype=np.float32) + tvec.flatten() * 3.0
        # T = np.array([0.0, 1.0, 4.0], dtype=np.float32)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 图像预处理（从ZJUMoCap的1024*1024）
        image = cv2.undistort(image, K, dist, None)  # 去除畸变
        lanczos = self.cfg.get('lanczos', False)
        interpolation = cv2.INTER_LANCZOS4 if lanczos else cv2.INTER_LINEAR

        # image = cv2.resize(image, (self.w, self.h), interpolation=interpolation)
        # mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
        mask = torch.zeros_like(image)  # dummy mask

        # K[0, :] *= self.w / self.W
        # K[1, :] *= self.h / self.H
        K[0, 0] *= 2
        K[1, 1] *= 2
        fx, fy = K[0, 0], K[1, 1]
        FovX, FovY = focal2fov(fx, self.w), focal2fov(fy, self.h)
        # 537.6245, 539.263
        # todo:
        # Compute posed SMPL body
        minimal_shape = self.metadata['minimal_shape']  # (6890, 3)
        gender = self.metadata['gender']  # 'neutral'

        model_dict = self.data[idx]
        n_smpl_points = minimal_shape.shape[0]  # 6890
        trans = model_dict['trans'].astype(np.float32)  # (3, )
        bone_transforms = model_dict['bone_transforms'].astype(np.float32)  # (24, 4, 4)
        # Also get GT SMPL poses
        root_orient = model_dict['root_orient'].astype(np.float32)  # (3, )
        pose_body = model_dict['pose_body'].astype(np.float32)  # (63, )
        pose_hand = model_dict['pose_hand'].astype(np.float32)  # (6, )
        # Jtr_posed = model_dict['Jtr_posed'].astype(np.float32)
        pose = np.concatenate([root_orient, pose_body, pose_hand], axis=-1)
        pose = Rotation.from_rotvec(pose.reshape([-1, 3]))

        pose_mat_full = pose.as_matrix()  # 24 x 3 x 3
        pose_mat = pose_mat_full[1:, ...].copy()  # 23 x 3 x 3
        pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape(
            [-1, 9])  # 24 x 9, root rotation is set to identity
        # TODO:
        pose_rot_full = pose_mat_full.reshape([-1, 9])  # 24 x 9, including root rotation

        # Minimally clothed shape
        posedir = self.posedirs[gender]  # (6890, 3, 207)
        Jtr = self.metadata['Jtr']  # (24, 3)

        # canonical SMPL vertices without pose correction, to normalize joints
        center = np.mean(minimal_shape, axis=0)  # (3, )
        minimal_shape_centered = minimal_shape - center  # (6890, 3)
        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        # compute pose condition
        Jtr_norm = Jtr - center  # (24, 3)
        Jtr_norm = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtr_norm -= 0.5 # 0.5
        Jtr_norm *= 2.
        # # todo:

        # final bone transforms that transforms the canonical Vitruvian-pose mesh to the posed mesh
        # without global translation
        bone_transforms_02v = self.metadata['bone_transforms_02v']  # (24, 4, 4)
        bone_transforms = bone_transforms @ np.linalg.inv(bone_transforms_02v)  # (24, 4, 4)
        bone_transforms = bone_transforms.astype(np.float32)  # (24, 4, 4)
        bone_transforms[:, :3, 3] += trans  # add global offset
        # todo:

        return Camera(
            frame_id=frame_idx,
            cam_id=int(cam_name),
            K=K, R=R, T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=image,
            mask=mask,
            gt_alpha_mask=None,
            image_name=f"c{int(cam_name):02d}_f{frame_idx if frame_idx >= 0 else -frame_idx - 1:06d}",
            data_device=self.cfg.data_device,
            # human params
            rots=torch.from_numpy(pose_rot).float().unsqueeze(0),
            Jtrs=torch.from_numpy(Jtr_norm).float().unsqueeze(0),
            bone_transforms=torch.from_numpy(bone_transforms),
        )

    def __getitem__(self, i):
        return self._get_camera(i)


    def _mediapipe_to_smpl(self, mediapipe_joints):
        ...
        """
        将 MediaPipe 关键点转换为 SMPL 模型参数
        :param mediapipe_joints: (33, 3) 的 MediaPipe 3D 关键点
        :return: model_dict (SMPL 格式)
        """
        smpl_parents = np.array(
            [-1, 0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 17, 18])  # SMPL 关节父子关系

        # 1️⃣ 关键点映射
        smpl_to_mediapipe = [23, 23, 24, 11, 25, 26, 9, 27, 28, 0, 11, 12, 13, 14, 15, 16]
        joints_smpl = np.array([mediapipe_joints[i] for i in smpl_to_mediapipe])

        # 2️⃣ 计算 trans
        trans = joints_smpl[0]

        # 3️⃣ 计算 root_orient
        hip_vector = joints_smpl[1] - joints_smpl[2]
        hip_vector /= np.linalg.norm(hip_vector)
        spine_vector = joints_smpl[3] - joints_smpl[0]
        spine_vector /= np.linalg.norm(spine_vector)

        root_rotation_matrix = np.eye(3)
        root_rotation_matrix[:, 0] = hip_vector
        root_rotation_matrix[:, 1] = np.cross(spine_vector, hip_vector)
        root_rotation_matrix[:, 2] = spine_vector

        root_orient = Rotation.from_matrix(root_rotation_matrix).as_rotvec()

        # 4️⃣ 计算 pose_body
        pose_body = np.zeros((21, 3))
        joint_rotations = [Rotation.identity() for _ in range(24)]  # 假设所有关节的旋转矩阵

        for i in range(1, 22):
            parent = smpl_parents[i]
            local_rotation = Rotation.from_matrix(
                np.linalg.inv(joint_rotations[parent].as_matrix()) @ joint_rotations[i].as_matrix()
            ).as_rotvec()
            pose_body[i - 1] = local_rotation

        # 5️⃣ 计算 pose_hand
        pose_hand = np.zeros(6)
        pose_hand[:3] = Rotation.from_matrix(joint_rotations[20].as_matrix()).as_rotvec()
        pose_hand[3:] = Rotation.from_matrix(joint_rotations[21].as_matrix()).as_rotvec()

        # 6️⃣ 组装 model_dict
        model_dict = {
            "trans": trans.astype(np.float32),
            "bone_transforms": np.zeros((24, 4, 4)),  # 暂不计算骨骼变换
            "root_orient": root_orient.astype(np.float32),
            "pose_body": pose_body.reshape(-1).astype(np.float32),
            "pose_hand": pose_hand.astype(np.float32)
        }

        return model_dict

    def readPointCloud(self, n_points=50_000):
        random_init = self.cfg.get('random_init', False)
        ply_path = os.path.join(self.root_dir, self.subject, 'random_pc.ply' if random_init else 'cano_smpl.ply')

        try:
            pcd = fetchPly(ply_path)  # if ply_path exists
            return pcd
        except:
            if random_init:
                # 随机初始化点云
                aabb = self.metadata['aabb']  # 边界框
                coord_min = aabb.coord_min.unsqueeze(0).numpy()
                coord_max = aabb.coord_max.unsqueeze(0).numpy()
                xyz_norm = np.random.rand(n_points, 3)  # sample from Uniform Distribution

                xyz = xyz_norm * coord_min + (1. - xyz_norm) * coord_max
                rgb = np.ones_like(xyz) * 255
            else:
                mesh = trimesh.Trimesh(vertices=self.metadata['smpl_verts'], faces=self.faces)

                xyz = mesh.sample(n_points)
                rgb = np.ones_like(xyz) * 255

            storePly(ply_path, xyz, rgb)
            pcd = fetchPly(ply_path)
            return pcd


# if __name__ == "__main__":
#