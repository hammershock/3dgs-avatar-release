import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from trimesh.path.exchange.misc import polygon_to_path
from typing_extensions import TypedDict

from scene.cameras import Camera
from utils.camera_utils import freeview_camera
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB
from utils.graphics_utils import focal2fov


class PoseGroundTruth(TypedDict):
    betas: np.ndarray
    frames: List[np.ndarray]
    root_orient: List[np.ndarray]
    pose_body: List[np.ndarray]
    pose_hand: List[np.ndarray]
    trans: List[np.ndarray]


class CameraParams(TypedDict):
    K: np.ndarray
    D: np.ndarray
    R: np.ndarray
    T: np.ndarray


def fix_symmetry(arr):
    # Break symmetry if given in float16:
    if arr.dtype == np.float16:
        return arr.astype(np.float32) +  1e-4 * np.random.randn(*arr.shape)
    return arr.astype(np.float32)


def get_bbox(pt3ds, padding=0.):
    coord_max = np.max(pt3ds, axis=0)
    coord_min = np.min(pt3ds, axis=0)
    padding = (coord_max - coord_min) * padding
    coord_max += padding
    coord_min -= padding
    return coord_min, coord_max


class ZJUMoCapDataset(Dataset):
    def __init__(self, cfg, split='train'):
        super().__init__()
        self.cfg = cfg
        self.split = split
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

        self.H, self.W = 1024, 1024 # hardcoded size of ZJU-MoCap original images
        self.h, self.w = cfg.img_hw  # actual size we use to train our model...

        def _get(split) -> Tuple[List[str], List[int]]:
            if split == 'train':
                return self.train_views, self.train_frames
            elif split == 'val':
                return self.val_cams, self.val_frames
            elif split == 'test':
                return self.cfg.test_views[self.test_mode], self.cfg.test_frames[self.test_mode]
            elif split == 'predict':
                return self.cfg.predict_views, self.cfg.predict_frames
            else:
                raise ValueError

        def load_cam_views(path):
            with open(path, 'r') as f:
                cam_params: Dict[str, dict] = json.load(f)
                return cam_params

        # we use some smpl body models to initialize our guassians...
        self.faces, self.skinning_weights, self.posedirs, self.J_regressor = self._load_smpl_model()
        # load camera views series dict
        self.cam_params: dict = load_cam_views(path=os.path.join(self.root_dir, self.subject, 'cam_params.json'))

        cam_names, (start_frame, end_frame, sampling_rate) = _get(split=split)
        assert len(cam_names) > 0, "no cam available, check if the dataset or configuration is correct"  # or uncomment the next line
        # cam_names = self.cam_params['all_cam_names']

        subject_dir = os.path.join(self.root_dir, self.subject)  # the directory of the filming series...

        if split == 'predict':
            seq_chosen = ['gBR_sBM_cAll_d04_mBR1_ch05_view1',
                    'gBR_sBM_cAll_d04_mBR1_ch06_view1',
                    'MPI_Limits-03099-op8_poses_view1',
                    'canonical_pose_view1'][self.cfg.get('predict_seq', 0)]

            model_dir = os.path.join(subject_dir, seq_chosen, '*.npz')
            model_files = sorted(glob.glob(model_dir))
            # n = 5
            # (-5, -4, -3, -2, -1)
            # (-1, -2, -3, -4, -5)
            frames = list(reversed(range(-len(model_files), 0)))  # 倒序索引
            # frames = list(range(len(model_files)))
        else:
            # else if 'train', 'test', 'val' split, we should use the seq of ZJUMoCap
            assert not self.cfg.get('arah_opt', False), "arah opt not available"
            model_dir = os.path.join(subject_dir, 'models/*.npz')
            model_files = sorted(glob.glob(model_dir))
            frames = list(range(len(model_files)))

        self.model_files = model_files
        if end_frame == 0:
            end_frame = len(model_files)
        frame_slice = slice(start_frame, end_frame, sampling_rate)
        model_files = model_files[frame_slice]
        frames = frames[frame_slice]
        self.frames: List[int] = frames
        self.model_files_list = model_files

        # add freeview rendering
        if cfg.freeview:
            # with open(os.path.join(self.root_dir, self.subject, 'freeview_cam_params.json'), 'r') as f:
            #     self.cam_params = json.load(f)
            model_dict = np.load(model_files[0])
            trans = model_dict['trans'].astype(np.float32)
            self.cam_params = freeview_camera(self.cam_params[cam_names[0]], trans)
            cam_names = self.cam_params['all_cam_names']

        # load data
        self.load_data_metadata(split, cfg.freeview, cam_names, subject_dir, frame_slice, frames, model_files)

        # get meta data
        self.metadata = self.load_metadata()

        if self.cfg.get('preload', True):
            self.cam_params = [self._get_camera(idx) for idx in range(len(self))]

    def load_data_metadata(self, split, freeview, cam_names, subject_dir, frame_slice, frames, model_files):
        self.data = []

        # when you do some prediction on some unseen poses or use freeview cameras,
        # there is no ground truth according to the human pose or camera param
        use_dummies: bool = (split == 'predict' or freeview)

        for cam_idx, cam_name in enumerate(cam_names):
            cam_dir = os.path.join(subject_dir, cam_name)

            if use_dummies:
                img_files, mask_files = None, None
            else:
                img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))[frame_slice]
                mask_files = sorted(glob.glob(os.path.join(cam_dir, '*.png')))[frame_slice]

            dummy_img_file = os.path.join(subject_dir, '1', '000000.jpg')
            dummy_mask_file = os.path.join(subject_dir, '1', '000000.png')

            for i, frame_index in enumerate(frames):
                # for each frame in each camera's view...
                img_file = img_files[i] if not use_dummies else dummy_img_file
                mask_file = mask_files[i] if not use_dummies else dummy_mask_file
                model_file = model_files[i]

                self.data.append({
                    'cam_idx': cam_idx,
                    'cam_name': cam_name,
                    'data_idx': i,
                    'frame_idx': frame_index,
                    'img_file': img_file,  # image files path
                    'mask_file': mask_file,  # mask file path
                    'model_file': model_file,  # SMPL model file path
                })


    def load_metadata(self):
        print("data_path", self.model_files[0])
        metadata = self.get_cano_smpl_verts(self.model_files[0])

        if self.split != 'train':
            return metadata

        indices = [idx for idx, _ in enumerate(self.model_files)]
        start, end, step = self.train_frames
        if end == 0:
            end = len(indices)
        frames = indices[start: end: step]
        frame_dict = {frame: i for i, frame in enumerate(frames)}
        metadata.update({
            'faces': self.faces,
            'posedirs': self.posedirs,
            'J_regressor': self.J_regressor,
            'cameras_extent': 3.469298553466797,  # hardcoded, used to scale the threshold for scaling/image-space gradient
            'frame_dict': frame_dict,
        })

        if self.cfg.train_smpl and self.split == 'train':
            pose_gt = self.load_pose_ground_truth(self.frames, self.model_files_list)
            metadata.update(pose_gt)

        return metadata


    def get_cano_smpl_verts(self, data_path):
        """get a star-pose model"""
        gender = 'neutral'  #
        minimal_shape = fix_symmetry(np.load(data_path)['minimal_shape'])
        J_regressor = self.J_regressor[gender]
        Jtr = np.dot(J_regressor, minimal_shape)  # Joints

        skinning_weights = self.skinning_weights[gender]
        bone_transforms_02v = get_02v_bone_transforms(Jtr)

        T = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]  # vertices in cano pose
        cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=self.faces)
        # print("padding", self.cfg.padding)
        coord_min, coord_max = get_bbox(vertices, padding=self.cfg.padding)

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

    @staticmethod
    def load_pose_ground_truth(frames, model_files_list) -> PoseGroundTruth:
        # load all smpl fitting of the training sequence
        ret = defaultdict(list)
        for idx, (frame, model_file) in enumerate(zip(frames, model_files_list)):
            model_dict = np.load(model_file)
            if idx == 0:
                ret['betas'] = model_dict['betas'].astype(np.float32)  # betas is the common value

            ret['frames'].append(frame)
            ret['root_orient'].append(model_dict['root_orient'].astype(np.float32))
            ret['pose_body'].append(model_dict['pose_body'].astype(np.float32))
            ret['pose_hand'].append(model_dict['pose_hand'].astype(np.float32))
            ret['trans'].append(model_dict['trans'].astype(np.float32))

        return ret

    @staticmethod
    def _load_smpl_model(base_dir="body_models/misc"):
        faces: np.ndarray = np.load(os.path.join(base_dir, 'faces.npz'))['faces']  # (n, 3)
        skinning_weights: Dict[str, np.ndarray] = dict(np.load(os.path.join(base_dir, 'skinning_weights_all.npz')))
        posedirs: Dict[str, np.ndarray] = dict(np.load(os.path.join(base_dir, 'posedirs_all.npz')))
        J_regressor: Dict[str, np.ndarray] = dict(np.load(os.path.join(base_dir, 'J_regressors.npz')))
        return faces, skinning_weights, posedirs, J_regressor


    def __len__(self):
        return len(self.data)

    def _get_camera(self, idx, data_dict=None):
        if data_dict is None:
            data_dict = self.data[idx]

        cam_idx = data_dict['cam_idx']
        cam_name = data_dict['cam_name']
        data_idx = data_dict['data_idx']
        frame_idx = data_dict['frame_idx']
        img_file = data_dict['img_file']
        mask_file = data_dict['mask_file']
        model_file = data_dict['model_file']

        # 从数据中提取到相机内参，外参
        K = np.array(self.cam_params[cam_name]['K'], dtype=np.float32).copy()
        dist = np.array(self.cam_params[cam_name]['D'], dtype=np.float32).ravel()
        R = np.array(self.cam_params[cam_name]['R'], np.float32)
        T = np.array(self.cam_params[cam_name]['T'], np.float32)

        # note that in ZJUMoCap the camera center does not align perfectly
        # here we try to offset it by modifying the extrinsic...
        M = np.eye(3)
        M[0, 2] = (K[0, 2] - self.W / 2) / K[0, 0]
        M[1, 2] = (K[1, 2] - self.H / 2) / K[1, 1]
        K[0, 2] = self.W / 2
        K[1, 2] = self.H / 2
        R = M @ R
        T = M @ T

        R = np.transpose(R)
        T = T[:, 0]

        image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)

        if self.refine:
            mask = cv2.imread(mask_file)
            mask = mask.sum(-1)
            mask[mask != 0] = 100
            mask = mask.astype(np.uint8)
        else:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        # 图像预处理（从ZJUMoCap的1024*1024）
        image = cv2.undistort(image, K, dist, None)  # 去除畸变
        mask = cv2.undistort(mask, K, dist, None)
        lanczos = self.cfg.get('lanczos', False)
        interpolation = cv2.INTER_LANCZOS4 if lanczos else cv2.INTER_LINEAR

        image = cv2.resize(image, (self.w, self.h), interpolation=interpolation)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        mask = mask != 0
        image[~mask] = 255. if self.white_bg else 0.
        image = image / 255.

        image = torch.from_numpy(image).permute(2, 0, 1).float()  # (C, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        # update camera parameters
        K[0, :] *= self.w / self.W
        K[1, :] *= self.h / self.H

        fx, fy = K[0, 0], K[1, 1]
        FovX, FovY = focal2fov(fx, self.w), focal2fov(fy, self.h)
        # Compute posed SMPL body
        minimal_shape = self.metadata['minimal_shape']  # (6890, 3)
        gender = self.metadata['gender']  # 'neutral'

        model_dict = np.load(model_file)
        n_smpl_points = minimal_shape.shape[0]  # 6890
        trans = model_dict['trans'].astype(np.float32)  # (3, )
        # print("trans", trans)
        bone_transforms = model_dict['bone_transforms'].astype(np.float32)  # (24, 4, 4)

        # Also get GT SMPL poses
        root_orient = model_dict['root_orient'].astype(np.float32)  # (3, )
        pose_body = model_dict['pose_body'].astype(np.float32)  # (63, )  # (21, 3)
        pose_hand = model_dict['pose_hand'].astype(np.float32)  # (6, )  # (2, 3)
        # print("root orient", root_orient)

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
        center = np.mean(minimal_shape, axis=0)
        # print("center: ", center)

        minimal_shape_centered = minimal_shape - center

        cano_max = minimal_shape_centered.max()
        cano_min = minimal_shape_centered.min()
        padding = (cano_max - cano_min) * 0.05

        # compute pose condition
        Jtr_norm = Jtr - center  # (24, 3)
        Jtr_norm = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.
        # todo:

        # final bone transforms that transforms the canonical Vitruvian-pose mesh to the posed mesh
        # without global translation
        bone_transforms_02v = self.metadata['bone_transforms_02v']  # (24, 4, 4)
        bone_transforms = bone_transforms @ np.linalg.inv(bone_transforms_02v)  # (24, 4, 4)  # 到cano的变换
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
            rots=torch.from_numpy(pose_rot).float().unsqueeze(0),  # *
            Jtrs=torch.from_numpy(Jtr_norm).float().unsqueeze(0),  # *
            bone_transforms=torch.from_numpy(bone_transforms),  # *
        )

    def __getitem__(self, idx) -> Camera:
        return self.cam_params[idx] if self.cfg.get('preload', True) else self._get_camera(idx)

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
