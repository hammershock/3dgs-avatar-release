#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import torch
import trimesh

from models import GaussianConverter
from scene.gaussian_model import GaussianModel
from dataset import load_dataset
from utils.dataset_utils import get_02v_bone_transforms, AABB


BASE_DIR = "body_models/misc"
faces: np.ndarray = np.load(os.path.join(BASE_DIR, 'faces.npz'))['faces']  # (n, 3)
skinning_weights = dict(np.load(os.path.join(BASE_DIR, 'skinning_weights_all.npz')))
posedirs = dict(np.load(os.path.join(BASE_DIR, 'posedirs_all.npz')))
J_regressor = dict(np.load(os.path.join(BASE_DIR, 'J_regressors.npz')))

SMPL_FILE_PATH = "./data/ZJUMoCap/CoreView_377/models/000000.npz"


class Scene:
    gaussians : GaussianModel

    def __init__(self, cfg, gaussians : GaussianModel, save_dir : str):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg
        self.save_dir = save_dir

        self.gaussians = gaussians

        # load train dataset
        self.train_dataset = load_dataset(cfg.dataset, split='train')
        self.metadata = self.train_dataset.metadata

        # load test dataset
        if cfg.mode == 'train':
            self.test_dataset = load_dataset(cfg.dataset, split='val')
        elif cfg.mode == 'test':
            self.test_dataset = load_dataset(cfg.dataset, split='test')
        elif cfg.mode == 'predict':
            self.test_dataset = load_dataset(cfg.dataset, split='predict')
        else:
            raise ValueError

        self.cameras_extent = self.metadata['cameras_extent']

        self.gaussians.create_from_pcd(self.test_dataset.readPointCloud(), spatial_lr_scale=self.cameras_extent)

        self.converter = GaussianConverter(cfg, self.metadata).cuda()

    def train(self):
        self.converter.train()

    def eval(self):
        self.converter.eval()

    def optimize(self, iteration):
        gaussians_delay = self.cfg.model.gaussian.get('delay', 0)
        if iteration >= gaussians_delay:
            self.gaussians.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.converter.optimize()

    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True):
        return self.converter(self.gaussians, viewpoint_camera, iteration, compute_loss)

    def get_skinning_loss(self):
        loss_reg = self.converter.deformer.rigid.regularization()
        loss_skinning = loss_reg.get('loss_skinning', torch.tensor(0.).cuda())
        return loss_skinning

    def save(self, iteration):
        """save as .ply"""
        point_cloud_path = os.path.join(self.save_dir, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_checkpoint(self, iteration):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((self.gaussians.capture(),
                    self.converter.state_dict(),
                    self.converter.optimizer.state_dict(),
                    self.converter.scheduler.state_dict(),
                    iteration), self.save_dir + "/ckpt" + str(iteration) + ".pth")

    def load_checkpoint(self, path):
        res = torch.load(path)
        (gaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = res
        self.gaussians.restore(gaussian_params, self.cfg.opt)
        self.converter.load_state_dict(converter_sd)
        # self.converter.optimizer.load_state_dict(converter_opt_sd)
        # self.converter.scheduler.load_state_dict(converter_scd_sd)


def get_cano_smpl_verts(data_path):
    gender = 'neutral'
    minimal_shape = fix_symmetry(np.load(data_path)['minimal_shape'])
    j = J_regressor[gender]
    Jtr = np.dot(j, minimal_shape)  # Joints

    # skinning_weights
    skw = skinning_weights[gender].astype(np.float32)

    # smpl_verts
    bone_transforms_02v = get_02v_bone_transforms(Jtr)
    T = np.matmul(skw, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
    vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]  # vertices in cano pose
    smpl_verts = vertices.astype(np.float32)

    # cano_mesh
    cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=faces)

    # aabb
    coord_min, coord_max = get_bbox(vertices, padding=0.1)
    aabb = AABB(coord_max, coord_min)
    # np.save("./bone_transform_02v.npy", bone_transforms_02v)
    return {
        'gender': gender,
        'smpl_verts': smpl_verts,  #
        'minimal_shape': minimal_shape,
        'Jtr': Jtr,
        'skinning_weights': skw,  #
        'bone_transforms_02v': bone_transforms_02v,
        'cano_mesh': cano_mesh,  #
        'coord_min': coord_min,
        'coord_max': coord_max,
        'aabb': aabb,  #
    }


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


class DuckDuckScene:
    gaussians : GaussianModel

    def __init__(self, cfg, gaussians : GaussianModel, save_dir : str = ""):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.cfg = cfg
        self.save_dir = ""
        self.gaussians = gaussians

        # load train dataset
        # TODO: load metadata
        self.metadata = get_cano_smpl_verts(SMPL_FILE_PATH)
        self.metadata['frame_dict'] = None
        self.cameras_extent =  3.469298553466797  # hardcoded, used to scale the threshold for scaling/image-space gradient
        self.converter = None  # lazy load

    def train(self):
        self.converter.train()

    def eval(self):
        self.converter.eval()

    def optimize(self, iteration):
        # TODO: cuz we use this class for Inference-Only, we don't need this method
        raise NotImplementedError
        # gaussians_delay = self.cfg.model.gaussian.get('delay', 0)
        # if iteration >= gaussians_delay:
        #     self.gaussians.optimizer.step()
        # self.gaussians.optimizer.zero_grad(set_to_none=True)
        # self.converter.optimize()

    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True):
        return self.converter(self.gaussians, viewpoint_camera, iteration, compute_loss)

    def get_skinning_loss(self):
        loss_reg = self.converter.deformer.rigid.regularization()
        loss_skinning = loss_reg.get('loss_skinning', torch.tensor(0.).cuda())
        return loss_skinning

    def save(self, iteration):
        # TODO: this is a bad method
        raise NotImplementedError
        # """save as .ply"""
        # point_cloud_path = os.path.join(self.save_dir, "point_cloud/iteration_{}".format(iteration))
        # self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_checkpoint(self, iteration):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((self.gaussians.capture(),
                    self.converter.state_dict(),
                    self.converter.optimizer.state_dict(),
                    self.converter.scheduler.state_dict(),
                    iteration), self.save_dir + "/ckpt" + str(iteration) + ".pth")

    def load_checkpoint(self, path):
        res = torch.load(path)
        (gaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = res
        self.gaussians.restore(gaussian_params, self.cfg.opt)
        n = converter_sd["texture.latent.weight"].shape[0]
        self.metadata['frame_dict'] = {str(i): i for i in range(n)}
        self.converter = GaussianConverter(self.cfg, self.metadata).cuda()
        self.converter.load_state_dict(converter_sd, strict=False)
        # self.converter.optimizer.load_state_dict(converter_opt_sd)
        # self.converter.scheduler.load_state_dict(converter_scd_sd)