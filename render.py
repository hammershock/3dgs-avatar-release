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
from os import makedirs

import cv2
import hydra
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import trange

import wandb
# from dataset import DummyDataset
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import fix_random, Evaluator, PSEvaluator
from visualize_smpl import bone_transform


def set_wandb_logger(config):
    wandb_name = f"{config.name}-{config.suffix}"
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project="gaussian-splatting-avatar",
        entity="fast-avatar-hammershock",
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),
    )


def render_and_save(scene, config, render_path, background, evaluator=None):
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    times, psnrs, ssims, lpipss = [], [], [], []
    print(config.model.pose_correction)
    for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
        # ========================Render Progress=====================================================
        # start to record time
        camera: Camera = scene.test_dataset[idx]

        iter_start.record()
        # TODO: Figure out Core function 'render'
        render_pkg = render(camera, config.opt.iterations, scene, config.pipeline, background, compute_loss=False, return_opacity=False)
        iter_end.record()
        torch.cuda.synchronize()
        elapsed = iter_start.elapsed_time(iter_end)  # 精确记录渲染在gpu上进行的时间
        times.append(elapsed)
        # ======================================= Save Result====================================================

        rendering = render_pkg["render"]
        # save as png
        # torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{camera.image_name}.png"))

        # report renderings images to wandb
        # wandb_img = [wandb.Image(rendering[None], caption=f'render_{camera.image_name}')]

        # ================================== visualize renderings ================================================
        frame = rendering.cpu().numpy().transpose(1, 2, 0) * 255.0
        orig = camera.data['image'].cpu().numpy().transpose(1, 2, 0)
        frame = frame.astype(np.uint8)
        orig = orig.astype(np.uint8)
        mask = frame == 0

        frame[mask] = (orig)[mask]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame
        cv2.imshow("Rendered Frame", frame)
        if cv2.waitKey(1) == 27:  # 按 `Esc` 退出
            break

        if evaluator is not None:
            gt = camera.original_image[:3, :, :]
            metrics = evaluator(rendering, gt)
            psnrs.append(metrics['psnr'].item())
            ssims.append(metrics['ssim'].item())
            lpipss.append(metrics['lpips'].item())
            # 记录 Ground Truth
            # wandb_img.append(wandb.Image(gt[None], caption=f'gt_{camera.image_name}'))

        # wandb.log({'test_images': wandb_img})

    # 计算最终的时间与评估指标
    results = {'metrics/time': np.mean(times[1:]) }
    if evaluator is not None:
        results['metrics/psnr'] = torch.mean(torch.stack(psnrs))
        results['metrics/ssim'] = torch.mean(torch.stack(ssims), dim=0)
        results['metrics/lpips'] = torch.mean(torch.stack(lpipss), dim=0)
    wandb.log(results)
    result_save_path = os.path.join(config.exp_dir, config.suffix, 'results.npz')
    kwargs = {k: (v.cpu().numpy() if torch.is_tensor(v) else v)
                    for k, v in results.items()
                    if v is not None}

    np.savez(result_save_path, **kwargs)
    try:
        if hasattr(scene.train_dataset, 'cap'):
            scene.train_dataset.cap.release()
        if hasattr(scene.train_dataset, 'cap'):
            scene.test_dataset.cap.release()
    except:
        pass


def predict(config, evaluator=None):
    with torch.no_grad():
        # load Scene
        gaussian_model = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussian_model, config.exp_dir)
        scene.eval()

        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, f"ckpt{config.opt.iterations}.pth")
        print(f"load from ckpt: {load_ckpt}")
        scene.load_checkpoint(load_ckpt)  # load checkpoint

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]  # set up background color
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')  # create render output folder
        makedirs(render_path, exist_ok=True)
        render_and_save(scene, config, render_path, background, evaluator=evaluator)


def setup_experiment_directory(config):
    """
    设置实验目录 `exp_dir`，如果未指定则自动创建。
    """
    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)


def get_experiment_suffix(config):
    """
    生成实验后缀 `config.suffix`，用于区分 `test` 和 `predict` 模式。
    """
    if config.mode == 'test':
        return f"{config.mode}-{config.dataset.test_mode}"

    if config.mode == 'predict':
        predict_dict = {
            'zjumocap': {0: 'dance0', 1: 'dance1', 2: 'flipping', 3: 'canonical'},
            'default': {0: 'rotation', 1: 'dance2'}
        }
        dataset_key = config.dataset.name if config.dataset.name in predict_dict else 'default'
        predict_mode = predict_dict[dataset_key].get(config.dataset.predict_seq, 'unknown')
        print(f"Predict Mode: {predict_mode}")
        suffix = f"{config.mode}-{predict_mode}"
        return suffix + '-freeview' if config.dataset.freeview else suffix

    raise ValueError(f"Unknown mode: {config.mode}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.wandb_disable = True
    # print("padding", config.padding)  # zjumocap
    config.dataset.preload = False  # 关闭预加载
    # pretrained_path = "models_pretrained/zju_377_mono/ckpt15000.pth"
    # config.load_ckpt = pretrained_path
    print(f"dataset_name", config.dataset_name)  # zju_377_mono

    setup_experiment_directory(config)
    config.suffix = get_experiment_suffix(config)

    set_wandb_logger(config)
    fix_random(config.seed)

    # predict
    if config.mode == "test":
        evaluator = PSEvaluator() if config.dataset.name == 'people_snapshot' else Evaluator()
        predict(config, evaluator=evaluator)
    elif config.mode == "predict":
        predict(config, evaluator=None)
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

if __name__ == "__main__":
    # python render.py mode=test dataset.test_mode=view dataset=zjumocap_377_mono
    # python render.py mode=predict dataset.predict_seq=0 dataset=zjumocap_377_mono
    main()
    cv2.destroyAllWindows()
