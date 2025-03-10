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

import torch
import numpy as np
from scene import Scene
import os
from tqdm import tqdm, trange
from os import makedirs
from gaussian_renderer import render
import torchvision
import cv2

from scene.cameras import Camera
from utils.general_utils import fix_random
from scene import GaussianModel

from utils.general_utils import Evaluator, PSEvaluator

import hydra
from omegaconf import OmegaConf
import wandb

def predict(config):
    with torch.set_grad_enabled(False):
        gaussian_model = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussian_model, config.exp_dir)
        scene.eval()

        # load scene from checkpoint
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:  # load_ckpt from pretrained as default
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        # background color
        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # output path
        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        times = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            camera = scene.test_dataset[idx]
            iter_start.record()

            render_pkg = render(camera, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=False)
            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(camera.image_name)),]
            wandb.log({'test_images': wandb_img})

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{camera.image_name}.png"))

            # evaluate
            times.append(elapsed)

        _time = np.mean(times[1:])
        wandb.log({'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 time=_time)



def test(config):
    with torch.no_grad():
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.eval()
        load_ckpt = config.get('load_ckpt', None)
        if load_ckpt is None:
            load_ckpt = os.path.join(scene.save_dir, "ckpt" + str(config.opt.iterations) + ".pth")
        scene.load_checkpoint(load_ckpt)

        bg_color = [1, 1, 1] if config.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(config.exp_dir, config.suffix, 'renders')
        makedirs(render_path, exist_ok=True)

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        evaluator = PSEvaluator() if config.dataset.name == 'people_snapshot' else Evaluator()

        psnrs = []
        ssims = []
        lpipss = []
        times = []
        for idx in trange(len(scene.test_dataset), desc="Rendering progress"):
            camera = scene.test_dataset[idx]

            iter_start.record()

            render_pkg = render(camera, config.opt.iterations, scene, config.pipeline, background,
                                compute_loss=False, return_opacity=False)

            iter_end.record()
            torch.cuda.synchronize()
            elapsed = iter_start.elapsed_time(iter_end)

            rendering = render_pkg["render"]

            gt = camera.original_image[:3, :, :]
            frame = rendering.cpu().numpy().transpose(1, 2, 0)
            frame *= 255.0
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # cv2.cvtColor(render_image, cv2.COLOR_RGB2BGR, render_image)
            # ground_truth = gt.cpu().numpy().transpose(1, 2, 0)
            # compare_image = np.concatenate([frame, ground_truth], axis=1)
            cv2.imshow("frame", frame)
            k = cv2.waitKey(1)
            if k == 27:
                break

            wandb_img = [wandb.Image(rendering[None], caption='render_{}'.format(camera.image_name)),
                         wandb.Image(gt[None], caption='gt_{}'.format(camera.image_name))]
            wandb.log({'test_images': wandb_img})

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"render_{camera.image_name}.png"))

            # evaluate
            if config.evaluate:
                metrics = evaluator(rendering, gt)
                psnrs.append(metrics['psnr'])
                ssims.append(metrics['ssim'])
                lpipss.append(metrics['lpips'])
            else:
                psnrs.append(torch.tensor([0.], device='cuda'))
                ssims.append(torch.tensor([0.], device='cuda'))
                lpipss.append(torch.tensor([0.], device='cuda'))
            times.append(elapsed)

        _psnr = torch.mean(torch.stack(psnrs))
        _ssim = torch.mean(torch.stack(ssims))
        _lpips = torch.mean(torch.stack(lpipss))
        _time = np.mean(times[1:])
        wandb.log({'metrics/psnr': _psnr,
                   'metrics/ssim': _ssim,
                   'metrics/lpips': _lpips,
                   'metrics/time': _time})
        np.savez(os.path.join(config.exp_dir, config.suffix, 'results.npz'),
                 psnr=_psnr.cpu().numpy(),
                 ssim=_ssim.cpu().numpy(),
                 lpips=_lpips.cpu().numpy(),
                 time=_time)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp', config.name)
    os.makedirs(config.exp_dir, exist_ok=True)

    # set wandb logger
    if config.mode == 'test':
        config.suffix = config.mode + '-' + config.dataset.test_mode
    elif config.mode == 'predict':
        predict_seq = config.dataset.predict_seq
        if config.dataset.name == 'zjumocap':
            predict_dict = {
                0: 'dance0',
                1: 'dance1',
                2: 'flipping',
                3: 'canonical'
            }
        else:
            predict_dict = {
                0: 'rotation',
                1: 'dance2',
            }
        predict_mode = predict_dict[predict_seq]
        print("predict_mode:", predict_mode)
        config.suffix = config.mode + '-' + predict_mode
    else:
        raise ValueError("unknown mode: {}".format(config.mode))
    if config.dataset.freeview:
        config.suffix = config.suffix + '-freeview'
    wandb_name = config.name + '-' + config.suffix
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project="gaussian-splatting-avatar",  # 'gaussian-splatting-avatar-test'
        entity="fast-avatar-hammershock",  # 'fast-avatar'
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='fork'),  #
    )

    fix_random(config.seed)

    if config.mode == 'test':
        test(config)
    elif config.mode == 'predict':
        predict(config)
    else:
        raise ValueError

if __name__ == "__main__":
    # mode=test dataset.test_mode=view dataset=zjumocap_377_mono
    #python render.py mode=predict dataset.predict_seq=0 dataset=zjumocap_377_mono
    main()
    cv2.destroyAllWindows()