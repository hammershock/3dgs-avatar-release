"""
demo 1: free-view rendering
"""
from itertools import cycle
import os
import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from tqdm.contrib import tenumerate

from gaussian_renderer import render
from motion_display import get_rotation_matrix
from scene import GaussianModel, DuckDuckScene
from scene.duck_camera import Camera


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)  # disable set struct
    config.model.pose_correction = {'name': "none", "delay": 5000}  # Don't use Pose Correction

    # setup gaussian model
    gaussian_model = GaussianModel(config.model.gaussian)

    # load Scene
    scene = DuckDuckScene(config, gaussian_model)
    model_path = "./exp/zju_001_mono-direct-mlp_field-ingp-shallow_mlp-default/ckpt15000.pth"
    scene.load_checkpoint(model_path)
    scene.eval()

    # background color
    bg_color = torch.tensor([0., 0., 0.], dtype=torch.float32).cuda()  # black

    # load smpl model sequence data
    from motion_display import MotionSeries

    series = MotionSeries("./scene_sequences/cxk_clip_cliff_hr48.npz",
                          root_orient=torch.pi * torch.tensor([0., 0., 1.]).unsqueeze(0),  # override root_orient
                          trans=torch.tensor([[0.0, 0.0, 4.0]]),  # override trans
                          )

    # ================ Hard-Coded Camera ViewPoint ==============================
    R = torch.eye(3).cuda()
    T = torch.tensor([0.0, 0.0, 0.0]).cuda()  # (3, )
    print(len(series))
    for i, params in tenumerate(series, desc="Rendering"):
        cam = Camera(R, T, height=1080, width=1920)
        cam.update_pose(params.rots.cuda(), params.Jtrs.cuda(), params.bone_transforms.cuda())

        # export
        name = str(i + 2587).zfill(6)
        out_models_path = os.path.join("./data/ZJUMoCap/CoreView_001/models", f"{name}.npz")
        params.export(out_models_path)

        with torch.no_grad():
            render_pkg = render(cam, 0, scene, config.pipeline, bg_color, compute_loss=False, return_opacity=False)
        # print(render_pkg.opacity_render.shape)  # (1, 512, 512)
        image = render_pkg.rendering_cv2
        cv2.imwrite(f"./data/ZJUMoCap/CoreView_001/1/{name}.jpg", image)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"./data/ZJUMoCap/CoreView_001/1/{name}.png", (image_gray != 0).astype(np.uint8) * 255)

        cv2.imshow("Rendered Frame", image)
        key = cv2.pollKey()
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
