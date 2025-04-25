"""
demo 1: free-view rendering
"""
from itertools import cycle

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

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
                          root_orient=torch.pi * torch.tensor([1., 0., 0.]).unsqueeze(0),  # override root_orient
                          trans=torch.zeros((1, 3)).float().cuda(),  # override trans
                          # correct_trans=True,
                          )

    # ================ Hard-Coded Camera ViewPoint ==============================
    R = torch.eye(3).cuda()
    T = torch.tensor([0.0, 0.0, 3.0]).cuda()  # (3, )

    # Rotation increment for each frame (in radians)
    dtheta = 0.01  # small rotation increment for each frame (in radians)

    # Get the rotation increment matrix for this frame
    dR = get_rotation_matrix(np.array([0.0, 1.0, 0.0]), dtheta)
    dR = torch.from_numpy(dR).to(R.device).float()

    for params in tqdm(cycle(series), desc="Rendering"):
        R = torch.matmul(R, dR)  # Apply the z-axis rotation increment (free-view camera)
        cam = Camera(R, T, height=512, width=512, K=None, fx=500.0, fy=500.0)
        cam.update_pose(params.rots.cuda(), params.Jtrs.cuda(), params.bone_transforms.cuda())

        with torch.no_grad():
            render_pkg = render(cam, 0, scene, config.pipeline, bg_color, compute_loss=False, return_opacity=False)
        # print(render_pkg.opacity_render.shape)  # (1, 512, 512)
        image = render_pkg.rendering_cv2
        cv2.imshow("Rendered Frame", image)
        key = cv2.pollKey()
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
