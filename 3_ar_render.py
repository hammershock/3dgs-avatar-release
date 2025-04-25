# Copyright (C) 2024. Haonan Zheng. All rights reserved.
"""
读取从视频中提取的每一帧的smpl模型，并将模型序列渲染成为视频

录制的smpl模型序列说明：
模型的顶点是理想化相机坐标系下的顶点，
fx, fy, cx, cy均是从图像的width, height估计出来的


"""
import cv2
import hydra
import numpy as np
# import smplx
import torch
from omegaconf import OmegaConf
from gaussian_renderer import render

# from common import constants
from scene.duck_camera import Camera
# from common.renderer_pyrd import Renderer
from motion_display import MotionSeries, ChArucoStream

from itertools import cycle


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)  # disable set struct
    config.model.pose_correction = {'name': "none", "delay": 5000}  # Don't use Pose Correction

    root_orient = torch.pi * torch.tensor([1., 0., 0.]).unsqueeze(0) * 1.50
    trans = torch.zeros((1, 3)).float().cuda()
    series = MotionSeries("scene_sequences/cxk_clip_cliff_hr48.npz", root_orient=root_orient, trans=trans)

    with ChArucoStream(0, True) as stream, torch.no_grad():
        from scene import GaussianModel, DuckDuckScene

        # setup gaussian model
        gaussian_model = GaussianModel(config.model.gaussian)

        # load Scene
        scene = DuckDuckScene(config, gaussian_model)
        scene.load_checkpoint("./exp/zju_386_mono-direct-mlp_field-ingp-shallow_mlp-default/ckpt15000.pth")
        # scene.load_checkpoint("./exp/zju_377_mono-direct-mlp_field-ingp-shallow_mlp-default/ckpt15000.pth")
        scene.eval()
        # background color
        bg_color = torch.tensor([0., 0., 0.], dtype=torch.float32).cuda()  # black
        import matplotlib
        matplotlib.use('tkagg')
        K = np.array([[503.69234812, 0., 324.57059016],  # (480, 680)
                                 [0., 503.95198528, 295.40637235],
                                 [0, 0, 1]], dtype=np.float32)
        distCoeffs = np.array([0.087944, -0.11943704, -0.00071627, -0.00221681, 0.00644727])
        config.pipeline.debug = True

        for frame, smpl_params in zip(stream, cycle(series)):
            # pose_aa = torch.tensor(smpl_params.pose.reshape(24, 3), device=device)  # (24, 3)
            # pose_rotmat = tgm.angle_axis_to_rotation_matrix(pose_aa)[:, :3, :3]  # (24, 3, 3)

            if stream.board_detected:
                R = stream.R
                T = stream.T

                cam = Camera(R.T, 4.0 * T, K=K)  # 注意cam中使用的R矩阵是列优先的

                # visualizer = CameraFrustumVisualizer(stream.R, stream.T, stream.width, stream.height, cam.FoVx, cam.FoVy, cam.znear, 5.0)
                cam.update_pose(smpl_params.rots.cuda(), smpl_params.Jtrs.cuda(), smpl_params.bone_transforms.cuda())
                # visualizer.plot_frustum(np.zeros((1, 3)))
                render_pkg = render(cam, 15000, scene, config.pipeline, bg_color, compute_loss=False, return_opacity=False)
                rendered_image = render_pkg.rendering_cv2
                # distort
                rendered_image = cv2.undistort(rendered_image, K, distCoeffs)  # undistort
                # map1, map2 = cv2.initUndistortRectifyMap(K, distCoeffs, None, K, (cam.image_width, cam.image_height), 5)
                # rendered_image = cv2.remap(rendered_image, map1, map2, interpolation=cv2.INTER_LINEAR)

                background_mask = rendered_image == 0
                # background_mask = (render_pkg.opacity_render == 0).squeeze(0).cpu().numpy()
                rendered_image[background_mask] = frame[background_mask]
            else:
                rendered_image = frame
            # frames.append(rendered_image)
            cv2.imshow("Rendered Frame", rendered_image)
            key = cv2.waitKey(1)  # 按ESC退出
            if key == 27:
                break
    # save_video_from_frames(frames, "./ar_rendering.mp4")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with torch.no_grad():
        main()
