import cv2
import hydra
import torch
from omegaconf import OmegaConf

from gaussian_renderer import render
from motion_display import VideoStream, MotionSeries
from scene import GaussianModel, DuckDuckScene
from scene.duck_camera import Camera


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)  # disable set struct
    config.model.pose_correction = {'name': "none", "delay": 5000}  # Don't use Pose Correction

    # setup gaussian model
    gaussian_model = GaussianModel(config.model.gaussian)

    # load Scene
    scene = DuckDuckScene(config, gaussian_model)
    # scene.load_checkpoint("./exp/zju_377_mono-direct-mlp_field-ingp-shallow_mlp-default/ckpt15000.pth")
    scene.load_checkpoint("./exp/zju_386_mono-direct-mlp_field-ingp-shallow_mlp-default/ckpt15000.pth")
    scene.eval()

    # background color
    bg_color = torch.tensor([0., 0., 0.], dtype=torch.float32).cuda()  # black

    # load smpl model sequence data
    from motion_display import MotionSeries
    # root_orient = torch.pi * torch.tensor([1., 0., 0.]).unsqueeze(0)
    series = MotionSeries("./scene_sequences/cxk_clip_cliff_hr48.npz")  # use original root_orient and global_trans

    # Frame loop ...
    R = torch.eye(3).cuda().float()
    T = torch.tensor([0.0, 0.0, 0.0]).cuda()  # (3, )

    with VideoStream("./videos/cxk_clip.mp4") as video:
        for frame, params in zip(video, series):
            # TODO: render on frame
            cam = Camera(R, T, height=video.height, width=video.width, K=video.K)  # Notice: Camera中R是列优先的
            cam.update_pose(params.rots.cuda(), params.Jtrs.cuda(), params.bone_transforms.cuda())
            render_pkg = render(cam, 0, scene, config.pipeline, bg_color, compute_loss=False, return_opacity=False)
            render_image = render_pkg.rendering_cv2
            background_mask = render_image == 0
            render_image[background_mask] = frame[background_mask]

            cv2.imshow("frame", render_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    with torch.no_grad():
        main()