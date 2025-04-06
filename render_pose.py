import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from gaussian_renderer import render
from human_body_prior.body_model.body_model import BodyModel
from scene import GaussianModel, DuckDuckScene
from utils.graphics_utils import focal2fov, getProjectionMatrix


# from scene.cameras import Camera

def getWorld2View2(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = Rt.inverse()
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = C2W.inverse()
    return Rt.float()


class Camera:
    def __init__(self, R, T, *args, **kwargs):
        self.fx, self.fy = 537.6245, 539.263

        self.image_height = 512  # 480
        self.image_width = 512  # 640

        self.FoVx = focal2fov(self.fx, self.image_width)
        self.FoVy = focal2fov(self.fy, self.image_height)

        self.cx, self.cy = self.image_width / 2, self.image_height / 2

        # self.world_view_transform = torch.tensor([[-0.7216, -0.2863, 0.6130, 0.0000],
        #                                           [0.6933, -0.2860, 0.7003, 0.0000],
        #                                           [-0.0109, 0.9154, 0.3658, 0.0000],
        #                                           [0.4548, 0.8221, 2.9253, 1.0000]], device='cuda:0')
        # self.full_proj_transform = torch.tensor([[-1.5155, -0.6031, 0.6130, 0.6130],
        #                                          [1.4560, -0.6024, 0.7004, 0.7003],
        #                                          [-0.0228, 1.9282, 0.3658, 0.3658],
        #                                          [0.9552, 1.7318, 2.9156, 2.9253]], device='cuda:0')
        # self.projection_matrix = torch.tensor([[2.1001, 0.0000, 0.0000, 0.0000],
        #                                        [0.0000, 2.1065, 0.0000, 0.0000],
        #                                        [0.0000, 0.0000, 1.0001, 1.0000],
        #                                        [0.0000, 0.0000, -0.0100, 0.0000]], device='cuda:0')

        self.zfar = 100.0
        self.znear = 0.01

        self.R = R
        self.T = T

        Rt = torch.zeros((4, 4), dtype=torch.float32).cuda()
        Rt[:3, :3] = R  # ?
        Rt[3, :3] = T
        Rt[3, 3] = 1.0

        # Rt = getWorld2View2(R, T, scale=2.0).T.cuda()

        self.world_view_transform = Rt
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.frame_id = -1

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float32)

    def update_pose(self, rots, Jtrs, bone_transforms):
        # data = np.load("params_package.npz")
        # self.data = {}
        # for key, value in data.items():
        #     self.data[key] = torch.from_numpy(value).float().cuda()
        self.rots = rots
        self.Jtrs = Jtrs
        self.bone_transforms = bone_transforms

    def __getattr__(self, name):
        if name in self.data:
            return self.data[name]
        else:
            raise AttributeError(name)


# load basic bodyModel
body_model = BodyModel(bm_path='body_models/smpl/neutral/model.pkl', num_betas=10, batch_size=1).cuda()


def load_model(filepath):
    data = np.load(filepath)
    old_trans = data['global_t'][0].astype(np.float32)  # (3, )
    trans = np.zeros_like(old_trans)

    joints = data['pred_joints'][0, :24, :]  # (24, 3)  # 骨骼点位置

    # Normalize joints
    center = np.array(
        [1.0950731e-07, -1.7741595e-05, 2.6038391e-05])  # hard-coded, params calculated in the training set
    cano_max, cano_min = 0.858846, -1.1424413
    padding = 0.10006436109542848
    Jtr_norm = joints - center  # (24, 3)
    Jtr_norm = (Jtr_norm - cano_min + padding) / (cano_max - cano_min) / 1.1
    Jtr_norm -= 0.5
    Jtr_norm *= 2.
    Jtrs = torch.from_numpy(Jtr_norm).float().unsqueeze(0)  # normalized Joints

    # Also get GT SMPL poses
    pose = data['pose'][0].astype(np.float32)  # (72, )
    # TODO: parse stacked pose vector
    pose_body = torch.from_numpy(pose[3:66]).float().unsqueeze(0)  # (1, 63)
    pose_hand = torch.from_numpy(pose[66:]).float().unsqueeze(0)  # (1, 6)

    pose = Rotation.from_rotvec(pose.reshape([-1, 3]))  # Rodrigues
    pose_mat_full = pose.as_matrix()  # (24, 3, 3)
    pose_mat = pose_mat_full[1:, ...].copy()  # (23, 3, 3)
    pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0).reshape([-1, 9])  # (24, 9)

    # pose_rot = pose_mat_full.reshape([-1, 9])
    rots = torch.from_numpy(pose_rot).float().unsqueeze(0)  # rots as features
    betas = torch.from_numpy(data['shape'][0]).float().unsqueeze(0)

    # TODO: get bone_transforms!
    body = body_model(root_orient=torch.zeros((1, 3), dtype=torch.float32).cuda(),  # (1, 3)
                      pose_body=pose_body.cuda(),  # (1, 63)
                      pose_hand=pose_hand.cuda(),  # (1, 6)
                      betas=betas.cuda(),  # (1, 10)
                      trans=torch.from_numpy(trans).reshape((1, 3)).cuda()
                      )  # (1, 3)

    # TODO: get bone transforms
    bone_transforms = body.bone_transforms.float()  # (1, 24, 4, 4)

    # bone_transforms = data['bone_transforms'][0].astype(np.float32)  # (24, 4, 4)
    # bone_transforms = np.tile(np.eye(4), (24, 1, 1))  # (24, 4, 4)  # 全1,A-pose
    return rots, Jtrs, bone_transforms


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)  # disable set struct
    config.model.pose_correction = {'name': "none", "delay": 5000}  # Don't use Pose Correction

    # setup gaussian model
    gaussian_model = GaussianModel(config.model.gaussian)

    # load Scene
    # config.model.texture.latent_dim = 0
    scene = DuckDuckScene(config, gaussian_model)
    scene.eval()
    scene.load_checkpoint("./exp/zju_377_mono-direct-mlp_field-ingp-shallow_mlp-default/ckpt15000.pth")

    # background color
    bg_color = torch.tensor([0., 0., 0.], dtype=torch.float32).cuda()  # black

    # load smpl model data
    rots, Jtrs, bone_transforms = load_model("./gym_clip_cliff_hr48.npz")
    # rots (1, 24, 9)
    # Jtrs (1, 24, 3)
    # bone_transforms (1, 24, 4, 4)

    # TODO: Build A Duck Camera Class...
    # data: {FoVx, FoVy, image_height, image_width, world_view_transform, full_proj_transform, camera_center}
    R = torch.tensor([[-0.72163749, -0.28632199, 0.61297983],
                      [0.69329592, -0.28599222, 0.70034063],
                      [-0.01087123, 0.91537363, 0.36575785]])
    T = torch.tensor([0.45483854, 0.82210373, 2.92526793])

    cam = Camera(R, T)
    cam.update_pose(rots.cuda(), Jtrs.cuda(), bone_transforms.cuda())

    render_pkg = render(cam, 15000, scene, config.pipeline, bg_color, compute_loss=False, return_opacity=False)

    rendering = render_pkg["render"]
    # save as png
    # torchvision.utils.save_image(rendering, f"out_rendering.png")

    frame = rendering.cpu().numpy().transpose(1, 2, 0) * 255.0
    frame = frame.astype(np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Rendered Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    with torch.no_grad():
        main()
