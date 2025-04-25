from tqdm.contrib import tenumerate
import os
from tqdm import tqdm
from motion_display import VideoStream
import numpy as np
import cv2


if __name__ == '__main__':
    # 3. make smpl data
    npz_data = np.load("../data/ZJUMoCap/CoreView_377/models/000001.npz")
    print(list(npz_data.keys()))
    # ['minimal_shape', 'betas', 'Jtr_posed', 'bone_transforms', 'trans', 'root_orient', 'pose_body', 'pose_hand']

    from human_body_prior.body_model.body_model import BodyModel
    import numpy as np
    import torch

    basic_smpl_model_path = '../body_models/smpl/neutral/model.pkl'
    faces_path = '../body_models/misc/faces.npz'

    # load basic bodyModel
    body_model = BodyModel(bm_path=basic_smpl_model_path, num_betas=10, batch_size=1).cuda()
    faces = np.load(faces_path)['faces']

    # ['detection_all', 'focal_l', 'pose', 'shape', 'global_orient', 'global_t', 'pred_joints']
    from motion_display import MotionSeries

    series = MotionSeries("../scene_sequences/dance2_cliff_hr48.npz")
    for i, smpl_params in tenumerate(series):
        betas = torch.from_numpy(smpl_params.shape).float().cuda().unsqueeze(0)
        # Get shape vertices
        body = body_model(betas=betas)
        minimal_shape = body.v.detach().cpu().numpy()[0]  # 1. minimal shape in T pose, given betas
        poses = smpl_params.pose[None, ...]
        pose_body = poses[:, 3:66].copy()  #
        pose_hand = poses[:, 66:].copy()  #
        trans = smpl_params.global_t

        # set root orient, pose
        # body = body_model(root_orient=smpl_params.root_orient,  # (1, 3) !
        #                   pose_body=torch.from_numpy(pose_body).float().cuda(),  # (1, 63)
        #                   pose_hand=torch.from_numpy(pose_hand).float().cuda(),  # (1, 6)
        #                   betas=betas,  # (10, )
        #                   trans=torch.from_numpy(trans).unsqueeze(0).float().cuda())  # (1, 3)
        #
        # vertices = body.v.detach().cpu().numpy()[0]  # posed verti

        # Jtr_posed(1, 24, 3)(24, 3)
        # bone_transforms(1, 24, 4, 4)(24, 4, 4)
        # ['minimal_shape', 'betas', 'Jtr_posed', 'bone_transforms', 'trans', 'root_orient', 'pose_body', 'pose_hand']
        data = dict(
            minimal_shape=minimal_shape,
            betas=betas.cpu().detach().numpy(),
            Jtr_posed=smpl_params.Jtr_posed.detach().squeeze().cpu().numpy(),
            bone_transforms=smpl_params.bone_transforms_orig.squeeze().detach().cpu().numpy(),
            trans=smpl_params.global_t,
            root_orient=smpl_params.root_orient[0].detach().cpu().numpy(),
            pose_body=pose_body[0],
            pose_hand=pose_hand[0]
        )
        name = str(i).zfill(6)
        os.makedirs("../data/ZJUMoCap/CoreView_001/models", exist_ok=True)
        np.savez(f"../data/ZJUMoCap/CoreView_001/models/{name}.npz", **data)



