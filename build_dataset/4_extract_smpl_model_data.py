import os

import numpy as np
import torch
from tqdm.contrib import tenumerate

from human_body_prior.body_model.body_model import BodyModel
from motion_display import MotionSeries


if __name__ == '__main__':
    # in_models_path = "../scene_sequences/dance2_out_cliff_hr48.npz"
    in_models_path = "../scene_sequences/cxk_clip_cliff_hr48.npz"
    out_models_dir = "../data/ZJUMoCap/CoreView_001/models"


    # load basic bodyModel
    basic_smpl_model_path = '../body_models/smpl/neutral/model.pkl'
    faces_path = '../body_models/misc/faces.npz'
    body_model = BodyModel(bm_path=basic_smpl_model_path, num_betas=10, batch_size=1).cuda()
    faces = np.load(faces_path)['faces']

    root_orient = torch.pi * torch.tensor([1., 0., 0.]).unsqueeze(0),  # override root_orient
    trans = torch.zeros((1, 3)).float().cuda(),  # override trans
    series = MotionSeries(in_models_path, root_orient=root_orient, trans=trans)

    os.makedirs(out_models_dir, exist_ok=True)

    for i, smpl_params in tenumerate(series):
        name = str(i).zfill(6)
        out_models_path = os.path.join(out_models_dir, f"{name}.npz")
        smpl_params.export(out_models_path)





