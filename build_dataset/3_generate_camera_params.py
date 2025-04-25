import json

import numpy as np

from common.utils import estimate_focal_length

if __name__ == "__main__":
    # input_video width and height
    width = 1920
    height = 1080
    out_path = "../data/ZJUMoCap/CoreView_001/cam_params.json"

    f = estimate_focal_length(height, width)
    K = np.array([[f, 0., width / 2],
                  [0., f, height / 2],
                  [0, 0, 1]], dtype=np.float32)
    D = np.zeros((5, 1), dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    T = np.zeros((3, 1), dtype=np.float32)
    data = {
        "1": {"K": K.tolist(), "D": D.tolist(), "R": R.tolist(), "T": T.tolist()},
        "all_cam_names": ['1']
    }

    with open(out_path, 'w') as f:
        json.dump(data, f)

