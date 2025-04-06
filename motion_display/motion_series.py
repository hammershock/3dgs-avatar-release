import numpy as np
from typing import NamedTuple


class SMPLParameters(NamedTuple):
    pose: np.ndarray  #
    shape: np.ndarray
    global_t: np.ndarray
    pred_joints: np.ndarray
    focal_l: np.ndarray


class MotionSeries:
    def __init__(self, filepath):
        self.data = np.load(filepath)  # npz file path

        for key in ['pose', 'shape', 'global_t', 'pred_joints', 'focal_l']:
            assert key in self.data.keys(), "missing key '{}'".format(key)

        self._index = 0

    def __len__(self):
        return len(self.data['pose'])

    def __getitem__(self, idx) -> SMPLParameters:
        return SMPLParameters(
            pose=self.data['pose'][idx],
            shape=self.data['shape'][idx],
            global_t=self.data['global_t'][idx],
            pred_joints=self.data['pred_joints'][idx],
            focal_l=self.data['focal_l'][idx]
        )

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
        result = self[self._index]
        self._index += 1
        return result


if __name__ == '__main__':
    npz_file_path = "your/npz/filepath.npz"
    series = MotionSeries(npz_file_path)
    from itertools import cycle
    for item in cycle(series):
        print(item)
        # THIS IS AN ENDLESS LOOP...