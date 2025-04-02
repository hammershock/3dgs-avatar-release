from typing import TypedDict, Literal, Dict, List
import numpy as np
from numpy.typing import NDArray
import trimesh

GenderLiteral = Literal['male', 'female', 'neutral']

class MetadataType(TypedDict):
    # ================================= METADATA ====================================================
    # faces: ndarray(13776, 3)
    # posedirs: {'male': ndarray(6890, 3, 207), 'female': ..., 'neutral': ...}
    # J_regressor: {'male': ndarray(24, 6890), 'female': ..., 'neutral': ...}
    # camera_extent: 3.469298553466797 # hard coded
    # frame_dict: {int->int}
    # gender: 'neutral'
    # smpl_verts: ndarray(6890, 3)
    # minimal_shape: ndarray(6890, 3)
    # Jtr: ndarray(24, 3)
    # skinning_weights: ndarray(6890, 24)
    # bone_transform_O2v: ndarray(24, 4, 4)
    # cano_mesh: Trimesh(faces=(13776, 3), vertices=ndarray(6890, 3))
    # coord_min: ndarray(3, )
    # coord_max: ndarray(3, )
    # betas: ndarray(1, 10)
    # frames: List(int * 570)
    # root_orient List(ndarray(3) * 570)
    # pose_body List(ndarray(63) * 570)
    # pose_hand List(ndarray(6) * 570)
    # trans List(ndarray(3) * 570)
    # ======================================================================================================

    faces: NDArray[np.int32]                      # (13776, 3)
    posedirs: Dict[GenderLiteral, NDArray[np.float32]]  # (6890, 3, 207)
    J_regressor: Dict[GenderLiteral, NDArray[np.float32]]  # (24, 6890)
    camera_extent: float
    frame_dict: Dict[int, int]
    gender: GenderLiteral
    smpl_verts: NDArray[np.float32]               # (6890, 3)
    minimal_shape: NDArray[np.float32]            # (6890, 3)
    Jtr: NDArray[np.float32]                      # (24, 3)
    skinning_weights: NDArray[np.float32]         # (6890, 24)
    bone_transform_O2v: NDArray[np.float32]       # (24, 4, 4)
    cano_mesh: trimesh.Trimesh                    # faces=(13776,3), vertices=(6890,3)
    coord_min: NDArray[np.float32]                # (3,)
    coord_max: NDArray[np.float32]                # (3,)
    betas: NDArray[np.float32]                    # (1, 10)
    frames: List[int]                             # 570 items
    root_orient: List[NDArray[np.float32]]        # (3,) * 570
    pose_body: List[NDArray[np.float32]]          # (63,) * 570
    pose_hand: List[NDArray[np.float32]]          # (6,) * 570
    trans: List[NDArray[np.float32]]              # (3,) * 570
