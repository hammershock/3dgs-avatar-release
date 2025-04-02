import os

import numpy as np
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib
from scipy.spatial.transform import Rotation


matplotlib.use('tkAgg')

rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()


# add ZJUMoCAP dataloader
def get_02v_bone_transforms(Joints):
    """
    Specify the bone transformations that transform a SMPL A-pose mesh
    to a star-shaped A-pose (i.e. Vitruvian A-pose)
    Args:
        Joints: (24, 3)
    Returns:
        transforms: (24, 4, 4)
    """
    trans = np.tile(np.eye(4), (24, 1, 1))  # (4, 4) -> (24, 4, 4)

    def rotate(chain, R):
        for i, joint_idx in enumerate(chain):
            trans[joint_idx, :3, :3] = R
            t = Joints[joint_idx].copy()  # current joint
            if i > 0:
                parent = chain[i - 1]
                t_p = Joints[parent].copy()  # parent joint
                t = R @ (t - t_p)  # t = np.dot(rot, t - t_p)  #
                t += trans[parent, :3, -1].copy()
            trans[joint_idx, :3, -1] = t
        trans[chain, :3, -1] -= np.dot(Joints[chain], R.T)

    rotate([1, 4, 7, 10], rot45p)  # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    rotate([2, 5, 8, 11], rot45n)  # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)

    return trans


def load_smpl(model_path):
    data = np.load(model_path, allow_pickle=True)
    # minimal_shape = data['minimal_shape']
    # betas = data['betas']
    # Jtr_posed = data['Jtr_posed']
    # bone_transforms = data['bone_transforms']
    # trans = data['trans']
    # root_orient = data['root_orient']
    # pose_body = data['pose_body']
    # pose_hand = data['pose_hand']
    return data


def visualize(vertices, faces):
    # 3D 可视化
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 3D 网格
    mesh = Poly3DCollection(vertices[faces], alpha=0.1, edgecolor="k")
    ax.add_collection3d(mesh)

    # 设定坐标轴
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=10, azim=45)  # 设置观察角度

    # # 坐标轴范围
    # x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    # y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    # z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.title("SMPL 3D Model")
    plt.show()


def visualize_edges(vertices, faces):
    """
    仅使用 Line3DCollection 绘制 3D 网格的边。

    参数：
    - vertices: (N, 3) SMPL 顶点坐标
    - faces: (M, 3) 三角面索引
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 生成网格的边
    edges = []
    for face in faces:
        for i in range(3):  # 每个三角形有 3 条边
            start, end = face[i], face[(i + 1) % 3]
            edges.append([vertices[start], vertices[end]])

    # 创建 Line3DCollection（黑色边）
    edge_collection = Line3DCollection(edges, colors='k', linewidths=0.5)
    ax.add_collection3d(edge_collection)

    # 设定坐标轴
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=10, azim=45)  # 设置观察角度

    # 坐标轴范围
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    plt.title("SMPL 3D Model (Wireframe)")
    plt.show()


def _load_smpl_model(base_dir="./body_models/misc"):
    faces: np.ndarray = np.load(os.path.join(base_dir, 'faces.npz'))['faces']  # (n, 3)
    skinning_weights = dict(np.load(os.path.join(base_dir, 'skinning_weights_all.npz')))
    posedirs = dict(np.load(os.path.join(base_dir, 'posedirs_all.npz')))
    J_regressor = dict(np.load(os.path.join(base_dir, 'J_regressors.npz')))
    return faces, skinning_weights, posedirs, J_regressor


def _mediapipe_to_smpl(mediapipe_joints):
    ...
    """
    将 MediaPipe 关键点转换为 SMPL 模型参数
    :param mediapipe_joints: (33, 3) 的 MediaPipe 3D 关键点
    :return: 
        trans: 整体平移
        root_orient:
        pose_body:
        pose_hand:
    """
    smpl_parents = np.array(
        [-1, 0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 17, 18])  # SMPL 关节父子关系

    # 1️⃣ 关键点映射
    smpl_to_mediapipe = [23, 23, 24, 11, 25, 26, 9, 27, 28, 0, 11, 12, 13, 14, 15, 16]
    joints_smpl = np.array([mediapipe_joints[i] for i in smpl_to_mediapipe])

    # 2️⃣ 计算 trans
    trans = joints_smpl[0]

    # 3️⃣ 计算 root_orient
    hip_vector = joints_smpl[1] - joints_smpl[2]
    hip_vector /= np.linalg.norm(hip_vector)
    spine_vector = joints_smpl[3] - joints_smpl[0]
    spine_vector /= np.linalg.norm(spine_vector)

    root_rotation_matrix = np.eye(3)
    root_rotation_matrix[:, 0] = hip_vector
    root_rotation_matrix[:, 1] = np.cross(spine_vector, hip_vector)
    root_rotation_matrix[:, 2] = spine_vector

    root_orient = Rotation.from_matrix(root_rotation_matrix).as_rotvec()

    # 4️⃣ 计算 pose_body
    pose_body = np.zeros((21, 3))
    joint_rotations = [Rotation.identity() for _ in range(24)]  # 假设所有关节的旋转矩阵

    for i in range(1, 22):
        parent = smpl_parents[i]
        local_rotation = Rotation.from_matrix(
            np.linalg.inv(joint_rotations[parent].as_matrix()) @ joint_rotations[i].as_matrix()
        ).as_rotvec()
        pose_body[i - 1] = local_rotation

    # 5️⃣ 计算 pose_hand
    pose_hand = np.zeros(6)
    pose_hand[:3] = Rotation.from_matrix(joint_rotations[20].as_matrix()).as_rotvec()
    pose_hand[3:] = Rotation.from_matrix(joint_rotations[21].as_matrix()).as_rotvec()

    trans = trans.astype(np.float32)
    root_orient = root_orient.astype(np.float32)
    pose_body = pose_body.astype(np.float32)
    pose_hand = pose_hand.astype(np.float32)

    return trans, root_orient, pose_body, pose_hand


def load_mediapipe(model_path):
    data = np.load(model_path, allow_pickle=True)
    # data = np.load(file_path, allow_pickle=True)
    kpts = data['keypoints']
    # frame_ids = data['frame_ids']
    return kpts


def fix_random_issues(arr):
    if arr.dtype == np.float16:
        return arr.astype(np.float32) + 1e-4 * np.random.randn(*arr.shape)
    return arr.astype(np.float32)


def _get_cano_smpl_verts(minimal_shape, J_regressor, skinning_weights, faces):
    """
    将SMPL最基础的模型形状转换到标准姿态（大字型）
    Compute star-posed SMPL body vertices.
    To get a consistent canonical space,
    we do not add pose blend shape
    """
    J0 = np.dot(J_regressor, minimal_shape)
    # Get bone transformations that transform a SMPL A-pose mesh to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = get_02v_bone_transforms(J0)  # (24, 4, 4)
    T = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])  # (6890, 4, 4)
    vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]  # (6890, 3)

    cano_mesh = trimesh.Trimesh(vertices=vertices.astype(np.float32), faces=faces)

    return {
        'smpl_verts': vertices.astype(np.float32),
        'minimal_shape': minimal_shape,
        'Jtr': J0,
        'skinning_weights': skinning_weights.astype(np.float32),
        'bone_transforms_02v': bone_transforms_02v,
        'cano_mesh': cano_mesh,
    }



def bone_transform(skinning_weights, transforms):
    T = np.matmul(skinning_weights, transforms.reshape([-1, 16])).reshape([-1, 4, 4])  # (6890, 4, 4)
    vertices = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]  # (6890, 3)
    return vertices


if __name__ == '__main__':
    file_path = "./data/ZJUMoCap/CoreView_377/models/000001.npz"
    data = load_smpl(file_path)
    faces, skinning_weights, posedirs, J_regressor = _load_smpl_model()  # 基本参数
    minimal_shape = fix_random_issues(data['minimal_shape'])  # 基本形状
    J0 = np.dot(J_regressor['neutral'], minimal_shape)
    T = get_02v_bone_transforms(J0)
    V = bone_transform(skinning_weights['neutral'], T)

    # V = bone_transform(skinning_weights['neutral'], data['bone_transforms'])
    visualize(V, faces)
    # # _get_cano_smpl_verts(data['minimal_shape'], J_regressor['neutral'], skinning_weights['neutral'], faces)
    # mediapipe_path = "./gym.npz"
    # kpts = load_mediapipe(mediapipe_path)
    # mediapipe_data = kpts[30]  # 第30帧的数据
    # print(mediapipe_data.shape)
    # # trans, root_orient, pose_body, pose_hand = _mediapipe_to_smpl(kpts)
    #

