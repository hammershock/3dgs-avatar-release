import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
matplotlib.use('tkAgg')

# 定义骨骼连线的关系（每对索引为骨骼的连接）
bone_connections = [
    (0, 1), (1, 2), (2, 3),  # 左肩、上臂、肘部、前臂
    (0, 4), (4, 5), (5, 6),  # 右肩、上臂、肘部、前臂
    (0, 7), (7, 8), (8, 9),  # 脊柱连线
    (0, 10), (10, 11), (11, 12),  # 颈部和头部
    (3, 14), (14, 15), (15, 16),  # 左腿
    (6, 17), (17, 18), (18, 19),  # 右腿
    (9, 20), (20, 21), (21, 22),  # 左脚
    (12, 23)  # 右脚
]


class CameraFrustumVisualizer:
    def __init__(self, R, T, image_width, image_height, fovx, fovy, znear, zfar):
        """
        初始化相机光锥体可视化器

        参数:
            R: 3x3 相机旋转矩阵
            T: 3x1 相机平移向量
            image_width: 图像宽度
            image_height: 图像高度
            fovx: 水平视场角 (弧度)
            fovy: 垂直视场角 (弧度)
            znear: 近平面距离
            zfar: 远平面距离
        """
        self.R = R
        self.T = T
        self.image_width = image_width
        self.image_height = image_height
        self.fovx = fovx
        self.fovy = fovy
        self.znear = znear
        self.zfar = zfar

        # 计算焦距
        self.focal_length_x = 0.5 * image_width / np.tan(fovx / 2)
        self.focal_length_y = 0.5 * image_height / np.tan(fovy / 2)

    def plot_frustum(self, joints):
        """
        绘制相机的光锥体
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 定义相机的位置（世界坐标系中的位置）
        camera_position = -np.dot(self.R.T, self.T)  # 相机位置 = -R^T * T

        # 计算视锥体的四个角点
        near_top_left = np.array([-self.focal_length_x * self.znear / self.image_width,
                                  self.focal_length_y * self.znear / self.image_height,
                                  -self.znear])
        near_top_right = np.array([self.focal_length_x * self.znear / self.image_width,
                                   self.focal_length_y * self.znear / self.image_height,
                                   -self.znear])
        near_bottom_left = np.array([-self.focal_length_x * self.znear / self.image_width,
                                     -self.focal_length_y * self.znear / self.image_height,
                                     -self.znear])
        near_bottom_right = np.array([self.focal_length_x * self.znear / self.image_width,
                                      -self.focal_length_y * self.znear / self.image_height,
                                      -self.znear])

        # 将四个角点从相机坐标系转换到世界坐标系
        near_top_left_world = np.dot(self.R, near_top_left) + camera_position
        near_top_right_world = np.dot(self.R, near_top_right) + camera_position
        near_bottom_left_world = np.dot(self.R, near_bottom_left) + camera_position
        near_bottom_right_world = np.dot(self.R, near_bottom_right) + camera_position

        # 绘制近平面
        ax.plot([near_top_left_world[0], near_top_right_world[0]],
                [near_top_left_world[1], near_top_right_world[1]],
                [near_top_left_world[2], near_top_right_world[2]], color='b')
        ax.plot([near_top_left_world[0], near_bottom_left_world[0]],
                [near_top_left_world[1], near_bottom_left_world[1]],
                [near_top_left_world[2], near_bottom_left_world[2]], color='b')
        ax.plot([near_bottom_left_world[0], near_bottom_right_world[0]],
                [near_bottom_left_world[1], near_bottom_right_world[1]],
                [near_bottom_left_world[2], near_bottom_right_world[2]], color='b')
        ax.plot([near_top_right_world[0], near_bottom_right_world[0]],
                [near_top_right_world[1], near_bottom_right_world[1]],
                [near_top_right_world[2], near_bottom_right_world[2]], color='b')

        # 计算远平面四个角点
        far_top_left = near_top_left * self.zfar / self.znear
        far_top_right = near_top_right * self.zfar / self.znear
        far_bottom_left = near_bottom_left * self.zfar / self.znear
        far_bottom_right = near_bottom_right * self.zfar / self.znear

        # 将远平面角点从相机坐标系转换到世界坐标系
        far_top_left_world = np.dot(self.R, far_top_left) + camera_position
        far_top_right_world = np.dot(self.R, far_top_right) + camera_position
        far_bottom_left_world = np.dot(self.R, far_bottom_left) + camera_position
        far_bottom_right_world = np.dot(self.R, far_bottom_right) + camera_position

        # 绘制远平面
        ax.plot([far_top_left_world[0], far_top_right_world[0]],
                [far_top_left_world[1], far_top_right_world[1]],
                [far_top_left_world[2], far_top_right_world[2]], color='r')
        ax.plot([far_top_left_world[0], far_bottom_left_world[0]],
                [far_top_left_world[1], far_bottom_left_world[1]],
                [far_top_left_world[2], far_bottom_left_world[2]], color='r')
        ax.plot([far_bottom_left_world[0], far_bottom_right_world[0]],
                [far_bottom_left_world[1], far_bottom_right_world[1]],
                [far_bottom_left_world[2], far_bottom_right_world[2]], color='r')
        ax.plot([far_top_right_world[0], far_bottom_right_world[0]],
                [far_top_right_world[1], far_bottom_right_world[1]],
                [far_top_right_world[2], far_bottom_right_world[2]], color='r')

        # 绘制从相机到远平面的连线
        ax.plot([camera_position[0], near_top_left_world[0]],
                [camera_position[1], near_top_left_world[1]],
                [camera_position[2], near_top_left_world[2]], color='g')
        ax.plot([camera_position[0], near_top_right_world[0]],
                [camera_position[1], near_top_right_world[1]],
                [camera_position[2], near_top_right_world[2]], color='g')
        ax.plot([camera_position[0], near_bottom_left_world[0]],
                [camera_position[1], near_bottom_left_world[1]],
                [camera_position[2], near_bottom_left_world[2]], color='g')
        ax.plot([camera_position[0], near_bottom_right_world[0]],
                [camera_position[1], near_bottom_right_world[1]],
                [camera_position[2], near_bottom_right_world[2]], color='g')

        # 绘制从近平面四个角点到远平面四个角点的连线
        ax.plot([near_top_left_world[0], far_top_left_world[0]],
                [near_top_left_world[1], far_top_left_world[1]],
                [near_top_left_world[2], far_top_left_world[2]], color='c')
        ax.plot([near_top_right_world[0], far_top_right_world[0]],
                [near_top_right_world[1], far_top_right_world[1]],
                [near_top_right_world[2], far_top_right_world[2]], color='c')
        ax.plot([near_bottom_left_world[0], far_bottom_left_world[0]],
                [near_bottom_left_world[1], far_bottom_left_world[1]],
                [near_bottom_left_world[2], far_bottom_left_world[2]], color='c')
        ax.plot([near_bottom_right_world[0], far_bottom_right_world[0]],
                [near_bottom_right_world[1], far_bottom_right_world[1]],
                [near_bottom_right_world[2], far_bottom_right_world[2]], color='c')

        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c='r', marker='o', label='Joints')
        if joints.shape[0] == 24:
            # 绘制骨骼连线
            for (start_idx, end_idx) in bone_connections:
                ax.plot([joints[start_idx, 0], joints[end_idx, 0]],
                        [joints[start_idx, 1], joints[end_idx, 1]],
                        [joints[start_idx, 2], joints[end_idx, 2]], color='g', lw=2)

        # 设置图形的显示范围
        # ax.set_xlim([-3, 3])
        # ax.set_ylim([-3, 3])
        # ax.set_zlim([-3, 3])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()

    def plot_frustum_and_skeleton(self, joints):
        """
        绘制光锥体和骨骼关键点连线
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制相机光锥体
        self.plot_frustum(ax)

        # 骨骼关键点


        # 设置图形的显示范围
        # ax.set_xlim([-10, 10])
        # ax.set_ylim([-10, 10])
        # ax.set_zlim([-10, 10])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()


def plot_pts3d(pts3d):
    """pt3ds: np.ndarray(N, 3)"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c='r', marker='o', label='Joints')
    # 设置图形的显示范围
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == "__main__":
    # 使用示例
    R = np.eye(3)  # 相机的旋转矩阵（单位矩阵，表示没有旋转）
    T = np.array([0, 0, 5])  # 相机的平移向量（相机位置在Z轴上5单位）
    image_width = 800
    image_height = 600
    fovx = np.radians(60)  # 水平视场角为60度
    fovy = np.radians(45)  # 垂直视场角为45度
    znear = 0.1  # 近平面
    zfar = 100.0  # 远平面

    # 创建并可视化相机光锥体
    visualizer = CameraFrustumVisualizer(R, T, image_width, image_height, fovx, fovy, znear, zfar)
    visualizer.plot_frustum()
