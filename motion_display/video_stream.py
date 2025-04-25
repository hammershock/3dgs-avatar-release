import cv2
import numpy as np
import warnings


def estimate_focal_length(img_h, img_w):
    return (img_w * img_w + img_h * img_h) ** 0.5  # fov: 55 degree


class VideoStream:
    def __init__(self, video_path, width=None, height=None):
        self.video_path = video_path
        self.resize = (width is not None) and (height is not None)
        self.width = width
        self.height = height
        self.cap = None
        self.frame_count = None
        self.fps = None

        self.fx, self.fy = None, None
        self.K = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video at {self.video_path}")
        else:
            print(f"Successfully opened video at {self.video_path}")

        # 获取视频的总帧数
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # 如果没有指定宽度和高度，则根据视频实际尺寸设置
        if self.width is None or self.height is None:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.fx = self.fy = estimate_focal_length(self.height, self.width)
        self.cx = self.width / 2
        self.cy = self.height / 2

        self.K = np.array([[self.fx, 0., self.cx],
                           [0., self.fy, self.cy],
                           [0, 0, 1]], dtype=np.float32)

        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __getitem__(self, index):
        """返回视频中的某一帧，若指定了尺寸则进行缩放"""
        if index < 0 or index >= self.frame_count:
            raise IndexError(f"Index {index} is out of range for the video with {self.frame_count} frames.")

        # 设置视频读取的当前位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        ret, frame = self.cap.read()
        if not ret:
            warnings.warn(f"Error: Failed to read frame at index {index}")
            raise IndexError(f"Cannot read frame at index {index}")

        # 如果指定了width和height，进行缩放
        if self.resize:
            frame = cv2.resize(frame, (self.width, self.height))

        return frame

    def __len__(self):
        """返回视频的总帧数"""
        return self.frame_count

    def __iter__(self):
        """支持迭代访问视频的每一帧"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置视频读取到第一个帧
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            # warnings.warn("Error: Failed to capture image")
            raise StopIteration

        # 如果指定了width和height，进行缩放
        if self.resize:
            frame = cv2.resize(frame, (self.width, self.height))

        return frame



def save_images_as_video(image_list, output_video_path, fps=30, frame_size=None):
    """
    将一组图像保存为 MP4 视频文件

    参数:
        image_list (list): OpenCV 图像列表，每个元素是一个 numpy 数组，表示一帧图像
        output_video_path (str): 输出视频文件的路径（例如 'output.mp4'）
        fps (int): 视频的帧率（默认为 30）
        frame_size (tuple): 视频的帧尺寸 (width, height)，如果为 None，则根据第一个图像的尺寸自动设定
    """
    if not image_list:
        raise ValueError("图像列表不能为空")

    # 获取第一张图像的尺寸（如果没有提供的话）
    if frame_size is None:
        height, width, _ = image_list[0].shape
        frame_size = (width, height)

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式编码器
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # 将每一帧图像写入视频文件
    for img in image_list:
        video_writer.write(img)

    # 释放视频写入对象
    video_writer.release()
    print(f"视频已保存到 {output_video_path}")