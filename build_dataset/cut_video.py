import cv2
import numpy as np

def convert_time_to_frame_index(time_str, fps):
    """将时间格式 (min:sec) 转换为帧索引"""
    minutes, seconds = map(int, time_str.split(':'))
    return int((minutes * 60 + seconds) * fps)

def extract_video_segment_and_save(cap, start_time, end_time, fps, video_writer):
    """提取给定时间段的视频片段并直接写入新的视频文件"""
    start_frame = convert_time_to_frame_index(start_time, fps)
    end_frame = convert_time_to_frame_index(end_time, fps)

    # 设置视频的读取位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # 如果需要，可以在这里对帧进行调整，比如缩放等
        # frame = cv2.resize(frame, (width, height))

        # 将每一帧写入新的视频文件
        video_writer.write(frame)

if __name__ == '__main__':
    time_segments = [
        ("0:41", "0:46"),
        ("1:04", "1:16"),
        ("1:42", "2:07"),
        ("2:29", "2:59"),
        ("3:13", "3:23"),
        ("3:43", "4:30"),
        ("4:48", "5:09"),
        ("5:18", "6:02"),
        ("6:22", "6:45"),
        ("7:23", "8:30"),
        ("8:40", "8:57"),
        ("9:08", "9:18"),
        ("9:36", "11:27"),
        ("12:06", "13:42"),
        ("14:09", "16:06"),
        ("16:50", "17:38"),
        ("18:17", "18:29"),
        ("18:41", "19:30"),
        ("19:50", "20:00"),
        ("20:21", "20:32"),
        ("20:46", "21:00"),
        ("21:16", "21:40"),
        ("21:54", "22:14"),
        ("22:27", "22:40"),
        ("23:18", "23:38"),
        ("23:50", "25:20"),
        ("25:29", "25:55"),
        ("26:10", "26:34"),
        ("27:02", "27:12"),
        ("27:32", "27:45"),
        ("28:00", "28:17")
    ]

    cap = cv2.VideoCapture("../videos/dance.mp4")
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS of the video: {fps}")

    # 获取视频宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入对象，输出路径为 'dance2.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video_writer = cv2.VideoWriter("../videos/dance2.mp4", fourcc, fps, (width, height))

    # 提取每个时间片段的帧并保存到新视频文件
    for start_time, end_time in time_segments:
        print(f"Extracting from {start_time} to {end_time}")
        extract_video_segment_and_save(cap, start_time, end_time, fps, video_writer)

    # 释放视频资源
    cap.release()
    video_writer.release()
    print("New video saved as 'dance2.mp4'")
