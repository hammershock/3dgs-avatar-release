import cv2
from tqdm.contrib import tenumerate

from motion_display import VideoStream

if __name__ == '__main__':
    video_path = "../videos/dance2_backup.mp4"
    output_video_path = "../videos/dance2.mp4"

    with VideoStream(video_path) as video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式编码器
        frame_size = (video.width, video.height)
        video_writer = cv2.VideoWriter(output_video_path, fourcc, video.fps, frameSize=frame_size)
        for i, frame in tenumerate(video):
            if i % 10 == 0:
                video_writer.write(frame)

        video_writer.release()

