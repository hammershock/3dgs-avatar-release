"""
预处理，YOLO视频图像分割
"""
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from motion_display import VideoStream


def load_data(video, batch_size=32):
    images = []
    for frame in video:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(image_rgb)
        if len(images) == batch_size:
            yield images
            images = []
    if images:
        yield images

if __name__ == '__main__':
    video_path = "../videos/dance2.mp4"
    out_path = "./masks_dance2.npy"
    model_path = "./yolov8x-seg.pt"
    out_video_path = "../videos/dance2_out.mp4"
    batch_size = 8

    model = YOLO(model_path)

    masks = []
    kernel = np.ones((3, 3), np.uint8)

    with VideoStream(video_path) as video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式编码器
        frame_size = (video.width, video.height)
        video_writer = cv2.VideoWriter(out_video_path, fourcc, video.fps, frameSize=frame_size)

        total = video.frame_count // batch_size
        for batch in tqdm(load_data(video, batch_size=batch_size), total=total):
            results = model.predict(batch, verbose=False, conf=0.5, classes=0, imgsz=(1088, 1920), retina_masks=True)
            for i, result in enumerate(results):
                image = batch[i].copy()
                if result.masks is None:
                    continue
                mask = (result.masks.data[0].cpu().numpy() * 255.0).astype(np.uint8)
                binary_mask = (mask > 127).astype(np.uint8)  # 使用布尔类型代替np.uint8
                binary_mask = cv2.erode(binary_mask, kernel, iterations=3)
                binary_mask = binary_mask.astype(bool)
                masks.append(binary_mask)
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

                # # visualize
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # image[~binary_mask] = 0
                # cv2.imshow("masked image", image)
                # cv2.pollKey()

    masks = np.array(masks)
    np.save(out_path, masks)
    video_writer.release()
    # cv2.destroyAllWindows()
