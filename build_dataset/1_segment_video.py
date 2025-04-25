import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
    model = YOLO('yolov8x-seg.pt')
    batch_size = 8
    masks = []
    idx = 0
    # idx_limit = 3000
    kernel = np.ones((3, 3), np.uint8)
    with VideoStream(video_path) as video:
        total = video.frame_count // batch_size # len(video) // batch_size
        for batch in tqdm(load_data(video, batch_size=batch_size), total=total):
            results = model.predict(batch, verbose=False, conf=0.5, classes=0, imgsz=(1088, 1920), retina_masks=True)
            # if idx >= idx_limit:
            #     break
            for i, result in enumerate(results):
                image = batch[i].copy()
                if result.masks is None:
                    masks.append(np.zeros_like(image[:, :, 0], dtype=bool))
                    break
                mask = (result.masks.data[0].cpu().numpy() * 255.0).astype(np.uint8)

                binary_mask = (mask > 127).astype(np.uint8)  # 使用布尔类型代替np.uint8
                binary_mask = cv2.erode(binary_mask, kernel, iterations=3)
                binary_mask = binary_mask.astype(bool)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image[~binary_mask] = 0
                cv2.imshow("masked image", image)
                cv2.pollKey()
                masks.append(binary_mask)
                idx += 1

    masks = np.array(masks)
    np.save("masks_dance2.npy", masks)
    cv2.destroyAllWindows()
