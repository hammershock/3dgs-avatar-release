"""generate mask"""
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from glob import glob
import os


def load_data(image_paths, batch_size=32):
    images = []
    jpg_paths = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image_rgb)
        jpg_paths.append(image_path)
        if len(images) == batch_size:
            yield images, jpg_paths
            images = []
            jpg_paths = []
    if images:
        yield images, jpg_paths


if __name__ == '__main__':
    batch_size = 8

    model = YOLO("./runs/segment/train16/weights/best.pt")
    print(f"model loaded")
    image_paths = glob("../data/ZJUMoCap/CoreView_001/1/*.jpg")
    image_paths.sort()
    image_paths = image_paths[1600:]
    p_bar = tqdm(image_paths, desc="processing images")
    for image_path in p_bar:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = model.predict(image_rgb, verbose=False, conf=0.5, classes=0, imgsz=1088, retina_masks=True)
        result = results[0]
        if result.masks is None:
            print(f"skip {image_path}")
            continue

        mask = (result.masks.data[0].cpu().numpy() * 255.0).astype(np.uint8)
        binary_mask = (mask > 127).astype(np.uint8)  # 使用布尔类型代替np.uint8
        basename = os.path.basename(image_path).split(".")[0]
        # cv2.imwrite(f"./seg_data/masks/{basename}.png", binary_mask)
        image_show = image.copy()
        image_show[mask < 127] = 0
        cv2.imshow("mask", binary_mask * 255)
        cv2.waitKey(1)
        p_bar.set_postfix(processing=f"{basename}.png")
        path = f"../data/ZJUMoCap/CoreView_001/1/{basename}.png"
        if os.path.exists(f"../data/ZJUMoCap/CoreView_001/{basename}.png"):
            os.remove(path)
        cv2.imwrite(path, binary_mask * 255)

    cv2.destroyAllWindows()
