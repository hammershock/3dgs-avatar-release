"""convert png mask to YOLO .txt format"""
import os

import cv2
import numpy as np
import shutil
from glob import glob
from tqdm import tqdm
from tqdm.contrib import tenumerate


def generate_txt(png_mask_file_path, out_txt_file_path):
    """generate txt file from png mask"""
    img = cv2.imread(png_mask_file_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_recover = np.zeros((h, w), dtype=np.uint8)
    # The YOLO format doesn't allow holes in mask. So when it's converted to segments/polygons, the information is lost.
    with open(out_txt_file_path, 'w') as f:
        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue
            epsilon = 0.0003 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True).squeeze()
            if len(approx) < 3:
                continue
            cv2.drawContours(mask_recover, [approx], -1, 255, cv2.FILLED)
            normalized = approx.astype(np.float32) / np.array([w, h])
            points_str = ' '.join([f"{x:.6f} {y:.6f}" for x, y in normalized])
            f.write(f"0 {points_str}\n")  # class 0
    return mask_recover


if __name__ == "__main__":
    data_dir = "./seg_data/masks"
    os.makedirs("./datasets/train/labels", exist_ok=True)
    os.makedirs("./datasets/valid/labels", exist_ok=True)
    os.makedirs("./datasets/train/images", exist_ok=True)
    os.makedirs("./datasets/valid/images", exist_ok=True)

    # glob all .png files
    png_files = glob(os.path.join(data_dir, "*.png"))
    png_files.sort()
    for i, mask_path in tenumerate(png_files):
        if i >= 1600:
            continue
        basename = os.path.basename(mask_path).split(".")[0]
        if i % 10 != 0:
            mask_recover = generate_txt(mask_path, f"./datasets/train/labels/{basename}.txt")
            shutil.copy(f"./seg_data/images/{basename}.jpg", f"./datasets/train/images/{basename}.jpg")
        else:
            mask_recover = generate_txt(mask_path, f"./datasets/valid/labels/{basename}.txt")
            shutil.copy(f"./seg_data/images/{basename}.jpg", f"./datasets/valid/images/{basename}.jpg")
        image = cv2.imread(f"./seg_data/images/{basename}.jpg")
        image[mask_recover < 128] = 0
        cv2.imshow("mask", image)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
