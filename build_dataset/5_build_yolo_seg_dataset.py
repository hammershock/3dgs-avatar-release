"""build YOLO seg dataset"""
import os
import shutil
from glob import glob
from tqdm import tqdm

def copy_images(source_dir, dest_dir):
    """copy images from source_dir to dest_dir"""
    # glob all .jpg files
    jpgs = glob(os.path.join(source_dir, "*.jpg"))
    pngs = glob(os.path.join(source_dir, "*.png"))
    jpgs.sort()
    pngs.sort()
    os.makedirs(dest_dir, exist_ok=True)
    assert len(jpgs) == len(pngs)
    jpg_dir = os.path.join(dest_dir, "images")
    png_dir = os.path.join(dest_dir, "masks")
    os.makedirs(jpg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    for jpg, png in tqdm(zip(jpgs, pngs)):
        basename = os.path.basename(png).split(".")[0]
        jpg_path = os.path.join(jpg_dir, basename + ".jpg")
        png_path = os.path.join(png_dir, basename + ".png")
        shutil.copy(jpg, jpg_path)
        shutil.copy(png, png_path)

if __name__ == "__main__":
    source_dir = "../data/ZJUMoCap/CoreView_001/1"
    dest_dir = "./seg_data"
    copy_images(source_dir, dest_dir)
