import torch
torch.cuda.empty_cache()

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("./yolov8x-seg.pt")
    model.train(
        data="./dance.yaml",  # 数据集配置 文件路径
        epochs=10,  # 训练轮数
        batch=4,
        imgsz=1088,
        patience=50,
        save=True,  # 保存训练结果
        device=0,  # 使用GPU训练（如有）
    )

