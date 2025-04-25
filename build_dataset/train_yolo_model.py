from ultralytics import YOLO

if __name__ == "__main__":
    data_dir = "../data/ZJUMoCap/CoreView_001/1"
    model_path = "./yolov8x-seg.pt"
    model_save_path = "./yolov8x-seg-finetuned.pt"
    batch_size = 8

    # 加载预训练模型
    model = YOLO(model_path)

    # 训练模型
    model.train(
        data="dance.yaml",  # 数据集配置文件路径
        epochs=100,  # 训练轮数
        batch=batch_size,  # 批次大小
        imgsz=640,  # 输入图像尺寸
        patience=50,  # 早停耐心值
        project="custom_train",  # 项目名称
        name="seg_exp",  # 实验名称
        save=True,  # 保存训练结果
        val=True,  # 启用验证
        device=0,  # 使用GPU训练（如有）
    )

    # 保存微调后的模型
    model.save(model_save_path)