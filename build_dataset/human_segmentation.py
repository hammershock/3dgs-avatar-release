from motion_display import VideoStream
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


def visualize_instance_segmentation(image, results):
    # 复制原始图像
    vis_image = image.copy()

    # 获取所有检测结果  # only one input image
    boxes = results[0].boxes.xyxy.cpu().numpy()
    masks = results[0].masks.xy
    class_ids = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    # 生成随机颜色
    colors = np.random.randint(0, 255, size=(len(boxes), 3))

    # 绘制每个实例
    for idx, (box, mask, class_id, conf) in enumerate(zip(boxes, masks, class_ids, confidences)):
        if class_id == 0:  # 0代表person类
            # 绘制边界框
            color = tuple(map(int, colors[idx]))
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # 绘制分割掩码
            mask_points = mask.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(vis_image, [mask_points], color)

            # 添加标签
            label = f"Person {conf:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return vis_image


if __name__ == '__main__':
    with VideoStream("../videos/dance.mp4") as video:
        frame = video[530]

    # 加载预训练的YOLOv8分割模型
    model = YOLO('yolov8x-seg.pt')  # 使用nano版本，可替换为yolov8x-seg获得更高精度

    # 使用OpenCV读取图像
    # image = cv2.imread('test_image.jpg')  # 替换为你的图片路径
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 进行实例分割推理
    results = model.predict(image_rgb, classes=[0], conf=0.5)  # 只检测人物，置信度阈值0.5
    mask = (results[0].masks.data.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    print(mask.shape)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)

    # 可视化结果
    segmented_image = visualize_instance_segmentation(image_rgb, results)

    # 使用matplotlib显示结果
    plt.figure(figsize=(12, 8))
    plt.imshow(segmented_image)
    plt.axis('off')
    plt.show()

