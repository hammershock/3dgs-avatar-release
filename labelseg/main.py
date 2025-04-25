import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import json

class ImageMaskingApp:
    def __init__(self, root, data_path, config_file="config.json"):
        self.root = root
        self.root.title("图像掩膜应用")

        # 获取所有图像路径和掩膜路径
        self.image_paths = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jpg')])
        self.mask_paths = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.png')])

        # 尝试从配置文件加载当前索引
        self.config_file = config_file
        self.current_index = self.load_config()  # 加载上次编辑的图像索引，默认为0
        self.image = Image.open(self.image_paths[self.current_index])
        self.img_array = np.array(self.image)
        self.mask = cv2.imread(self.mask_paths[self.current_index], cv2.IMREAD_GRAYSCALE) == 255  # mask为True表示有效区域

        # 转换为Tkinter可显示的图像
        self.tk_image = ImageTk.PhotoImage(self.image)

        # 创建Canvas来显示图像
        self.canvas = tk.Canvas(root, width=self.image.width, height=self.image.height)
        self.canvas.pack()

        # 按钮区域
        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # previous按钮
        self.previous_button = tk.Button(button_frame, text="Previous", command=self.show_previous_image)
        self.previous_button.pack(side=tk.LEFT, padx=5)

        # next按钮
        self.next_button = tk.Button(button_frame, text="Next", command=self.show_next_image)
        self.next_button.pack(side=tk.LEFT, padx=5)

        # save按钮
        self.save_button = tk.Button(button_frame, text="Save", command=self.save_mask)
        self.save_button.pack(side=tk.LEFT, padx=5)

        # 进度显示
        self.progress_label = tk.Label(root, text=f"Image {self.current_index + 1}/{len(self.image_paths)}")
        self.progress_label.pack(side=tk.BOTTOM, fill=tk.X)

        self.mode = "+"

        # 显示图像
        self.show_masked_image()

        # 鼠标事件绑定
        self.start_x = None
        self.start_y = None
        self.current_coords = []
        self.canvas.bind("<Button-1>", self.on_left_press)  # 左键按下
        self.canvas.bind("<Button-3>", self.on_right_press)  # 右键按下

        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<B3-Motion>", self.on_mouse_drag)

        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<ButtonRelease-3>", self.on_mouse_release)

        # 快捷键绑定
        self.root.bind("<a>", self.show_previous_image)
        self.root.bind("<d>", self.show_next_image)
        self.root.bind("<space>", self.save_mask)

    def on_left_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.current_coords = [(self.start_x, self.start_y)]
        self.mode = "+"

    def on_mouse_drag(self, event):
        self.current_coords.append((event.x, event.y))

    def on_mouse_release(self, event):
        self.current_coords.append((self.start_x, self.start_y))  # 闭合路径
        self.update_mask(self.mode)  # 更新掩膜
        self.show_masked_image()

    def on_right_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.current_coords = [(self.start_x, self.start_y)]
        self.mode = "-"  # change mode

    def update_mask(self, mode):
        coords = np.array(self.current_coords, dtype=np.int32)
        temp_mask = np.zeros(self.img_array.shape[:2], dtype=np.uint8)

        # 填充多边形区域
        cv2.fillPoly(temp_mask, [coords], 255)

        # 更新现有mask
        if mode == "+":
            self.mask[temp_mask == 255] = True
        elif mode == "-":
            self.mask[temp_mask == 255] = False

    def show_masked_image(self):
        # 创建一个图像副本，修改其边缘区域
        image_show = self.img_array.copy()

        # 提取掩膜的边缘
        edges = cv2.dilate(self.mask.astype(np.uint8), None) - self.mask.astype(np.uint8)  # 扩张后减去原掩膜，得到边缘

        # 将边缘部分设置为红色
        image_show[edges == 1] = [255, 0, 0]  # 红色边缘 [255, 0, 0]

        # 更新显示的图像
        masked_image = Image.fromarray(image_show)
        self.tk_image = ImageTk.PhotoImage(masked_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # 更新进度显示
        self.progress_label.config(text=f"Image {self.current_index + 1}/{len(self.image_paths)}")


    def show_previous_image(self, event=None):
        if self.current_index > 0:
            self.save_mask()
            self.current_index -= 1
            self.load_image_and_mask()

    def show_next_image(self, event=None):
        if self.current_index < len(self.image_paths) - 1:
            self.save_mask()
            self.current_index += 1
            self.load_image_and_mask()


    def load_image_and_mask(self):
        self.image = Image.open(self.image_paths[self.current_index])
        self.img_array = np.array(self.image)
        self.mask = cv2.imread(self.mask_paths[self.current_index], cv2.IMREAD_GRAYSCALE) == 255  # mask为True表示有效区域

        # 转换为Tkinter可显示的图像
        self.tk_image = ImageTk.PhotoImage(self.image)

        # 显示图像
        self.show_masked_image()

    def save_mask(self, event=None):
        mask_to_save = np.uint8(self.mask) * 255  # 将布尔值mask转换为0-255的范围
        cv2.imwrite(self.mask_paths[self.current_index], mask_to_save)  # 保存掩膜到文件
        print(f"Mask saved to {self.mask_paths[self.current_index]}")
        self.save_config()  # 保存当前编辑的索引

    def save_config(self):
        # 将当前图像的索引保存到配置文件中
        with open(self.config_file, "w") as f:
            json.dump({"current_index": self.current_index}, f)

    def load_config(self):
        # 尝试从配置文件加载当前索引
        if os.path.exists(self.config_file):
            with open(self.config_file, "r") as f:
                config = json.load(f)
                return config.get("current_index", 0)  # 默认返回索引0
        return 0  # 默认返回索引0

if __name__ == "__main__":
    root = tk.Tk()
    data_path = "../data/ZJUMoCap/CoreView_001/1"  # 替换为实际数据路径
    app = ImageMaskingApp(root, data_path)
    root.mainloop()
