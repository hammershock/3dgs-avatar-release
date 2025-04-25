视频输入
1. 视频预处理（剪辑，抽帧）2min以内，不要太长
2. YOLO分割mask，获取mask包
3. CLIFF获取人体模型序列
4. 仿照ZJU-MoCap格式构建数据集
    - mask，图像
    - 解析CLIFF输出结果，生成模型文件
    - 生成相机参数文件
