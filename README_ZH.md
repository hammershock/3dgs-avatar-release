本项目采用的smplx模型，包括以下关键的key：
minimal_shape (6890, 3)
betas (1, 10)
Jtr_posed (24, 3)  # 骨骼关节位置
bone_transforms (24, 4, 4)  # 
trans (3,)  # 全局平移
root_orient (3,)  # 全局旋转
pose_body (63,)  # (21, 3)
pose_hand (6,)  # (2, 3)