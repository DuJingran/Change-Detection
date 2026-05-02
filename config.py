# config.py — 所有超参数与路径配置

import os

#  路径配置
data_root   = os.path.join("data", "CLCD")    # 数据集根目录
save_dir    = "checkpoints"                   # 权重保存目录
log_dir     = "logs"                          # 日志目录

#  数据配置
patch_size  = 256       # 训练时随机裁剪大小
in_channels = 3         # 输入波段数
num_classes = 2         # 0=不变，1=变化

# 模型配置 可选 "siamunet_conc", "siamunet_diff", "ef_unet"
model_name = "ef_unet"

# 训练配置
epochs = 50                 # 训练轮数
batch_size = 8              # 每轮训练的样本数
num_workers = 4             
lr = 1e-3                   # 学习率
lr_step = 15                # 每隔多少轮学习率衰减一次
lr_gamma = 0.5              # 学习率衰减系数
weight_decay = 1e-4         # L2正则化系数
class_weights = [1.0, 5.0]  # 类别权重[背景权重，变化区域权重]
device = "cuda"             # 训练设备 "cuda"、"cpu"

# 评估配置
eval_split = "test"

