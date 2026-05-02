#CLCD 数据集读取、裁剪、增强


import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


# 图像归一化
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize(img):
    return (img - mean) / std


#  数据集类
class CLCDDataset(Dataset):

    # 拼接完整路径、构建三个子目录、读取 time1/ 下的所有文件名、检查文件是否存在

    def __init__(self, root, split = "train", patch_size = 256, augment = True):
        super().__init__()
        self.root = Path(root) / split   #./data/CLCD/train
        self.split = split
        self.patch_size = patch_size
        self.use_augment = augment and (split == "train")

        # 三个子目录
        self.dir_time1 = self.root / "time1"
        self.dir_time2 = self.root / "time2"
        self.dir_label = self.root / "label"

        # 文件名列表（以 time1/ 为基准，三个文件名一致）
        self.names = sorted(os.listdir(self.dir_time1))

        if len(self.names) == 0:
            raise RuntimeError(f"未在 {self.dir_time1} 下找到任何图像文件，请检查路径")

    def __len__(self) :
        return len(self.names)
    
    # PIL打开图像转为RGB（3通道）
    def load_image(self,path):
        return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    def load_label(self,path):
        lbl = np.array(Image.open(path).convert("L"), dtype=np.int64)
        return (lbl > 128).astype(np.int64)  # 二值化：大于128的变成1，否则0

    def random_crop(self,img_a,img_b,label):

        #三张图同步随机裁剪
        h, w = img_a.shape[:2]   # (高度, 宽度, 通道数) 这里取前两个
        ps = self.patch_size
        top  = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return (
            img_a[top:top+ps, left:left+ps],
            img_b[top:top+ps, left:left+ps],
            label[top:top+ps, left:left+ps],
        )


    def apply_augment(self,img_a,img_b,label) :

        # 水平翻转
        if random.random() > 0.5:
            img_a = np.fliplr(img_a).copy()
            img_b = np.fliplr(img_b).copy()
            label = np.fliplr(label).copy()
        # 垂直翻转
        if random.random() > 0.5:
            img_a = np.flipud(img_a).copy()
            img_b = np.flipud(img_b).copy()
            label = np.flipud(label).copy()
        # 随机旋转 90°/180°/270°
        k = random.randint(0, 3)
        if k > 0:
            img_a = np.rot90(img_a, k).copy()
            img_b = np.rot90(img_b, k).copy()
            label = np.rot90(label, k).copy()
        return img_a, img_b, label

    #返回一个样本 获取文件名-加载三个图-预处理（裁剪、增强）-标准化2个图-转成Tensor并返回
    def __getitem__(self, idx):
        name = self.names[idx]

        img_a = self.load_image(self.dir_time1 / name)
        img_b = self.load_image(self.dir_time2 / name)
        label = self.load_label(self.dir_label / name)

        # 训练：随机裁剪 + 增强
        if self.use_augment:
            img_a, img_b, label = self.random_crop(img_a, img_b, label)
            img_a, img_b, label = self.apply_augment(img_a, img_b, label)

        # 标准化
        img_a = normalize(img_a)
        img_b = normalize(img_b)

        # HWC → CHW，转 Tensor
        img_a = torch.from_numpy(img_a.transpose(2, 0, 1))   # (3, H, W)
        img_b = torch.from_numpy(img_b.transpose(2, 0, 1))   # (3, H, W)
        label = torch.from_numpy(label)                      # (H, W)  int64

        return img_a, img_b, label


#  DataLoader    根据 config 返回 (train_loader, val_loader) 
  
# 创建实例
def build_dataloaders(cfg):

    train_ds = CLCDDataset(
        root       = cfg.data_root,
        split      = "train",
        patch_size = cfg.patch_size,
        augment    = True,
    )
    val_ds = CLCDDataset(
        root       = cfg.data_root,
        split      = "val",
        patch_size = cfg.patch_size,
        augment    = False,
    )
    
# 把 Dataset 包装成 DataLoader 
    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.batch_size,
        shuffle     = True,  # 打乱顺序
        num_workers = cfg.num_workers,
        pin_memory  = True,
        drop_last   = True,    # 丢弃最后一个不完整的patch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = 4,
        shuffle     = False,  # 不打乱顺序
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )

    print(f"[Dataset] train={len(train_ds)}张, val={len(val_ds)}张")
    return train_loader, val_loader
