# utils/dataset.py
# CLCD 数据集读取、裁剪、增强

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


# 图像归一化
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize(img):
    return (img - mean) / std


#  基类
class ChangeDetectionDataset(Dataset):
    names: list

# CLCD 数据集
class CLCDDataset(ChangeDetectionDataset):

    def __init__(self, root, split="train", patch_size=256, augment=True):
        super().__init__()
        self.root        = Path(root) / split
        self.split       = split
        self.patch_size  = patch_size
        self.use_augment = augment and (split == "train")

        self.dir_time1 = self.root / "time1"
        self.dir_time2 = self.root / "time2"
        self.dir_label = self.root / "label"

        self.names = sorted(os.listdir(self.dir_time1))

    def __len__(self):
        return len(self.names)

    def load_image(self, path):
        return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    def load_label(self, path):
        lbl = np.array(Image.open(path).convert("L"), dtype=np.int64)
        return (lbl > 128).astype(np.int64)

    def random_crop(self, img_a, img_b, label):
        h, w = img_a.shape[:2]
        ps   = self.patch_size
        top  = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return (
            img_a[top:top+ps, left:left+ps],
            img_b[top:top+ps, left:left+ps],
            label[top:top+ps, left:left+ps],
        )

    def apply_augment(self, img_a, img_b, label):
        if random.random() > 0.5:
            img_a = np.fliplr(img_a).copy()
            img_b = np.fliplr(img_b).copy()
            label = np.fliplr(label).copy()
        if random.random() > 0.5:
            img_a = np.flipud(img_a).copy()
            img_b = np.flipud(img_b).copy()
            label = np.flipud(label).copy()
        k = random.randint(0, 3)
        if k > 0:
            img_a = np.rot90(img_a, k).copy()
            img_b = np.rot90(img_b, k).copy()
            label = np.rot90(label, k).copy()
        return img_a, img_b, label

    def __getitem__(self, idx):
        name  = self.names[idx]
        img_a = self.load_image(self.dir_time1 / name)
        img_b = self.load_image(self.dir_time2 / name)
        label = self.load_label(self.dir_label / name)

        if self.use_augment:
            img_a, img_b, label = self.random_crop(img_a, img_b, label)
            img_a, img_b, label = self.apply_augment(img_a, img_b, label)

        img_a = normalize(img_a)
        img_b = normalize(img_b)

        # HWC → CHW
        img_a = torch.from_numpy(img_a.transpose(2, 0, 1))  # (3, H, W)
        img_b = torch.from_numpy(img_b.transpose(2, 0, 1))  # (3, H, W)
        label = torch.from_numpy(label)                     # (H, W) int64

        return img_a, img_b, label


# 数据集注册表
DATASET_REGISTRY: dict = {
    "clcd": CLCDDataset,
    # "levir": LevirCDDataset,
}


def build_dataset(cfg, split: str, augment: bool) -> ChangeDetectionDataset:
    name = cfg.dataset_name.lower()

    return DATASET_REGISTRY[name](
        root       = cfg.data_root,
        split      = split,
        patch_size = cfg.patch_size,
        augment    = augment,
    )


def build_dataloaders(cfg):
    train_ds = build_dataset(cfg, split="train", augment=True)
    val_ds   = build_dataset(cfg, split="val",   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = 4,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )

    print(f"[Dataset] train={len(train_ds)}张, val={len(val_ds)}张")
    return train_loader, val_loader
