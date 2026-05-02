"""
eval.py — 测试 / 评估入口

功能：
  1. 加载 checkpoints/best_model.pth
  2. 在 test（或 val）集上跑推理
  3. 打印 Precision / Recall / F1 / IoU / OA
  4. 可选：将预测掩膜保存为 PNG 到 outputs/pred/

运行：
  uv run python eval.py
  uv run python eval.py --split val --checkpoint checkpoints/best_model.pth --save_pred --pred_dir outputs/val_pred  
  # 选择一个checkpoint , 在 val 集评估，预测图保存到指定目录
  

"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

import config as cfg
from models import build_model
from utils  import CLCDDataset, ConfusionMeter

#  命令行参数

parser = argparse.ArgumentParser()
parser.add_argument("--split",      default=cfg.eval_split, help="val 或 test")
parser.add_argument("--checkpoint", default=cfg.checkpoint,  help="权重路径")
parser.add_argument("--save_pred",  action="store_true",      help="是否保存预测掩膜")
parser.add_argument("--pred_dir",   default="outputs/pred",   help="预测图输出目录")
args = parser.parse_args()

device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
print(f"[Device] {device}")
print(f"[Split]  {args.split}")
print(f"[Ckpt]   {args.checkpoint}\n")


#  数据集
dataset = CLCDDataset(
    root       = cfg.data_root,
    split      = args.split,
    patch_size = cfg.patch_size,
    augment    = False,           # 评估时不做增强
)
loader = DataLoader(
    dataset,
    batch_size  = 1,              # 逐张推理，方便保存原图对应结果
    shuffle     = False,
    num_workers = cfg.num_workers,
    pin_memory  = True,
)
print(f"[Dataset] {args.split} 集共 {len(dataset)} 张\n")


#  模型加载
model = build_model(
    model_name  = cfg.model_name,
    in_channels = cfg.in_channels,
    num_classes = cfg.num_classes,
).to(device)

ckpt = torch.load(args.checkpoint, map_location=device)

# 兼容两种保存格式：纯 state_dict 或 dict（含 epoch 等信息）
state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

saved_epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
saved_f1    = ckpt.get("best_f1", "?") if isinstance(ckpt, dict) else "?"
print(f"[Ckpt] 来自 Epoch {saved_epoch}，训练时 Val F1 = {saved_f1}\n")


#  保存目录
if args.save_pred:
    os.makedirs(args.pred_dir, exist_ok=True)
    print(f"[SavePred] 预测掩膜将保存到 {args.pred_dir}/\n")


#  推理 + 评估
meter = ConfusionMeter()

with torch.no_grad():
    for idx, (img_a, img_b, label) in enumerate(loader):
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        label = label.to(device)

        logits = model(img_a, img_b)        # (1, C, H, W)
        pred   = logits.argmax(dim=1)       # (1, H, W)

        meter.update(pred, label)

        # ── 可选：保存预测掩膜（白=变化，黑=不变）──
        if args.save_pred:
            pred_np  = pred[0].cpu().numpy().astype(np.uint8) * 255
            fname    = dataset.names[idx]
            save_path = os.path.join(args.pred_dir, fname)
            Image.fromarray(pred_np).save(save_path)



#  输出最终指标
metrics = meter.compute()
sep = "─" * 42
print(f"\n{sep}")
print(f"  评估集：{args.split.upper()}  |  模型：{cfg.model_name}")
print(sep)
print(f"  Precision : {metrics['precision']:.4f}")
print(f"  Recall    : {metrics['recall']:.4f}")
print(f"  F1        : {metrics['f1']:.4f}")
print(f"  IoU       : {metrics['iou']:.4f}")
print(f"  OA        : {metrics['oa']:.4f}")
print(sep)

