"""
eval.py — 测试 / 评估入口

功能：
  1. 加载指定 checkpoint（或自动找最新 best_*.pth）
  2. 在 test / val 集上推理，计算 Precision / Recall / F1 / IoU / OA
  3. 可选：将预测掩膜保存为 PNG 到 --pred_dir

运行示例：
  uv run python eval.py
  uv run python eval.py --split val
  uv run python eval.py --checkpoint checkpoints/best_ef_unet_20250101_120000.pth --save_pred
  uv run python eval.py --split val --save_pred --pred_dir outputs/val_pred
"""

import os
import glob
import argparse
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

import config as cfg
from models import build_model
from utils import build_dataset, ConfusionMeter 


# 命令行参数 
def parse_args():
    parser = argparse.ArgumentParser(description="Change Detection 评估脚本")
    parser.add_argument("--split",      default=cfg.eval_split, help="评估集：val 或 test")
    parser.add_argument("--checkpoint", default=cfg.checkpoint, help="权重路径；留空则自动找 checkpoints/ 下最新的 best_*.pth")
    parser.add_argument("--model",      default=cfg.model_name, help="模型名称（与 checkpoint 一致）")
    parser.add_argument("--dataset",    default=cfg.dataset_name, help="数据集名称")
    parser.add_argument("--save_pred",  action="store_true", help="是否保存预测掩膜（白=变化，黑=不变）")
    parser.add_argument("--pred_dir",   default="outputs/pred", help="预测图输出目录")
    return parser.parse_args()

# 自动查找最新 checkpoint 
def find_latest_checkpoint(save_dir):
    pattern = os.path.join(save_dir, "best_*.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(
            f"在 {save_dir}/ 下未找到任何 best_*.pth，"
            "请用 --checkpoint 手动指定路径。"
        )
    # 按文件修改时间取最新
    return max(candidates, key=os.path.getmtime)


# 主程序
if __name__ == "__main__":
    args = parse_args()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 确定 checkpoint 路径
    ckpt_path = args.checkpoint or find_latest_checkpoint(cfg.save_dir)
    print(f"[Device]  {device}")
    print(f"[Split]   {args.split}")
    print(f"[Ckpt]    {ckpt_path}\n")

    # 数据集
    cfg.dataset_name = args.dataset
    dataset = build_dataset(
        cfg     = cfg,
        split   = args.split,
        augment = False,         # 评估时不做增强
    )
    loader = DataLoader(
        dataset,
        batch_size  = 1,         # 逐张推理，方便保存与原图对应
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )
    print(f"[Dataset] {args.dataset} · {args.split} 集共 {len(dataset)} 张\n")

    # 模型 
    model = build_model(
        model_name  = args.model,
        in_channels = cfg.in_channels,
        num_classes = cfg.num_classes,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = (
        ckpt["state_dict"]
        if isinstance(ckpt, dict) and "state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state_dict)
    model.eval()

    saved_epoch = ckpt.get("epoch",    "?") if isinstance(ckpt, dict) else "?"
    saved_f1    = ckpt.get("best_f1", "?") if isinstance(ckpt, dict) else "?"
    print(f"[Ckpt]    来自 Epoch {saved_epoch}，训练时 Val F1 = {saved_f1}\n")

    # 预测图保存目录
    if args.save_pred:
        os.makedirs(args.pred_dir, exist_ok=True)
        print(f"[SavePred] 预测掩膜将保存到 {args.pred_dir}/\n")

    # 推理 + 评估
    meter = ConfusionMeter()

    with torch.no_grad():
        for idx, (img_a, img_b, label) in enumerate(loader):
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            label = label.to(device)

            logits = model(img_a, img_b)       # (1, C, H, W)
            pred   = logits.argmax(dim=1)      # (1, H, W)
            meter.update(pred, label)

            # 可选：保存预测掩膜
            if args.save_pred:
                pred_np = pred[0].cpu().numpy().astype(np.uint8) * 255
                if hasattr(dataset, "names"):
                    fname = dataset.names[idx]
                else:
                    fname = f"{idx:05d}.png"
                Image.fromarray(pred_np).save(os.path.join(args.pred_dir, fname)
)

    # 输出
    metrics = meter.compute()
    sep = "─" * 44
    print(f"\n{sep}")
    print(f"  评估集：{args.split.upper()} | 模型：{args.model} | 数据集：{args.dataset}")
    print(sep)
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  IoU       : {metrics['iou']:.4f}")
    print(f"  OA        : {metrics['oa']:.4f}")
    print(sep)
