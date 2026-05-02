"""
train.py — 训练入口

流程：
  1. 读取 config.py 配置
  2. 构建 DataLoader（train + val）
  3. 构建模型、损失函数、优化器、学习率调度器
  4. 逐 epoch 训练，每轮结束后在 val 上评估
  5. 按 F1 保存最佳权重到 checkpoints/

运行：
  uv run python train.py
"""

import os
import time
import argparse
import torch
import torch.optim as optim
from datetime import datetime

import config as cfg
from models import build_model
from utils import build_dataloaders, build_loss, ConfusionMeter

# 命令行覆盖
def parse_args():
    parser = argparse.ArgumentParser(description="Change Detection 训练脚本")
    parser.add_argument("--model",       default=cfg.model_name,   help="模型名称")
    parser.add_argument("--dataset",     default=cfg.dataset_name, help="数据集名称")
    parser.add_argument("--epochs",      type=int,   default=cfg.epochs)
    parser.add_argument("--batch_size",  type=int,   default=cfg.batch_size)
    parser.add_argument("--lr",          type=float, default=cfg.lr)
    parser.add_argument("--weight_decay",type=float, default=cfg.weight_decay)
    parser.add_argument("--device",      default=cfg.device)
    parser.add_argument("--resume",      default="",
                        help="从指定 checkpoint 续训，留空则从头训练")
    return parser.parse_args()

# 单epoch训练
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for img_a, img_b, label in loader:
        img_a  = img_a.to(device)
        img_b  = img_b.to(device)
        label  = label.to(device)

        optimizer.zero_grad()
        logits = model(img_a, img_b)      # (B, C, H, W)
        loss   = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


# 验证 
@torch.no_grad()   # 禁用梯度计算
def evaluate(model, loader, criterion, device):
    model.eval()
    meter = ConfusionMeter()
    total_loss = 0.0

    for img_a, img_b, label in loader:
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        label = label.to(device)

        logits = model(img_a, img_b)
        loss   = criterion(logits, label)
        total_loss += loss.item()
        pred   = logits.argmax(dim=1)    # (B, H, W)
        meter.update(pred, label)

    return total_loss / len(loader), meter.compute()


# 主程序入口 
if __name__ == "__main__":

    args = parse_args()

    # 设备
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 目录
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.log_dir,  exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_tag    = f"{args.model}_{timestamp}"
    best_path  = os.path.join(cfg.save_dir, f"best_{exp_tag}.pth")
    latest_path= os.path.join(cfg.save_dir, "latest.pth")
    log_path   = os.path.join(cfg.log_dir,  f"{exp_tag}_train_log.csv")

    cfg.dataset_name = args.dataset
    cfg.model_name   = args.model
    cfg.batch_size   = args.batch_size

    train_loader, val_loader = build_dataloaders(cfg)

    # 模型
    model = build_model(   
        model_name  = cfg.model_name,
        in_channels = cfg.in_channels,
        num_classes = cfg.num_classes,
    ).to(device)
    
    # 损失函数
    criterion = build_loss(
        loss_name     = "ce",              
        class_weights = cfg.class_weights, #类别权重
        device        = str(device),
    )

    # 优化器 调度器
    optimizer = optim.Adam(
        model.parameters(),
        lr           = cfg.lr, #学习率
        weight_decay = cfg.weight_decay, #权重衰减
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size = cfg.lr_step,  #每多少个 epoch 衰减一次
        gamma = cfg.lr_gamma,
    )

    # 断点续训 
    start_epoch = 1
    best_f1     = 0.0
    best_epoch  = 1          

    if args.resume and os.path.isfile(args.resume):
        print(f"[Resume] 载入 checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1     = ckpt.get("best_f1", 0.0)
        best_epoch  = ckpt.get("best_epoch", 1)
        print(f"[Resume] 从 Epoch {start_epoch} 继续，当前最佳 F1={best_f1:.4f}\n")

    # 日志文件
    if start_epoch == 1:                
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,precision,recall,f1,iou,oa,lr\n")

    # 主训练循环
    start_time = datetime.now()
    print(f"\n{'='*70}")
    print(f" 模型：{args.model}  数据集：{args.dataset}  Epochs：{args.epochs}")
    print(f" 开始：{start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        is_best = metrics["f1"] > best_f1
        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"F1={metrics['f1']:.4f}  IoU={metrics['iou']:.4f}  "
            f"lr={current_lr:.2e}  ({elapsed:.1f}s)"
            + ("  ★ best" if is_best else "")
        )

        # 写日志
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                f"{metrics['precision']:.6f},{metrics['recall']:.6f},"
                f"{metrics['f1']:.6f},{metrics['iou']:.6f},{metrics['oa']:.6f},"
                f"{current_lr:.2e}\n"
            )

        # 构造当前 checkpoint payload
        ckpt_payload = {
            "epoch"      : epoch,
            "model_name" : args.model,
            "state_dict" : model.state_dict(),
            "optimizer"  : optimizer.state_dict(),
            "scheduler"  : scheduler.state_dict(),
            "best_f1"    : best_f1,
            "best_epoch" : best_epoch,
        }

        # 保存最佳权重
        if is_best:
            best_f1    = metrics["f1"]
            best_epoch = epoch
            ckpt_payload["best_f1"]    = best_f1
            ckpt_payload["best_epoch"] = best_epoch
            torch.save(ckpt_payload, best_path)

    # 训练结束
    end_time     = datetime.now()
    total_minutes = (end_time - start_time).total_seconds() / 60
    print(f"\n{'='*70}")
    print(f" 训练结束  最佳 F1={best_f1:.4f}  (Epoch {best_epoch})")
    print(f" 权重路径: {best_path}")
    print(f" 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" 总耗时:   {total_minutes:.1f} 分钟")
    print(f"{'='*70}\n")
