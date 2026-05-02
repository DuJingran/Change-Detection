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
import torch
import torch.optim as optim
from datetime import datetime

import config as cfg
from models import build_model
from utils import build_dataloaders, build_loss, ConfusionMeter


# 单 epoch 训练
def train_one_epoch(epoch):
    model.train() 
    total_loss = 0.0
    n_batches  = len(train_loader)

    for batch_idx, (img_a, img_b, label) in enumerate(train_loader, start=1):
        img_a = img_a.to(device)
        img_b = img_b.to(device)
        label = label.to(device)

        optimizer.zero_grad() #梯度清零
        logits = model(img_a, img_b)     # (B, C, H, W) logits 输出每个像素的类别分数
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / n_batches


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

    # 设备
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 目录
    os.makedirs(cfg.save_dir, exist_ok=True)  #checkpoints
    os.makedirs(cfg.log_dir,  exist_ok=True)  #logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(cfg.save_dir, f"{cfg.model_name}_{timestamp}.pth")
    log_path = os.path.join(cfg.log_dir, f"{cfg.model_name}_{timestamp}_train_log.csv")

    train_loader, val_loader = build_dataloaders(cfg)

    # 构建模型
    model = build_model(   
        model_name  = cfg.model_name,
        in_channels = cfg.in_channels,
        num_classes = cfg.num_classes,
    ).to(device)
    
    # 构建损失函数
    criterion = build_loss(
        loss_name     = "ce",              
        class_weights = cfg.class_weights, #类别权重
        device        = str(device),
    )

    # 构建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr           = cfg.lr, #学习率
        weight_decay = cfg.weight_decay, #权重衰减
    )

    # 构建学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size = cfg.lr_step,  #每多少个 epoch 衰减一次
        gamma = cfg.lr_gamma,
    )

    # 日志文件
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,precision,recall,f1,iou,oa,lr\n")

    # 主训练循环
    best_f1 = 0.0

    start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f" 模型：{cfg.model_name}   Epoch：{cfg.epochs}   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    for epoch in range(1, cfg.epochs + 1):

        t0 = time.time()
        train_loss = train_one_epoch(epoch)   # 训练
        val_loss, metrics = evaluate(model, val_loader, criterion, device)  # 验证

        scheduler.step()                    # 更新学习率
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        print(
            f"Epoch [{epoch:03d}/{cfg.epochs}]  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"IoU={metrics['iou']:.4f}  "
            f"lr={current_lr:.2e}  "
            f"({elapsed:.1f}s)"
            f"{' ★ best' if metrics['f1'] > best_f1 else ''}"
        )

        # 保存日志
        with open(log_path, "a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                f"{metrics['precision']:.6f},{metrics['recall']:.6f},"
                f"{metrics['f1']:.6f},{metrics['iou']:.6f},{metrics['oa']:.6f},"
                f"{current_lr:.2e}\n"
            )

        # 保存最佳权重
        best_marker = ""
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch"     : epoch,
                    "model_name": cfg.model_name,
                    "state_dict": model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "best_f1"   : best_f1,
                },
                ckpt_path,
            )

    end_time = datetime.now()
    total_time = end_time - start_time
    total_minutes = total_time.total_seconds() / 60

    print(f"  训练结束  最佳 F1={best_f1:.4f} (Epoch {best_epoch})")
    print(f"  结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  总耗时: {total_minutes:.1f} 分钟")

