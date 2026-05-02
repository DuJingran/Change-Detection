"""
损失函数。
默认使用加权交叉熵（WeightedCE）应对正负样本不平衡。
可按需启用 DiceLoss 或组合损失。
loss_name 可选：
    "ce"       → WeightedCrossEntropyLoss
    "dice"     → DiceLoss（不含 CE，通常不单独用）
    "combined" → CombinedLoss（CE + Dice）

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


#  加权交叉熵

class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, class_weights=(1.0, 5.0), device="cpu"):
        super().__init__()
        w = torch.tensor(class_weights, dtype=torch.float32, device=device)  
        self.ce = nn.CrossEntropyLoss(weight=w)

    def forward(self, logits, labels):
        return self.ce(logits, labels)

#  Dice Loss（对小目标友好）

class DiceLoss(nn.Module):

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, labels) :
        prob = F.softmax(logits, dim=1)[:, 1]   # (B, H, W) 变化类概率
        tgt  = (labels == 1).float()             # (B, H, W) 二值 ground truth

        prob_flat = prob.reshape(-1)
        tgt_flat  = tgt.reshape(-1)

        intersection = (prob_flat * tgt_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            prob_flat.sum() + tgt_flat.sum() + self.smooth
        )
        return 1.0 - dice

#  组合损失：CE + Dice
 # total_loss = alpha * CE + (1-alpha) * Dice

class CombinedLoss(nn.Module):

    def __init__(
        self,
        alpha = 0.5,
        class_weights=(1.0, 5.0),
        device="cpu",
    ):
        super().__init__()
        self.alpha = alpha
        self.ce    = WeightedCrossEntropyLoss(class_weights, device)
        self.dice  = DiceLoss()

    def forward(self, logits, labels):
        return self.alpha * self.ce(logits, labels) + (1 - self.alpha) * self.dice(logits, labels)


#  损失函数工厂
def build_loss(loss_name = "ce", class_weights=(1.0, 5.0), device="cpu") :

    name = loss_name.lower()
    if name == "ce":
        return WeightedCrossEntropyLoss(class_weights, device)
    elif name == "dice":
        return DiceLoss()
    elif name == "combined":
        return CombinedLoss(class_weights=class_weights, device=device)
    else:
        raise ValueError(f"未知损失函数 '{loss_name}'，可选：ce / dice / combined")