"""
变化检测评价指标（针对二分类：0=不变，1=变化）。所有指标均以「变化类（正类）」为基准计算。
"""

from dataclasses import dataclass, field
from typing import Dict
import torch

#  混淆矩阵累加器

@dataclass
class ConfusionMeter:
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, pred, label):

        pred  = pred.view(-1)  # 把 (B, H, W) 展平成一维
        label = label.view(-1)

        self.tp += int(((pred == 1) & (label == 1)).sum())
        self.fp += int(((pred == 1) & (label == 0)).sum())
        self.fn += int(((pred == 0) & (label == 1)).sum())
        self.tn += int(((pred == 0) & (label == 0)).sum())

    def compute(self) :
        """返回各指标字典，均以变化类为正例"""
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-8

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * tp / (2 * tp + fp + fn + eps)
        iou       = tp / (tp + fp + fn + eps)
        oa        = (tp + tn) / (tp + fp + fn + tn + eps)   

        return {
            "precision" : round(precision, 6),
            "recall"    : round(recall,    6),
            "f1"        : round(f1,        6),
            "iou"       : round(iou,       6),
            "oa"        : round(oa,        6),
        }

    def summary_str(self) :
        m = self.compute()
        return (
            f"Precision={m['precision']:.4f}  "
            f"Recall={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}  "
            f"IoU={m['iou']:.4f}  "
            f"OA={m['oa']:.4f}"
        )


#  单次计算（不需要累加时的便捷函数）
def compute_metrics(pred,label) :

    meter = ConfusionMeter()
    meter.update(pred, label)
    return meter.compute()