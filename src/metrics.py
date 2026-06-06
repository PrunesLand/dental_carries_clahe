"""
Segmentation metrics computed from raw confusion-matrix counts.
Accumulating TP/FP/FN/TN across batches before computing ratios avoids
the biased micro-average that batch-level averaging produces.
"""

import numpy as np
import torch
import torch.nn as nn


def confusion_counts(
    pred: np.ndarray, target: np.ndarray
) -> tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) for binary predictions at threshold 0.5."""
    p = (pred.flatten() > 0.5).astype(np.int64)
    t = (target.flatten() > 0.5).astype(np.int64)
    TP = int((p * t).sum())
    FP = int((p * (1 - t)).sum())
    FN = int(((1 - p) * t).sum())
    TN = int(((1 - p) * (1 - t)).sum())
    return TP, FP, FN, TN


def metrics_from_counts(
    TP: int, FP: int, FN: int, TN: int, smooth: float = 1e-6
) -> dict[str, float]:
    """Compute DSC, IoU, recall, precision, F1, and accuracy from counts."""
    precision = TP / (TP + FP + smooth)
    recall = TP / (TP + FN + smooth)
    return {
        "dsc":       (2 * TP) / (2 * TP + FP + FN + smooth),
        "iou":       TP / (TP + FP + FN + smooth),
        "recall":    recall,
        "precision": precision,
        "f1":        (2 * precision * recall) / (precision + recall + smooth),
        "accuracy":  (TP + TN) / (TP + TN + FP + FN + smooth),
    }


class DiceLoss(nn.Module):
    """Standard soft Dice loss (kept for reference; experiment uses TverskyLoss)."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
