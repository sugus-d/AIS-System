"""评估指标计算。"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, f1_score

from utils.constants import SEVERITY_BINS, SEVERITY_LABELS

CLINICAL = 20.0


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = CLINICAL,
    clinical: float = CLINICAL,
) -> dict:
    """计算回归与分类指标。

    Args:
        y_true:    真实目标值。
        y_pred:    预测目标值。
        threshold: 预测二值化阈值。
        clinical:  真实标签的临床阈值。

    Returns:
        包含 r, rmse, f1, sens, spec, tn, fp, fn, tp 的字典。
    """
    r_val, _ = pearsonr(y_pred, y_true)
    rmse_val = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    y_bin = (y_true > clinical).astype(int)
    pred_bin = (y_pred > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, pred_bin, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

    return {
        "r": float(r_val),
        "rmse": rmse_val,
        "f1": float(f1),
        "sens": float(sens),
        "spec": float(spec),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_4class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: list[float] | None = None,
    labels: list[str] | None = None,
) -> dict:
    """4 分类评估指标（按 severity 分箱）。

    分箱: Normal(0-10°), Mild(10-20°), Moderate(20-40°), Severe(≥40°)。

    Args:
        y_true: 真实 max_cobb 值 (N,)。
        y_pred: 预测 max_cobb 值 (N,)。
        bins:   分箱边界，默认 [0, 10, 20, 40, inf]。
        labels: 类名，默认 Normal/Mild/Moderate/Severe。

    Returns:
        dict: 包含以下键:
            - macro_f1: 4 类宏平均 F1
            - weighted_f1: 加权平均 F1
            - total_accuracy: 总体准确率（对角线和 / 总数）
            - confusion_matrix: 4×4 混淆矩阵 (list of lists)
            - per_class: {类名: {precision, recall, f1, accuracy, support}}
            - rmse: 回归 RMSE
            - r: Pearson 相关系数
    """
    if bins is None:
        bins = SEVERITY_BINS
    if labels is None:
        labels = SEVERITY_LABELS

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    # 分箱
    y_class = np.digitize(y_true, bins[1:-1])  # → 0,1,2,3 (左闭右开)
    p_class = np.digitize(y_pred, bins[1:-1])

    cm = confusion_matrix(y_class, p_class, labels=[0, 1, 2, 3])
    macro_f1 = float(f1_score(y_class, p_class, average="macro"))
    weighted_f1 = float(f1_score(y_class, p_class, average="weighted"))
    total_acc = float(np.sum(y_class == p_class) / n)

    # 回归指标
    r_val = float(pearsonr(y_pred, y_true)[0])
    rmse_val = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    # 每类指标
    per_class: dict[str, dict] = {}
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - cm[i, i])
        fp = int(cm[:, i].sum() - cm[i, i])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_c = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1_c, 4),
            "accuracy": round(accuracy, 4),
            "support": int(tp + fn),
        }

    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "total_accuracy": round(total_acc, 4),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "rmse": round(rmse_val, 4),
        "r": round(r_val, 4),
    }
