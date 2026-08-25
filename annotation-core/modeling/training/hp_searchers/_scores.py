"""评分工具 — 各超参搜索策略共用的模型评分函数。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from modeling._shared import CLINICAL, inv_transform
from modeling.contracts import DataSplitter


def _r2_score(y_true: NDArray, y_pred: NDArray) -> float:
    """计算 R² 决定系数。

    Args:
        y_true: 真实值。
        y_pred: 预测值。

    Returns:
        R² 值，ss_tot=0 时返回 -inf。
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else -np.inf


def _sens_spec_score(y_true: NDArray, y_pred: NDArray) -> float:
    """计算 Sens × Spec 评分。

    在变换空间评分（先 inv_transform 转换回原空间再二值化）。
    """
    y_raw = inv_transform(y_true)
    preds_raw = inv_transform(y_pred)
    tb = (y_raw > CLINICAL).astype(int)
    pb = (preds_raw > CLINICAL).astype(int)
    tn = float(np.sum((tb == 0) & (pb == 0)))
    fp = float(np.sum((tb == 0) & (pb == 1)))
    fn = float(np.sum((tb == 1) & (pb == 0)))
    tp = float(np.sum((tb == 1) & (pb == 1)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens * spec


def _macro_f1_score(y_true: NDArray, y_pred: NDArray) -> float:
    """计算 4 分类 Macro-F1。"""
    from modeling.metrics import compute_4class_metrics
    m = compute_4class_metrics(y_true, y_pred)
    return m["macro_f1"]


def _fold_scores(model: object, X: NDArray, y: NDArray,
                 splitter: DataSplitter, score_metric: str = "r2",
                 weight_components: list | None = None) -> list[float]:
    """在 splitter 的每折上评估模型，返回每折评分列表。

    当 weight_components 提供时，每折用 y_tr 实时计算权重（替代预计算+截取）。

    Args:
        model:      已配置参数（不含 fit）的模型实例。
        X:          特征矩阵。
        y:          目标值（已在变换空间）。
        splitter:   内层切分策略。
        score_metric: "r2", "sens_spec", "macro_f1"。
        weight_components: 权重乘区列表，用于内折实时计算。

    Returns:
        每折评分列表。
    """
    scores: list[float] = []
    for tr_idx, val_idx in splitter.split(y):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        if weight_components:
            w = np.ones(len(y_tr))
            for wc in weight_components:
                w *= wc.compute(y_tr)
            model.external_weight = w
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        if score_metric == "sens_spec":
            scores.append(_sens_spec_score(y_val, preds))
        elif score_metric == "macro_f1":
            scores.append(_macro_f1_score(y_val, preds))
        else:
            scores.append(_r2_score(y_val, preds))
    return scores
