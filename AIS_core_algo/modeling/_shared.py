"""跨模块共享常量与工具函数 — 避免 modeling.cv 与其他模块循环依赖。"""

from __future__ import annotations

import numpy as np

CLINICAL = 20.0
PIECEWISE_THRESHOLD = 48.0


def transform_target(y: np.ndarray, th: float = PIECEWISE_THRESHOLD) -> np.ndarray:
    """对严重 Cobb 角做对数压缩（piecewise log transform）。"""
    yt = y.copy()
    m = yt > th
    yt[m] = th + np.log(yt[m] - th + 1)
    return yt


def inv_transform(yt: np.ndarray, th: float = PIECEWISE_THRESHOLD) -> np.ndarray:
    """对数压缩的逆变换。"""
    y = yt.copy()
    m = y > th
    y[m] = th + np.exp(y[m] - th) - 1
    return np.maximum(y, 0)


def apply_calibration(
    preds: np.ndarray,
    bias: dict[int | str, float],
    boundaries: np.ndarray | None = None,
    ranges: dict[int, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Per-class 偏差校正应用段：分箱 → 减 bias → 钳制到类范围。

    训练侧（trainer/trainer_margin 拟合 bias 后应用）与预测侧（feature_pipeline
    用模型包 bias 应用）共用同一实现；bias 键 int（原样保存）与 str（JSON 序列化）
    都兼容。boundaries 缺省 cobb 分级边界，ranges 缺省 CLASS_RANGES。
    """
    from utils.constants import CLASS_RANGES, SEVERITY_BINS

    bin_edges = boundaries if boundaries is not None else np.asarray(SEVERITY_BINS[1:-1])
    class_ranges = ranges if ranges is not None else CLASS_RANGES
    pred_class = np.digitize(preds, bin_edges)
    corrected = preds.copy()
    for i in range(len(preds)):
        pc = int(pred_class[i])
        lo, hi = class_ranges[pc]
        bias_value = bias.get(pc, bias.get(str(pc), 0.0))
        corrected[i] = np.clip(preds[i] - bias_value, lo, hi)
    return corrected


def _stratify_bins(y: np.ndarray, clinical: float = CLINICAL, n_bins: int = 5) -> np.ndarray:
    """对 max_cobb 做分桶，用于 StratifiedKFold。"""
    p = np.percentile(y, np.linspace(0, 100, n_bins + 1))
    bins = np.sort(np.unique(np.concatenate([p, [clinical]])))
    return np.digitize(y, bins) - 1
