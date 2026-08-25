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


def _stratify_bins(y: np.ndarray, clinical: float = CLINICAL, n_bins: int = 5) -> np.ndarray:
    """对 max_cobb 做分桶，用于 StratifiedKFold。"""
    p = np.percentile(y, np.linspace(0, 100, n_bins + 1))
    bins = np.sort(np.unique(np.concatenate([p, [clinical]])))
    return np.digitize(y, bins) - 1
