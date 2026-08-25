"""MarginTrainer — 边界感知训练编排器。

与 Trainer 的区别：
  1. 每折偏差用 median 而非 mean，抗单折极端值
  2. 最终偏差截断到 ±15°，防止校准爆炸
  3. 分箱使用左闭右开 [lo, hi)，与 SEVERITY_BINS 定义一致
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from modeling.metrics import SEVERITY_BINS
from modeling.training.trainer import Trainer as BaseTrainer
from utils.logger import logger

MAX_BIAS = 15.0  # 最大允许偏差校正量


class MarginTrainer(BaseTrainer):
    """边界感知训练编排器 — 继承 BaseTrainer，覆盖校准方法。"""

    def _apply_calibration(
        self,
        preds: NDArray,
        y: NDArray,
        fold_preds: list[tuple[NDArray, NDArray]],
    ) -> tuple[NDArray, dict]:
        """Per-class 偏差校正（稳健版）。

        改进 vs BaseTrainer._apply_calibration：
          1. 每折偏差用 median 而非 mean（抗极端值）
          2. 跨折偏差截断到 ±15°（防爆炸）
          3. 分箱使用左闭右开 [lo, hi)，与 compute_4class_metrics 一致
        """
        boundaries = SEVERITY_BINS[1:-1]  # [10, 20, 40]

        # 收集每折每类的偏差（左闭右开）
        class_biases: dict[int, list[float]] = {c: [] for c in range(4)}
        for te_idx, y_pred_te in fold_preds:
            y_true_te = y[te_idx]
            labels = np.digitize(y_true_te, boundaries)
            for c in range(4):
                mask = labels == c
                if mask.sum() > 0:
                    resid = y_pred_te[mask] - y_true_te[mask]
                    class_biases[c].append(float(np.median(resid)))

        # 跨折偏差（median） + 截断
        avg_bias: dict[int, float] = {}
        clipped: dict[int, float] = {}
        for c in range(4):
            vals = class_biases[c]
            raw = float(np.median(vals)) if vals else 0.0
            if abs(raw) > MAX_BIAS:
                clipped[c] = raw
                avg_bias[c] = MAX_BIAS if raw > 0 else -MAX_BIAS
            else:
                avg_bias[c] = raw

        if clipped:
            logger.warning(f"校准: 偏差截断 {clipped}")

        # 应用校正 + 边界保护（复用 modeling._shared.apply_calibration 单点）
        from modeling._shared import apply_calibration

        corrected = apply_calibration(preds, avg_bias)

        return corrected, {
            "bias": avg_bias,
            "fold_counts": {c: len(v) for c, v in class_biases.items()},
            "clipped": clipped,
        }
