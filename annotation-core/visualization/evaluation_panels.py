"""模型评估面板 — 混淆矩阵热力图等报告渲染组件。

reports/pages/model_evaluation 的评估图表渲染层。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def render_confusion_matrix_4class(ax: plt.Axes, cm: list[list[int]], labels: list[str]) -> None:
    """用 matplotlib 渲染混淆矩阵热力图。"""
    cm_arr = np.array(cm)
    n = len(labels)
    ax.imshow(cm_arr, cmap="Blues", alpha=0.6)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if cm_arr[i, j] > cm_arr.max() * 0.6 else "black")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
