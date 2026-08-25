"""Lateral profile visualization: plot left/right contours.

注：本模块当前仅被 tests/landmarks/test_lateral_profile.py 引用（生产零引用）。
2026-08-04 面板整理决策保留：其测试覆盖 plot_lateral_profile 的渲染路径，
作为渲染层 API 的回归锚点。若后续确认渲染规范变更可移除。
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_lateral_profile(left_c: np.ndarray, right_c: np.ndarray, save_path: str) -> str:
    """绘制左右轮廓散点图。

    Args:
        left_c: 左轮廓 (M, 2)。
        right_c: 右轮廓 (M, 2)。
        save_path: 输出图片路径。

    Returns:
        save_path。
    """
    fig, ax = plt.subplots(figsize=(6, 8), dpi=150)
    if len(left_c) > 0:
        ax.scatter(
            left_c[:, 0],
            left_c[:, 1],
            c="orange",
            s=12,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
            label="left_contour",
        )
        ax.scatter(
            left_c[0, 0],
            left_c[0, 1],
            c="red",
            s=80,
            marker="X",
            edgecolors="k",
            linewidths=0.8,
            zorder=4,
            label="left_beginning_point",
        )
    if len(right_c) > 0:
        ax.scatter(
            right_c[:, 0],
            right_c[:, 1],
            c="green",
            s=12,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
            label="right_contour",
        )
        ax.scatter(
            right_c[0, 0],
            right_c[0, 1],
            c="blue",
            s=80,
            marker="X",
            edgecolors="k",
            linewidths=0.8,
            zorder=4,
            label="right_beginning_point",
        )
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
