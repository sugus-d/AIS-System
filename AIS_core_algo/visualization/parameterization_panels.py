"""Parameterization visualization panels — 纯渲染（接收 Axes，无 I/O 无计算）。

`draw_cut` 在传入 Axes 上绘制测地切割图；数据准备（Procrustes 对齐等）由
编排层 `commands/plot_parameterization.py` 完成，渲染层不 import 算法模块。
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

_OUTER_NAMES = [
    "neck_root_L",
    "shoulder_transition_L",
    "axilla_L",
    "waist_L",
    "waist_lower_L",
    "waist_lower_spine_point",
    "waist_lower_R",
    "waist_R",
    "axilla_R",
    "shoulder_transition_R",
    "neck_root_R",
    "neck_root_spine_point",
]


def _short_label(name: str) -> str:
    """生成标注点名称的简短别名（用于图面标注）。"""
    return (
        name.replace("shoulder_transition", "ST")
        .replace("neck_root", "NR")
        .replace("spine_", "P")
        .replace("axilla", "AX")
        .replace("waist", "WA")
        .replace("scapular_peaks", "SP")
        .replace("_", "")
    )


def draw_cut(
    ax: Axes,
    V: np.ndarray,
    Va: np.ndarray,
    bverts: np.ndarray,
    ov: np.ndarray,
    inner_idx: np.ndarray,
    mask: np.ndarray,
    sid: str,
) -> None:
    """单面板切割可视化：外部灰色、内部按 Z 高度着色。

    在传入 Axes 上绘制（figure/save 由编排层管理）。

    Args:
        ax: 目标坐标轴。
        V: (N, 3) 原始顶点。
        Va: (N, 3) Procrustes 对齐后顶点。
        bverts: 测地边界顶点索引。
        ov: 外部 landmark 顶点索引。
        inner_idx: 内部标注点顶点索引（肩胛峰/脊柱中点等，len=4）。
        mask: (N,) bool，True=边界内。
        sid: subject ID（用于标题）。
    """
    inside_viz = mask.copy()
    outside_viz = ~inside_viz

    # 内部区域：按 Z 高度着色（viridis 色图），显示三维形态
    ax.scatter(
        Va[inside_viz, 0],
        Va[inside_viz, 1],
        c=V[inside_viz, 2],
        s=8,
        cmap="viridis",
        alpha=0.6,
        zorder=1,
        edgecolors="none",
    )

    # 边界红色粗线（切割线）
    ax.plot(Va[bverts, 0], Va[bverts, 1], "r-", lw=3, zorder=2)

    # 外部区域：灰色，与内部形成对比
    ax.scatter(
        Va[outside_viz, 0],
        Va[outside_viz, 1],
        s=100,
        c="#8A8A8A",
        alpha=0.8,
        edgecolors="none",
        zorder=3,
    )

    # 外部边界标注点（红色大圆点）
    for i, n in enumerate(_OUTER_NAMES):
        vi = ov[i]
        ax.scatter(Va[vi, 0], Va[vi, 1], c="red", s=150, edgecolors="white", lw=2, zorder=4)
        ax.text(Va[vi, 0] + 0.2, Va[vi, 1], _short_label(n), fontsize=9, c="red", fontweight="bold")

    # 内部标注点（青色菱形）：肩胛峰、脊柱中点（位置由编排层按 TEMPLATE_LANDMARKS 顺序传入）
    for vi, n in zip(inner_idx, ["scapular_peaks_L", "scapular_peaks_R", "scapular_spine_point", "axilla_spine_point"], strict=False):
        ax.scatter(Va[int(vi), 0], Va[int(vi), 1], c="cyan", s=120, marker="D", edgecolors="white", lw=1.5, zorder=4)
        ax.text(Va[int(vi), 0] + 0.2, Va[int(vi), 1], _short_label(n), fontsize=9, c="cyan", fontweight="bold")

    ax.set_aspect("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 3)
    ax.set_title(f"Geodesic Cut — {sid} — {int(np.sum(inside_viz))}/{len(Va)} inside", fontsize=13)
