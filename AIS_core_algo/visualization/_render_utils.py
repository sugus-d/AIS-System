"""共享渲染工具——绘制细节拆分复用，非业务逻辑。

本文件放置跨 panel 共用的纯渲染函数，避免重复代码。
不包含业务特定的数据访问或编排逻辑。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.axes import Axes


def save_img(
    fig: plt.Figure, image_path: str, dpi: int = 500, pad_inches: float = 0, bbox_inches: str = "tight"
) -> None:
    """保存 matplotlib figure 到磁盘（渲染编排工具，从 utils.io 迁入）。"""
    Path(image_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=dpi, pad_inches=pad_inches, bbox_inches=bbox_inches)
    plt.close(fig)


def render_curvature_tripcolor(
    ax: Axes,
    vertices: np.ndarray,
    triangles: np.ndarray,
    curvature: np.ndarray,
    cr_range: tuple[float, float] = (-0.03, 0.03),
    alpha: float = 0.5,
) -> None:
    """在 axes 上绘制曲率图（tripcolor 底图）。

    Parameters
    ----------
    ax :
        目标 axes。
    vertices :
        (N, 3) 顶点坐标。
    triangles :
        (M, 3) 三角面索引。
    curvature :
        (N,) 曲率值。
    cr_range :
        (vmin, vmax) 色域范围。
    alpha :
        透明度。
    """
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    ax.tripcolor(triang, curvature, cmap="jet", shading="gouraud", vmin=cr_range[0], vmax=cr_range[1], alpha=alpha)


def draw_angle_arc(
    ax: Axes,
    left_pt: np.ndarray,
    right_pt: np.ndarray,
    center_pt: np.ndarray,
    left_dist: float,
    right_dist: float,
    angle_deg: float,
    arcwalk_d: float = 10,
) -> None:
    """在 axes 上绘制角度弧 + 距离标注 + 角度数值。

    用于 axilla / neck_root 等需显示两条参考线夹角的 panel。
    """
    ax.scatter([left_pt[0], right_pt[0]], [left_pt[1], right_pt[1]], c="orange", marker="o", s=30, zorder=6)
    for pt, dist in [(left_pt, left_dist), (right_pt, right_dist)]:
        ax.plot([center_pt[0], pt[0]], [center_pt[1], pt[1]], color="orange", linewidth=1, alpha=0.7)
        mid = (center_pt + pt) / 2
        ax.text(
            mid[0], mid[1], f"{dist:.0f}mm", color="orange", fontsize=8, fontweight="bold", ha="center", va="bottom"
        )
    ax.text(
        center_pt[0] + arcwalk_d,
        center_pt[1] + arcwalk_d,
        f"{angle_deg:.1f}°",
        color="orange",
        fontsize=11,
        fontweight="bold",
    )


def setup_info_bar(ax: Axes) -> None:
    """将 axes 配置为黑色信息栏（无刻度、无边框）。"""
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_empty_panel(ax: Axes, title: str = "") -> None:
    """渲染无数据占位面板。"""
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, "No data", color="gray", transform=ax.transAxes, ha="center", va="center")
    if title:
        ax.set_title(title, color="gray")
