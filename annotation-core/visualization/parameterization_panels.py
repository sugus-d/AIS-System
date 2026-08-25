"""Parameterization visualization panels — pure rendering, no I/O."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.interpolate import griddata

from parameterization.procrustes import compute_procrustes
from parameterization.template import TEMPLATE_LANDMARKS

_OUTER_NAMES = [
    "neck_root_L",
    "shoulder_transition_L",
    "axilla_L",
    "waist_L",
    "waist_lower_L",
    "spine_P4",
    "waist_lower_R",
    "waist_R",
    "axilla_R",
    "shoulder_transition_R",
    "neck_root_R",
    "spine_P0",
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
    V: np.ndarray,
    Va: np.ndarray,
    k: np.ndarray,
    bverts: np.ndarray,
    ov: np.ndarray,
    mask: np.ndarray,
    sid: str,
) -> tuple[Figure, Axes]:
    """单面板切割可视化：外部灰色、内部按 Z 高度着色。

    Returns (fig, ax) — 由调用方保存。
    """
    import matplotlib.pyplot as plt

    inside_viz = mask.copy()
    outside_viz = ~inside_viz

    fig, ax = plt.subplots(figsize=(12, 12))

    # 内部区域：按 Z 高度着色（viridis 色图），显示三维形态
    # Interior: colored by height
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

    # 外部边界标注点（红色大圆点）：肩臂转点、颈根、腋窝、腰部
    names = list(TEMPLATE_LANDMARKS.keys())
    for i, n in enumerate(_OUTER_NAMES):
        vi = ov[i]
        ax.scatter(Va[vi, 0], Va[vi, 1], c="red", s=150, edgecolors="white", lw=2, zorder=4)
        ax.text(Va[vi, 0] + 0.2, Va[vi, 1], _short_label(n), fontsize=9, c="red", fontweight="bold")

    # 内部标注点（青色菱形）：肩胛峰、脊柱中点
    for n in ["scapular_peaks_L", "scapular_peaks_R", "spine_P1", "spine_P2"]:
        vi = int(k[names.index(n)])
        ax.scatter(Va[vi, 0], Va[vi, 1], c="cyan", s=120, marker="D", edgecolors="white", lw=1.5, zorder=4)
        ax.text(Va[vi, 0] + 0.2, Va[vi, 1], _short_label(n), fontsize=9, c="cyan", fontweight="bold")

    ax.set_aspect("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 3)
    ax.set_title(f"Geodesic Cut — {sid} — {int(np.sum(inside_viz))}/{len(Va)} inside", fontsize=13)

    return fig, ax


def draw_heightmap(
    V: np.ndarray,
    Fc: np.ndarray,
    uv: np.ndarray,
    k: np.ndarray,
    sid: str,
) -> tuple[Figure, list[Axes]]:
    """三面板图：对齐网格、调和 UV、高度图。

    Returns (fig, axes) — 由调用方保存。
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 面板 1：Procrustes 对齐后的网格（按 Z 高度着色 + 外部标注点红色标记）
    ax = axes[0]
    Va = V.copy()
    s, R, t = compute_procrustes(V[k, :2], np.array([TEMPLATE_LANDMARKS[n] for n in list(TEMPLATE_LANDMARKS.keys())]))
    Va[:, :2] = s * V[:, :2] @ R.T + t
    ax.scatter(Va[:, 0], Va[:, 1], c=V[:, 2], s=2, cmap="viridis", alpha=0.5)
    for n in _OUTER_NAMES:
        vi = int(k[list(TEMPLATE_LANDMARKS.keys()).index(n)])
        ax.scatter(Va[vi, 0], Va[vi, 1], c="red", s=50)
    ax.set_aspect("equal")
    ax.set_title(f"Aligned mesh ({len(V)}v)", fontsize=11)

    # 面板 2：UV 参数化（调和映射结果，按 Z 高度着色）
    ax = axes[1]
    ax.tripcolor(uv[:, 0], uv[:, 1], Fc, V[:, 2], cmap="viridis", shading="gouraud")
    ax.set_aspect("equal")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-4.5, 2.5)
    ax.set_title("Harmonic UV", fontsize=11)

    # 面板 3：高度图（UV 空间插值成规则网格后 imshow 显示）
    ax = axes[2]
    xi = np.linspace(-2.5, 2.5, 256)
    yi = np.linspace(-4, 2, 256)
    XX, YY = np.meshgrid(xi, yi)
    Zg = griddata(uv, V[:, 2], (XX, YY), method="linear", fill_value=np.nan)
    im = ax.imshow(Zg, extent=[-2.5, 2.5, 2, -4], cmap="plasma", origin="upper", aspect="equal")
    for n in list(TEMPLATE_LANDMARKS.keys()):
        u, v = TEMPLATE_LANDMARKS[n]
        ax.scatter(u, v, c="white", s=30, edgecolors="k", zorder=5)
    ax.set_title("Height Map", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    return fig, list(axes)
