"""纯渲染函数：预测报告热力图 + landmark 连线图（物理空间）。

渲染层——无 I/O、无计算、无 open3d 依赖。数据以 np.ndarray + Python 原生
类型通过参数传入；色限（vmin/vmax）由编排层算好，地标用 flat 物理坐标。
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from visualization._render_utils import render_curvature_tripcolor

# 标注平台配色（MeshScene.tsx COLORS）：每类解剖点独立颜色，spine 白色
_LANDMARK_COLORS = {
    "neck_root": "#00ffff",
    "shoulder_transition": "#ff4444",
    "scapular_peaks": "#44ff44",
    "axilla": "#ff44ff",
    "waist": "#ffff44",
    "waist_lower": "#ff8c00",
}
# 左右对称对（扁平 _L/_R 键）
_LR_PAIRS = [
    ("neck_root_L", "neck_root_R"),
    ("shoulder_transition_L", "shoulder_transition_R"),
    ("scapular_peaks_L", "scapular_peaks_R"),
    ("axilla_L", "axilla_R"),
    ("waist_L", "waist_R"),
    ("waist_lower_L", "waist_lower_R"),
]
# 脊柱链解剖顺序（P0颈根→P1肩胛→P2腋窝→P5胸腰→P3腰→P4腰下）
_SPINE_CHAIN = [
    "neck_root_spine_point",
    "scapular_spine_point",
    "axilla_spine_point",
    "thoracic_spine_point",
    "waist_spine_point",
    "waist_lower_spine_point",
]


def render_heatmap(
    ax: Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    vmin: float,
    vmax: float,
    title: str = "",
) -> None:
    """物理空间（x-y 投影）标量场热力图（jet/gouraud，不透明）。

    Args:
        ax: 目标坐标轴。
        vertices: (N, 3) 物理空间顶点坐标（x/y 作投影轴）。
        faces: (M, 3) 三角面索引。
        values: (N,) 顶点标量。
        vmin: 色限下限（编排层算好：曲率类对称 median×5，粗糙度/法向角非对称分位）。
        vmax: 色限上限。
        title: 图题（含色限标注）。
    """
    render_curvature_tripcolor(ax, vertices, faces, values, cr_range=(vmin, vmax), alpha=1.0)
    ax.set_aspect("equal")
    ax.set_title(f"{title} [{vmin:.4g}, {vmax:.4g}]", fontsize=11)


def _overlay_landmarks(ax: Axes, flat: dict) -> None:
    """在已画好的底图上叠加 landmarks（直线连接 + 地标点）。

    供 back_panels.render_back_landmarks 复用（光照底图），
    标注逻辑一致（标注平台配色：spine 白色菱形，其余彩色圆点）。
    """
    def _point_coord(name: str) -> np.ndarray:
        return np.asarray(flat[name], dtype=np.float64)

    # 左右对称对：直线连接（颜色 = 对应解剖类）
    for l_name, r_name in _LR_PAIRS:
        pair_key = l_name.rsplit("_", 1)[0]
        ax.plot(
            [_point_coord(l_name)[0], _point_coord(r_name)[0]],
            [_point_coord(l_name)[1], _point_coord(r_name)[1]],
            color=_LANDMARK_COLORS[pair_key],
            lw=1.5,
            alpha=0.8,
            zorder=3,
        )
    # 脊柱链：直线连接（白色）
    for i in range(len(_SPINE_CHAIN) - 1):
        ax.plot(
            [_point_coord(_SPINE_CHAIN[i])[0], _point_coord(_SPINE_CHAIN[i + 1])[0]],
            [_point_coord(_SPINE_CHAIN[i])[1], _point_coord(_SPINE_CHAIN[i + 1])[1]],
            color="#ffffff",
            lw=2,
            alpha=0.9,
            zorder=3,
        )
    # 地标点（标注平台配色：spine 白色菱形，其余彩色圆点）
    for name, point in flat.items():
        if name.startswith("_"):  # 跳过 _features 等元数据键
            continue
        is_spine = name.endswith("_spine_point")
        if is_spine:
            color, size, marker = "#ffffff", 40, "D"
        else:
            key = name.rsplit("_", 1)[0]
            color, size, marker = _LANDMARK_COLORS[key], 30, "o"
        ax.scatter(
            point[0],
            point[1],
            c=color,
            s=size,
            marker=marker,
            edgecolors="k",
            linewidths=0.5,
            zorder=5,
        )


