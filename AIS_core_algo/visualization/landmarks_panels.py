"""纯渲染函数：解剖标注叠加曲率图。

本文件只包含 matplotlib 渲染函数，无文件 I/O、无几何计算、不导入 open3d 或
pipeline 模块。数据通过参数以 np.ndarray 和 Python 原生类型传入。
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from visualization._render_utils import render_curvature_tripcolor

# 扁平 18 键样式（FLAT_KEYS 语义键；双侧 L/R 共享 label 去重）
_LANDMARK_STYLE_DEF = {
    "neck_root": {"color": "cyan", "marker": "^"},
    "shoulder_transition": {"color": "red", "marker": "o"},
    "scapular_peaks": {"color": "lime", "marker": "s"},
    "axilla": {"color": "magenta", "marker": "v"},
    "waist": {"color": "yellow", "marker": "D"},
    "waist_lower": {"color": "orange", "marker": "D"},
}
_LANDMARK_LABELS = {
    "neck_root": "Neck root",
    "shoulder_transition": "Shoulder transition",
    "scapular_peaks": "Scapular peak",
    "axilla": "Axilla",
    "waist": "Waist",
    "waist_lower": "Waist lower",
}

LANDMARK_STYLE: dict[str, dict] = {}
for _name, _style in _LANDMARK_STYLE_DEF.items():
    LANDMARK_STYLE[f"{_name}_L"] = {**_style, "s": 120, "label": _LANDMARK_LABELS[_name]}
    LANDMARK_STYLE[f"{_name}_R"] = {**_style, "s": 120, "label": ""}
for _key in ("neck_root_spine_point", "scapular_spine_point", "axilla_spine_point", "waist_spine_point", "waist_lower_spine_point", "thoracic_spine_point"):
    LANDMARK_STYLE[_key] = {"color": "white", "marker": "x", "s": 80, "label": "Spine point"}


def render_curvature_landmarks_panel(
    ax: Axes,
    vertices: np.ndarray,
    triangles: np.ndarray,
    curvature: np.ndarray,
    landmarks: dict,
    subject: str,
) -> None:
    """渲染曲率图上叠加解剖标注。

    Args:
        ax: matplotlib 坐标轴。
        vertices: 顶点坐标 (N, 3)。
        triangles: 三角面片 (M, 3)。
        curvature: 顶点曲率值 (N,)。
        landmarks: 标注字典。
        subject: 被试 ID。
    """
    render_curvature_tripcolor(ax, vertices, triangles, curvature)
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{subject} — Landmarks", color="white", fontsize=13)

    plotted_labels: set[str] = set()
    for key, style in LANDMARK_STYLE.items():
        pts = landmarks.get(key)
        if pts is None or len(pts) == 0:
            continue
        pts = np.asarray(pts)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        label = style["label"] if style["label"] not in plotted_labels else ""
        plotted_labels.add(style["label"])
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=style["color"],
            marker=style["marker"],
            s=style["s"],
            label=label,
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )

    ax.legend(
        loc="lower left",
        fontsize=8,
        facecolor="black",
        labelcolor="white",
        framealpha=0.6,
    )
