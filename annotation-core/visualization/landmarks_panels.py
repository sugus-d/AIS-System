"""纯渲染函数：解剖标注叠加曲率图。

本文件只包含 matplotlib 渲染函数，无文件 I/O、无几何计算、不导入 open3d 或
pipeline 模块。数据通过参数以 np.ndarray 和 Python 原生类型传入。
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from visualization._render_utils import render_curvature_tripcolor

MIN_CAND_COORDS = 2  # 候选点至少包含的坐标分量数


LANDMARK_STYLE = {
    "neck_root": {"color": "cyan", "marker": "^", "s": 120, "label": "Neck root"},
    "shoulder_transition": {
        "color": "red",
        "marker": "o",
        "s": 120,
        "label": "Shoulder transition",
    },
    "scapular_peaks": {
        "color": "lime",
        "marker": "s",
        "s": 100,
        "label": "Scapular peak",
    },
    "axilla": {"color": "magenta", "marker": "v", "s": 100, "label": "Axilla"},
    "waist": {"color": "yellow", "marker": "D", "s": 100, "label": "Waist"},
    "spine_points": {"color": "white", "marker": "x", "s": 80, "label": "Spine point"},
}


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

    neck_debug = landmarks.get("neck_debug", {})
    if neck_debug:
        bin_dbg = neck_debug.get("bin_debug", [])
        for bd in bin_dbg:
            dash_len = 5.0
            alpha_val = 0.6 if bd.get("is_mode") else 0.25
            color_val = "cyan" if bd.get("is_mode") else "gray"
            lw_val = 1.5 if bd.get("is_mode") else 0.5
            ax.plot(
                [bd["x_left"] - dash_len, bd["x_left"] + dash_len],
                [bd["y"], bd["y"]],
                color=color_val,
                linewidth=lw_val,
                alpha=alpha_val,
                zorder=3,
            )
            ax.plot(
                [bd["x_right"] - dash_len, bd["x_right"] + dash_len],
                [bd["y"], bd["y"]],
                color=color_val,
                linewidth=lw_val,
                alpha=alpha_val,
                zorder=3,
            )

        cands = neck_debug.get("candidates", None)
        if not cands:
            pts: list[dict] = []
            for c in neck_debug.get("left_candidates", []):
                if isinstance(c, dict):
                    x = c.get("x")
                    y = c.get("y")
                else:
                    if len(c) >= MIN_CAND_COORDS:
                        x, y = c[0], c[1]
                    else:
                        continue
                pts.append({"x": float(x), "y": float(y)})
            for c in neck_debug.get("right_candidates", []):
                if isinstance(c, dict):
                    x = c.get("x")
                    y = c.get("y")
                else:
                    if len(c) >= MIN_CAND_COORDS:
                        x, y = c[0], c[1]
                    else:
                        continue
                pts.append({"x": float(x), "y": float(y)})
            cands = pts
        if cands:
            ax.scatter(
                [c["x"] for c in cands],
                [c["y"] for c in cands],
                c="red",
                s=4,
                alpha=0.3,
                zorder=4,
                edgecolors="none",
            )

    spine_mid = landmarks.get("spine_midline")
    if spine_mid is not None and len(spine_mid) > 1:
        ax.plot(
            spine_mid[:, 0],
            spine_mid[:, 1],
            color="gray",
            linewidth=1.0,
            alpha=0.6,
            linestyle="--",
            label="Spine midline",
        )

    ax.legend(
        loc="lower left",
        fontsize=8,
        facecolor="black",
        labelcolor="white",
        framealpha=0.6,
    )


def render_waist_debug_panel(
    ax: Axes,
    waist_debug: dict,
) -> None:
    """渲染腰部宽度调试剖面。

    Args:
        ax: matplotlib 坐标轴。
        waist_debug: 腰部调试字典，包含 y_centers, widths_raw, widths_smooth,
                     y_lo, y_hi, target_y 等键。
    """
    y_cen = waist_debug["y_centers"]
    widths_raw = waist_debug["widths_raw"]
    widths_smooth = waist_debug["widths_smooth"]
    y_lo = waist_debug["y_lo"]
    y_hi = waist_debug["y_hi"]
    target_y = waist_debug["target_y"]

    y_min_val = float(y_cen.min())
    y_range_val = float(y_cen.max()) - y_min_val
    frac = (y_cen - y_min_val) / y_range_val
    lo_frac = (y_lo - y_min_val) / y_range_val
    hi_frac = (y_hi - y_min_val) / y_range_val
    target_idx = int(np.argmin(np.abs(y_cen - target_y)))
    target_frac = frac[target_idx]
    waist_width = widths_smooth[target_idx]

    ax.set_facecolor("black")
    ax.scatter(widths_raw, frac, c="gray", s=6, alpha=0.5, label="Raw width")
    ax.plot(widths_smooth, frac, color="blue", linewidth=1.5, label="Smoothed")
    ax.axhspan(lo_frac, hi_frac, color="yellow", alpha=0.15)
    ax.axvline(
        waist_width,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Waist (frac={target_frac:.3f})",
    )
    ax.set_xlabel("Width", color="white")
    ax.set_ylabel("Y fraction", color="white")
    ax.tick_params(colors="white")
    ax.legend(
        loc="best",
        fontsize=8,
        facecolor="black",
        labelcolor="white",
        framealpha=0.6,
    )
