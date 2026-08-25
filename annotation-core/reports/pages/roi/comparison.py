"""ROI 三栏对比图 — 原图 / GT / 算法结果。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization.cut_panels import render_mesh_panel

from .data import EXPORT_DIR, gt_path, roi_path

EVAL_DIR = Path("results/eval/evaluation")
MIN_VERTICES = 50  # 顶点数低于该值视为无效网格


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    import open3d as o3d

    try:
        m = o3d.io.read_triangle_mesh(str(path))
        v = np.asarray(m.vertices, dtype=np.float64)
        t = np.asarray(m.triangles, dtype=np.int32)
        return (v, t) if len(v) >= MIN_VERTICES and len(t) > 0 else None
    except Exception:
        return None


def _simplify(v: np.ndarray, t: np.ndarray, limit: int = 50000) -> tuple[np.ndarray, np.ndarray]:
    if len(v) <= limit:
        return v, t
    import open3d as o3d

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(v)
    m.triangles = o3d.utility.Vector3iVector(t)
    return _load_mesh_simplified(m, min(30000, len(t) // 2))


def _load_mesh_simplified(m: object, target: int) -> tuple[np.ndarray, np.ndarray]:
    m = m.simplify_quadric_decimation(target)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.triangles, dtype=np.int32)


def render(sid: str) -> Path | None:
    """生成三栏对比图，返回 PNG 路径。"""
    png = EVAL_DIR / f"{sid}_comparison.png"
    rpath = roi_path(sid)

    if png.exists() and rpath.exists() and png.stat().st_mtime >= rpath.stat().st_mtime:
        return png

    orig = _load_mesh(EXPORT_DIR / sid / "original.ply")
    algo = _load_mesh(rpath)
    if orig is None or algo is None:
        return None

    orig_v, orig_t = _simplify(*orig)
    algo_v, algo_t = _simplify(*algo)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    render_mesh_panel(axes[0], orig_v, orig_t, transparency=0.7, edge_color="#888", edge_width=0.3)
    axes[0].set_title(f"{sid} 原图", fontsize=11, color="#555")
    axes[0].set_aspect("equal")

    gt = _load_mesh(gt_path(sid))
    if gt is not None:
        gt_v, gt_t = gt
        if len(gt_t) > 0:
            render_mesh_panel(axes[1], gt_v, gt_t, colormap="Purples", transparency=0.7, edge_color="#7f3b9e", edge_width=0.3)
        else:
            axes[1].scatter(gt_v[:, 0], gt_v[:, 1], c="#7f3b9e", s=0.8, alpha=0.6)
        axes[1].set_title(f"{sid} GT", fontsize=11, color="#7f3b9e")
    else:
        axes[1].set_frame_on(False)
        axes[1].text(0.5, 0.5, "GT 缺失", ha="center", va="center", fontsize=14, color="#ccc")
    axes[1].set_aspect("equal")

    all_v = np.vstack([orig_v, algo_v] + ([gt[0]] if gt is not None else []))
    xl, xh = float(all_v[:, 0].min()), float(all_v[:, 0].max())
    yl, yh = float(all_v[:, 1].min()), float(all_v[:, 1].max())
    xm, ym = (xh - xl) * 0.03, (yh - yl) * 0.03
    for ax in axes:
        ax.set_xlim(xl - xm, xh + xm)
        ax.set_ylim(yl - ym, yh + ym)

    render_mesh_panel(axes[2], algo_v, algo_t, colormap="Reds", transparency=0.7, edge_color="#d62728", edge_width=0.3)
    axes[2].set_title(f"{sid} 算法", fontsize=11, color="#d62728")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png
