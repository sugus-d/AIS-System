"""编排层：多 subject 对比图渲染（裁剪/参数化/粗糙度对比）。

接收 CLI 参数，调用对应渲染模块生成对比图，复用 plot_shared 共享辅助。
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from commands.plot_shared import _prepare_panel_data, _save_figure
from utils.logger import logger


def render_compare(
    subject: str,
    output_dir: str,
    skip_run: bool,
    angle: int,
    dilate: int,
    min_area: float,
    min_al_ratio: float,
) -> None:
    """多 subject 法线切除结果对比。subject 可传入逗号分隔的多个 ID。

    生成 3×N 网格，每个格子显示一个 subject 在给定角度下的切除分析结果。
    绿色 = 回填区域（无效），红色 = 有效切除区域。
    """
    from mesh.roi.cleanup import cut_normal_angle_pipeline
    from visualization.cut_panels import (
        _setup_panel_axes,
        draw_cut_boundary_edges,
        render_invalid_removed_area,
        render_mesh_panel,
        render_valid_removed_area,
    )

    subject_ids = [s.strip() for s in subject.split(",")]
    n = len(subject_ids)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 5, rows * 5))
    grid = fig.add_gridspec(rows, cols, wspace=0.05, hspace=0.15, left=0.01, right=0.99, top=0.97, bottom=0.01)

    for idx, subject_id in enumerate(subject_ids):
        mesh_filepath = os.path.join("results", "cache", subject_id, "extract_roi", "output.ply")
        if not os.path.exists(mesh_filepath):
            logger.warning(f"Skip {subject_id}: mesh not found")
            continue

        try:
            original_mesh = o3d.io.read_triangle_mesh(mesh_filepath)
            original_mesh.compute_vertex_normals()
            original_vertices = np.asarray(original_mesh.vertices, dtype=np.float64)
            original_triangles = np.asarray(original_mesh.triangles)

            kept, analysis, removed = cut_normal_angle_pipeline(
                original_mesh,
                angle,
                dilate,
                min_area=min_area,
                min_al_ratio=min_al_ratio,
            )
            logger.info(f"[{idx + 1}/{n}] {subject_id}: {len(removed)} removed")

            pd = _prepare_panel_data(original_triangles, analysis, removed)

            row, col = divmod(idx, cols)
            ax = fig.add_subplot(grid[row, col])
            _setup_panel_axes(ax)

            kv = np.asarray(kept.vertices, dtype=np.float64)
            kt = np.asarray(kept.triangles)
            render_mesh_panel(ax, kv, kt)
            render_valid_removed_area(ax, original_vertices, original_triangles, pd["valid_tris"])
            render_invalid_removed_area(ax, original_vertices, original_triangles, pd["invalid_tris"])
            draw_cut_boundary_edges(ax, analysis.get("cut_edges", []), analysis.get("segments", []), original_vertices)

            n_valid = sum(1 for r in analysis["removals"] if r["valid"])
            n_restored = len(analysis.get("restored_tris", []))
            vertices_removed = len(original_vertices) - len(kv)
            ax.set_title(
                f"{subject_id}  -{vertices_removed}v\nvalid: {n_valid}  restored: {n_restored}",
                fontsize=8,
                pad=2,
            )

        except Exception as exc:
            logger.error(f"[{idx + 1}/{n}] {subject_id}: {exc}")
            row, col = divmod(idx, cols)
            ax = fig.add_subplot(grid[row, col])
            ax.text(0.5, 0.5, f"{subject_id}\nERROR", ha="center", va="center", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    os.makedirs(output_dir, exist_ok=True)
    _save_figure(fig, os.path.join(output_dir, "subjects_comparison.png"), pad_inches=0)
