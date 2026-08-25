"""Mesh cut visualization panel rendering helpers.

All functions are pure rendering — no computation or I/O.
"""

from __future__ import annotations

from typing import Any

import matplotlib.tri as mtri
import numpy as np
import open3d as o3d
from matplotlib.axes import Axes


def render_mesh_panel(
    axes: Axes,
    vertex_array: np.ndarray,
    triangle_array: np.ndarray,
    colormap: str = "Greys",
    transparency: float = 0.6,
    edge_color: str = "gray",
    edge_width: float = 0.1,
) -> None:
    """用 tripcolor 单次渲染三角面（填充 + 边线）。"""
    triangulation = mtri.Triangulation(vertex_array[:, 0], vertex_array[:, 1], triangle_array)
    axes.tripcolor(
        triangulation,
        np.ones(len(vertex_array)),
        cmap=colormap,
        vmin=0,
        vmax=2,
        shading="flat",
        alpha=transparency,
        edgecolors=edge_color,
        linewidths=edge_width,
    )


def _setup_panel_axes(axes: Axes) -> None:
    """Configure subplot axes appearance."""
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(True)
        spine.set_color("#888888")
        spine.set_linewidth(0.8)


def render_valid_removed_area(
    axes: Axes,
    original_vertices: np.ndarray,
    original_triangles: np.ndarray,
    valid_tris: set[int],
) -> None:
    """Render valid removed area in light red fill."""
    if not valid_tris:
        return
    valid_faces = np.array(
        [original_triangles[tri_idx] for tri_idx in valid_tris],
        dtype=np.int32,
    )
    render_mesh_panel(
        axes,
        original_vertices,
        valid_faces,
        colormap="Reds",
        transparency=0.5,
        edge_color="#dd6666",
        edge_width=0.06,
    )


def render_invalid_removed_area(
    axes: Axes,
    original_vertices: np.ndarray,
    original_triangles: np.ndarray,
    invalid_tris: set[int],
) -> None:
    """Render invalid restored area — green fill + thin edge."""
    if not invalid_tris:
        return
    invalid_faces = np.array(
        [original_triangles[tri_idx] for tri_idx in invalid_tris],
        dtype=np.int32,
    )
    render_mesh_panel(
        axes,
        original_vertices,
        invalid_faces,
        colormap="Greens",
        transparency=0.3,
        edge_color="#338833",
        edge_width=0.06,
    )
    # Triplot overlay ensures thin strips are always visible
    inv_tri = mtri.Triangulation(
        original_vertices[:, 0],
        original_vertices[:, 1],
        invalid_faces,
    )
    axes.triplot(inv_tri, color="#338833", linewidth=0.6, alpha=0.8)


def _order_segment_path(
    segment_edges: list[tuple[int, int]],
) -> list[int]:
    """将切割边段排序为连续的顶点路径（贪心图遍历）。"""
    # 构建无向邻接表，每个顶点记录相邻顶点列表
    adj: dict[int, list[int]] = {}
    for a, b in segment_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if not adj:
        return []
    # 从度为 1 的端点出发（无端点的环则任选起点），贪心延展路径
    start = next((v for v, nb in adj.items() if len(nb) == 1), next(iter(adj)))
    path = [start]
    visited = {start}
    while True:
        candidates = [n for n in adj[path[-1]] if n not in visited]
        if not candidates:
            break
        path.append(candidates[0])
        visited.add(candidates[0])
    return path


def draw_cut_boundary_edges(
    axes: Axes,
    cut_edges: list[list[int]],
    segments: list[dict[str, Any]],
    original_vertices: np.ndarray,
) -> None:
    """Draw cut boundary edges as continuous segment polylines."""
    cut_edge_set: set[tuple[int, int]] = {tuple(e) for e in cut_edges}
    for seg in segments:
        seg_vids = set(seg.get("vertex_ids", []))
        seg_edges = [e for e in cut_edge_set if e[0] in seg_vids and e[1] in seg_vids]
        if not seg_edges:
            continue
        ordered = _order_segment_path(seg_edges)
        xs = [original_vertices[v, 0] for v in ordered]
        ys = [original_vertices[v, 1] for v in ordered]
        if seg["valid"]:
            axes.plot(xs, ys, color="#cc2222", linewidth=0.6, alpha=0.9)
        else:
            axes.plot(xs, ys, color="#338833", linewidth=0.6, alpha=0.8)


def render_panel_annotations(
    axes: Axes,
    angle: int,
    analysis_data: dict[str, Any],
    original_vertices: np.ndarray,
    kept_vertices: np.ndarray,
) -> None:
    """渲染面板标题，包含切除统计信息（移出的三角面数量和顶点数）。"""
    removals = analysis_data.get("removals", [])
    num_valid = sum(1 for r in removals if r["valid"])
    num_restored = len(analysis_data.get("restored_tris", []))
    vertices_removed = len(original_vertices) - len(kept_vertices)
    axes.set_title(
        f"<{angle} deg   -{vertices_removed}v\nvalid: {num_valid}  restored: {num_restored}",
        fontsize=9,
        pad=3,
    )


def render_single_panel(
    axes: Axes,
    angle: int,
    original_vertices: np.ndarray,
    original_triangles: np.ndarray,
    kept_mesh: o3d.geometry.TriangleMesh,
    panel_info: dict[str, Any],
    analysis_data: dict[str, Any],
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
) -> None:
    """渲染单个角度的子图面板。"""
    _setup_panel_axes(axes)
    kept_vertices = np.asarray(kept_mesh.vertices, dtype=np.float64)
    kept_triangles = np.asarray(kept_mesh.triangles)
    # Keep mesh background
    render_mesh_panel(axes, kept_vertices, kept_triangles)
    # Valid / invalid areas
    render_valid_removed_area(axes, original_vertices, original_triangles, panel_info["valid_tris"])
    render_invalid_removed_area(axes, original_vertices, original_triangles, panel_info["invalid_tris"])
    # Cut edges
    cut_edges = analysis_data.get("cut_edges", [])
    draw_cut_boundary_edges(axes, cut_edges, analysis_data.get("segments", []), original_vertices)
    # Annotation
    render_panel_annotations(axes, angle, analysis_data, original_vertices, kept_vertices)
    axes.set_xlim(x_lo, x_hi)
    axes.set_ylim(y_lo, y_hi)


def render_roi_extract_panel(
    axes: Axes,
    kept_vertices: np.ndarray,
    kept_triangles: np.ndarray,
    original_vertices: np.ndarray,
    original_triangles: np.ndarray,
    removed_tri_indices: set[int],
    title: str = "",
    *,
    kept_cmap: str = "Greys",
    kept_transparency: float = 0.9,
    kept_edge_color: str = "#888888",
    kept_edge_width: float = 0.05,
    removed_cmap: str = "Reds",
    removed_transparency: float = 0.5,
    removed_edge_color: str = "#dd6666",
    removed_edge_width: float = 0.06,
) -> None:
    """渲染单个 subject 的 ROI 提取结果面板。

    灰色填充 = 保留的 ROI mesh；浅红色填充 = 被切除的三角面。
    """
    _setup_panel_axes(axes)

    render_mesh_panel(
        axes,
        kept_vertices,
        kept_triangles,
        colormap=kept_cmap,
        transparency=kept_transparency,
        edge_color=kept_edge_color,
        edge_width=kept_edge_width,
    )

    if removed_tri_indices:
        render_valid_removed_area(axes, original_vertices, original_triangles, removed_tri_indices)

    if title:
        axes.set_title(title, fontsize=9, pad=3)

    axes.set_xlim(original_vertices[:, 0].min(), original_vertices[:, 0].max())
    axes.set_ylim(original_vertices[:, 1].min(), original_vertices[:, 1].max())
    from matplotlib.ticker import AutoLocator

    axes.xaxis.set_major_locator(AutoLocator())
    axes.yaxis.set_major_locator(AutoLocator())
    axes.set_xlabel("X (mm)", fontsize=8)
    axes.set_ylabel("Y (mm)", fontsize=8)
    axes.tick_params(labelsize=7)
