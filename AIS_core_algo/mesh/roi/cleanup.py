"""Public API for ROI mesh boundary cleanup and cut analysis.

Entry points
------------
cut_peninsulas
    Normal-angle protrusion detection & removal.
analyze_cut_boundary
    Statistical analysis of cut segments and removed components.
restore_invalid_cuts
    Re-attach triangles cut by invalid (thin) segments.

All functions delegate to _cut_analysis and _mesh_graph for ndarray-based
implementations.  cleanup.py is the open3d <-> ndarray bridge layer.

Notes
-----
S0009, S0005 are incomplete fragments (verts < 10k, bbox ~100mm),
not valid whole-body ROIs.  Their cut results should be disregarded.
"""

import numpy as np
import open3d as o3d

from utils.logger import logger

from ._cut_analysis import (
    analyze_cut_boundary as _analyze_cut_boundary_nd,
)
from ._cut_analysis import (
    compute_removed_triangles as _compute_removed_triangles_nd,
)
from ._cut_analysis import (
    restore_invalid_cuts as _restore_invalid_cuts_nd,
)
from ._mesh_graph import (
    build_triangle_adjacency,
)
from .bfs import largest_component

_MIN_CUT_TRIANGLES = 20  # 三角面数下限（太少直接返回，不做切割）


def _compute_horizontal_vertices(
    vertices: np.ndarray,
    triangles: np.ndarray,
    angle_threshold: float,
) -> set[int]:
    """Find vertices on near-horizontal faces (normal angle < threshold from XY)."""
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    tri_normals = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(tri_normals, axis=1, keepdims=True)
    norm[norm == 0] = 1
    tri_normals /= norm

    z_angle = np.abs(tri_normals[:, 2])
    threshold_cos = np.cos(np.radians(angle_threshold))
    horizontal_tris = np.where(z_angle < threshold_cos)[0]

    horizontal_v: set[int] = set()
    for ti in horizontal_tris:
        for v in triangles[ti]:
            horizontal_v.add(int(v))

    logger.info(
        "Horizontal vertices: %d / %d (angle < %s deg)",
        len(horizontal_v),
        len(vertices),
        angle_threshold,
    )
    return horizontal_v


def _mark_bridge_triangles(
    triangles: np.ndarray,
    horizontal_vertices: set[int],
    dilation_steps: int,
) -> np.ndarray:
    """Mark triangles to remove: those touching horizontal vertices, then dilate."""
    adj = build_triangle_adjacency(triangles)
    bridge: set[int] = set()
    for ti in range(len(triangles)):
        if any(int(v) in horizontal_vertices for v in triangles[ti]):
            bridge.add(ti)

    for _ in range(dilation_steps):
        new_bridge: set[int] = set(bridge)
        for ti in bridge:
            new_bridge.update(adj[ti])
        bridge = new_bridge

    logger.info(
        "Bridge triangles: %d / %d (dilation=%d)",
        len(bridge),
        len(triangles),
        dilation_steps,
    )
    return np.array(list(bridge), dtype=np.int32)


def _extract_kept_mesh(
    mesh: o3d.geometry.TriangleMesh,
    bridge_tris: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Remove bridge triangles and keep the largest connected component."""
    full_tris = np.asarray(mesh.triangles, dtype=np.int32)
    keep_mask = np.ones(len(full_tris), dtype=bool)
    keep_mask[bridge_tris] = False
    kept_tris = full_tris[keep_mask]

    result = o3d.geometry.TriangleMesh()
    result.vertices = mesh.vertices
    result.triangles = o3d.utility.Vector3iVector(kept_tris)
    result.remove_unreferenced_vertices()
    result = largest_component(result)

    logger.info(
        "Kept mesh: %d / %d tris, %d / %d verts",
        len(np.asarray(result.triangles)),
        len(full_tris),
        len(np.asarray(result.vertices)),
        len(np.asarray(mesh.vertices)),
    )
    return result


def _build_result_mesh(
    result_v: np.ndarray,
    result_t: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Build an open3d TriangleMesh from vertex and triangle arrays."""
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(result_v)
    result.triangles = o3d.utility.Vector3iVector(result_t)
    result.remove_unreferenced_vertices()
    result.compute_vertex_normals()
    return result


def cut_peninsulas(
    mesh: o3d.geometry.TriangleMesh,
    angle_threshold: float = 15.0,
    dilation_steps: int = 0,
    min_area: float = 150.0,
    min_al_ratio: float = 5.0,
) -> o3d.geometry.TriangleMesh:
    """ROI mesh 突起切除——水平法线切 + 有效性过滤.

    检测法线接近水平的顶点（悬挂突起物表面），标记关联三角面，
    沿邻接关系扩张后切除，保留最大连通分量，
    再将 area/length 比过低的无效切线段自动贴回。

    Args:
        mesh: 原始三角面 mesh（仅此函数接收 open3d 对象）。
        angle_threshold: 法线水平判定阈值，单位度，范围 (0, 90]。
        dilation_steps: 桥面扩张层数，默认 0。
        min_area: 有效切线最低面积（mm2），默认 150。
        min_al_ratio: 有效切线最低 area/length 比，默认 5.0。

    内部拆分为：
      _compute_horizontal_vertices()  --- 法线检测
      _mark_bridge_triangles()        --- 桥面标记 + 扩张
      _extract_kept_mesh()            --- 切除 + 最大分量
      -> restore_invalid_cuts()        --- 贴回无效切线联的三角面
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    if len(triangles) < _MIN_CUT_TRIANGLES:
        return mesh

    horizontal_v = _compute_horizontal_vertices(vertices, triangles, angle_threshold)
    if not horizontal_v:
        logger.info("No horizontal vertices found -- returning mesh unchanged")
        return mesh

    bridge_tris = _mark_bridge_triangles(triangles, horizontal_v, dilation_steps)
    if len(bridge_tris) >= len(triangles):
        logger.warning("All triangles are bridge -- returning mesh unchanged")
        return mesh

    kept = _extract_kept_mesh(mesh, bridge_tris)
    if kept.is_empty() or not kept.has_triangles():
        logger.warning("Kept mesh is empty after cut -- returning mesh unchanged")
        return mesh

    result_v, result_t = _restore_invalid_cuts_nd(
        vertices,
        triangles,
        np.asarray(kept.vertices, dtype=np.float64),
        np.asarray(kept.triangles, dtype=np.int32),
        min_area=min_area,
        min_al_ratio=min_al_ratio,
    )
    result = _build_result_mesh(result_v, result_t)

    logger.info(
        "cut_peninsulas: %dv -> %dv (angle=%s, dilate=%d)",
        len(vertices),
        len(np.asarray(result.vertices)),
        angle_threshold,
        dilation_steps,
    )
    return result


def analyze_cut_boundary(
    mesh: o3d.geometry.TriangleMesh,
    kept: o3d.geometry.TriangleMesh,
    min_seg_length: float = 3.0,
    min_area: float = 150.0,
    min_al_ratio: float = 5.0,
) -> dict:
    """Analyze cut boundary between two meshes.

    Wrapper that delegates to _cut_analysis.analyze_cut_boundary with
    ndarray arguments extracted from open3d TriangleMesh objects.
    """
    return _analyze_cut_boundary_nd(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.triangles, dtype=np.int32),
        np.asarray(kept.vertices, dtype=np.float64),
        np.asarray(kept.triangles, dtype=np.int32),
        min_seg_length=min_seg_length,
        min_area=min_area,
        min_al_ratio=min_al_ratio,
    )


def cut_normal_angle_pipeline(
    mesh: o3d.geometry.TriangleMesh,
    angle_threshold: float = 15.0,
    dilation_steps: int = 0,
    min_area: float = 150.0,
    min_al_ratio: float = 5.0,
    dilate_back: int = 3,
) -> tuple[o3d.geometry.TriangleMesh, dict, list[int]]:
    """Full cut pipeline: vertex-normal detect, dilate, cut, restore, analyze.

    使用顶点法线（而非三角面法线）检测水平面，与 ``cut_peninsulas`` 使用三角面法线的策略不同。
    返回三要素便于工具脚本直接使用。

    Args:
        mesh: 原始三角面 mesh。
        angle_threshold: 法线水平判定阈值，单位度。
        dilation_steps: 桥面扩张层数，0 = 不扩张。
        min_area: 有效切线最低面积（mm2）。
        min_al_ratio: 有效切线最低 area/length 比。
        dilate_back: 无效切线段回填时的扩张层数。

    Returns:
        (kept_mesh, analysis_dict, removed_tri_indices)
    """
    working = o3d.geometry.TriangleMesh(mesh)
    working.compute_vertex_normals()
    vertices = np.asarray(working.vertices, dtype=np.float64)
    triangles = np.asarray(working.triangles, dtype=np.int32)
    normals = np.asarray(working.vertex_normals, dtype=np.float32)

    xy_component = np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2)
    normal_angle_degrees = np.degrees(np.arctan2(np.abs(normals[:, 2]), np.maximum(xy_component, 1e-8)))
    horizontal_vertex_ids = set(np.where(normal_angle_degrees < angle_threshold)[0].tolist())

    bridge_tris: set[int] = set()
    for ti in range(len(triangles)):
        if any(int(v) in horizontal_vertex_ids for v in triangles[ti]):
            bridge_tris.add(ti)

    adj = build_triangle_adjacency(triangles)
    for _ in range(dilation_steps):
        expanded_bridge: set[int] = set(bridge_tris)
        for ti in bridge_tris:
            expanded_bridge.update(adj[ti])
        bridge_tris = expanded_bridge

    keep_mask = np.ones(len(triangles), dtype=bool)
    keep_mask[list(bridge_tris)] = False
    cut_mesh = o3d.geometry.TriangleMesh()
    cut_mesh.vertices = mesh.vertices
    cut_mesh.triangles = o3d.utility.Vector3iVector(triangles[keep_mask])
    cut_mesh.remove_unreferenced_vertices()
    cut_mesh = largest_component(cut_mesh)

    # Pre-restore removed tris (raw cut)
    kv_before = np.asarray(cut_mesh.vertices, dtype=np.float64)
    raw_removed = _compute_removed_triangles_nd(vertices, triangles, kv_before)

    # Restore invalid removals
    cut_mesh = restore_invalid_cuts(
        mesh,
        cut_mesh,
        min_area=min_area,
        min_al_ratio=min_al_ratio,
        dilate_back=dilate_back,
    )

    # Post-restore analysis
    analysis = analyze_cut_boundary(mesh, cut_mesh, min_area=min_area, min_al_ratio=min_al_ratio)

    # Post-restore removed tris
    kv_after = np.asarray(cut_mesh.vertices, dtype=np.float64)
    removed_tris = _compute_removed_triangles_nd(vertices, triangles, kv_after)

    # Restored tris = raw - post (the ones that were put back)
    post_set = set(removed_tris)
    restored_tris = [int(ti) for ti in raw_removed if int(ti) not in post_set]
    analysis["restored_tris"] = restored_tris

    return cut_mesh, analysis, removed_tris


def restore_invalid_cuts(
    mesh: o3d.geometry.TriangleMesh,
    kept: o3d.geometry.TriangleMesh,
    min_area: float = 150.0,
    min_al_ratio: float = 5.0,
    dilate_back: int = 3,
) -> o3d.geometry.TriangleMesh:
    """Restore areas cut by invalid (low area/length ratio) segments.

    Uses multi-layer expansion from invalid segment vertices to smooth
    the cut boundary.

    Args:
        dilate_back: Number of expansion layers (default 3).
    """
    result_v, result_t = _restore_invalid_cuts_nd(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.triangles, dtype=np.int32),
        np.asarray(kept.vertices, dtype=np.float64),
        np.asarray(kept.triangles, dtype=np.int32),
        min_area=min_area,
        min_al_ratio=min_al_ratio,
        dilate_back=dilate_back,
    )
    return _build_result_mesh(result_v, result_t)


__all__ = [
    "analyze_cut_boundary",
    "cut_normal_angle_pipeline",
    "cut_peninsulas",
    "restore_invalid_cuts",
]
