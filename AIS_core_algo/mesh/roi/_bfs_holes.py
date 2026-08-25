"""BFS hole filling — boundary-based hole detection and filling."""

import numpy as np

from utils.logger import logger

from ._mesh_graph import (
    build_edge_to_triangles,
    compute_triangle_areas,
)


def _fill_holes_by_boundary(
    component: set[int],
    vertices: np.ndarray,
    triangles: np.ndarray,
    adj: list[set[int]],
    max_hole_boundary: float = 200.0,
    max_hole_area: float = 3000.0,
    edge_to_tri: dict | None = None,
    tri_areas: np.ndarray | None = None,
) -> set[int]:
    """按外边界长度/面积补洞：找到 removed 连通分量的外边界，小的才补。

    真正的孔洞检测：
    1. 找出所有 removed 三角形的连通分量。
    2. 对每个分量，计算其外边界的总长度（与 kept 共享的边）和总面积。
    3. 如果两者都小于阈值，则视为内部空洞，全部补上。

    Parameters
    ----------
    component: 当前 kept 三角形索引集合。
    vertices: (n_vert, 3) 顶点数组。
    triangles: (n_tri, 3) 三角面数组。
    adj: 三角形邻接表。
    max_hole_boundary: 孔洞外边界最大长度（mm），默认 200mm。
    max_hole_area: 孔洞最大面积（mm²），默认 3000mm²。
    edge_to_tri: 预计算的边→三角面字典，为 None 则内部构建。
    tri_areas: 预计算的三角面面积数组，为 None 则内部计算。

    Returns 补洞后的 component set。
    """
    result = set(component)
    n_tri = len(triangles)
    all_tri_set = set(range(n_tri))

    # 找出所有 removed 三角形的连通分量
    removed_set = all_tri_set - result
    if not removed_set:
        return result

    if edge_to_tri is None:
        edge_to_tri = build_edge_to_triangles(triangles)
    if tri_areas is None:
        tri_areas = compute_triangle_areas(vertices, triangles)

    # 找出所有 removed 三角形的连通分量（完整 BFS，不早退，保证正确性）
    visited: set[int] = set()
    hole_components: list[list[int]] = []
    for seed in removed_set:
        if seed in visited:
            continue
        comp: list[int] = []
        stack = [seed]
        visited.add(seed)
        while stack:
            ti = stack.pop()
            comp.append(ti)
            for nj in adj[ti]:
                if nj in removed_set and nj not in visited:
                    visited.add(nj)
                    stack.append(nj)
        hole_components.append(comp)

    n_filled = 0
    for hole in hole_components:
        # 先算面积（O(n) C 级操作），面积超阈值则跳过边界长度计算（Python 循环较慢）
        hole_area = float(np.sum(tri_areas[hole]))
        if hole_area > max_hole_area:
            continue

        # 计算该孔洞的外边界长度（与 kept 共享的边的总长度）
        boundary_len = 0.0
        for ti in hole:
            a, b, c = triangles[ti]
            for edge in [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))]:
                for nj in edge_to_tri.get(edge, []):
                    if nj in result:  # 相邻的是 kept → 这条边是外边界
                        boundary_len += float(np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]))
                        break

        if boundary_len <= max_hole_boundary:
            for ti in hole:
                result.add(ti)
            n_filled += 1

    if n_filled > 0:
        logger.info(
            "Hole fill by boundary: filled %d holes (max_len=%.0fmm max_area=%.0fmm²)",
            n_filled,
            max_hole_boundary,
            max_hole_area,
        )
    return result
