"""沿多边形切割 mesh — 三角面分类法。"""

from __future__ import annotations

import numpy as np


def _is_inside(x: float, y: float, polygon: np.ndarray) -> bool:
    """射线法 point-in-polygon。"""
    n = len(polygon)
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # 水平射线向右，检查边是否跨越射线
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


def _keep_mask(
    tri_centers_xy: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """返回布尔 mask：哪些三角面在多边形内。"""
    mask = np.zeros(len(tri_centers_xy), dtype=bool)
    for i, (x, y) in enumerate(tri_centers_xy):
        mask[i] = _is_inside(x, y, polygon)
    return mask


def _extract_submesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    keep_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """根据 mask 提取子 mesh，重映射顶点索引。"""
    keep_tris = triangles[keep_mask]
    if len(keep_tris) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    used_verts = np.unique(keep_tris)
    vert_map = {old: new for new, old in enumerate(used_verts)}
    remapped = np.array([[vert_map[v] for v in tri] for tri in keep_tris], dtype=np.int64)

    return vertices[used_verts].copy(), remapped


def filter_by_polygon(
    vertices: np.ndarray,
    triangles: np.ndarray,
    polygon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """保留多边形内部的三角面。

    polygon: (M, 2) 的 (X, Y) 封闭曲线。
    """
    centers = vertices[triangles].mean(axis=1)
    mask = _keep_mask(centers[:, :2], polygon)
    if mask.sum() == 0:
        return vertices.copy(), triangles.copy()
    return _extract_submesh(vertices, triangles, mask)
