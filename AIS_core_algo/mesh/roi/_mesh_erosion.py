"""Mesh 边界侵蚀 — 逐层剥离最外层三角面。"""

from __future__ import annotations

import numpy as np

_MIN_KEPT_TRIANGLES = 50  # 剩余三角面数下限（太少停止剥离）


def _find_boundary_edges(
    triangles: np.ndarray,
) -> set[tuple[int, int]]:
    """找到所有边界边（只被一个三角面使用的边）。"""
    edges: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edges[key] = edges.get(key, 0) + 1

    return {e for e, count in edges.items() if count == 1}


def strip_boundary_tris(
    vertices: np.ndarray,
    triangles: np.ndarray,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """逐层剥离边界三角面，保留内部 core。

    每次迭代移除所有接触边界边的三角面。
    """
    current_t = triangles.copy()

    for _ in range(iterations):
        if len(current_t) < _MIN_KEPT_TRIANGLES:
            break

        boundary_edges = _find_boundary_edges(current_t)

        # 找到所有接触边界边的三角面
        boundary_tris = np.zeros(len(current_t), dtype=bool)
        for i, tri in enumerate(current_t):
            for j in range(3):
                a, b = int(tri[j]), int(tri[(j + 1) % 3])
                key = (a, b) if a < b else (b, a)
                if key in boundary_edges:
                    boundary_tris[i] = True
                    break

        keep = ~boundary_tris
        if keep.sum() == len(current_t) or keep.sum() == 0:
            break
        current_t = current_t[keep]

    # 重映射顶点索引
    used_verts = np.unique(current_t)
    vmap = {old: new for new, old in enumerate(used_verts)}
    remapped = np.array([[vmap[v] for v in tri] for tri in current_t], dtype=np.int64)
    out_v = vertices[used_verts].copy()
    out_t = remapped

    return out_v, out_t
