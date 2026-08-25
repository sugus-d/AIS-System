"""Moiré 条纹算法 — 每顶点到参考平面的距离计算。

渲染已抽到 `visualization/moire_panels.py::render_moire`（三层分离）；
本模块只保留算法层 :func:`compute_moire_distances`。
参考平面 `ax + by + cz + d = 0` 默认 `z = -20`（d=20），与旧版管线一致。
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from mesh.preprocess.alignment import calculate_distance_from_plane

__all__ = ["compute_moire_distances"]


def compute_moire_distances(
    mesh: o3d.geometry.TriangleMesh,
    plane_a: float = 0,
    plane_b: float = 0,
    plane_c: float = 1,
    plane_d: float = 20,
) -> np.ndarray:
    """计算网格每顶点到参考平面 (ax+by+cz+d=0) 的绝对距离，作为条纹层级依据。

    Args:
        mesh: ROI 三角网格。
        plane_a/b/c/d: 参考平面系数，默认 z=-20 平面。

    Returns:
        (N,) 每顶点距离。
    """
    return calculate_distance_from_plane(mesh, plane_a, plane_b, plane_c, plane_d)
