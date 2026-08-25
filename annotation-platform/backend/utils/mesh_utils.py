"""网格工具 — 从核心仓库 utils.mesh 复制，标注平台自包含。

解耦后不再依赖核心仓库的 utils 模块。
"""

from __future__ import annotations

import numpy as np

from .logger import logger

_XY_NDIM = 2
_XYZ_DIM = 3


def lift_2d_to_vertex(vertices: np.ndarray, pts2d: np.ndarray | None) -> np.ndarray | None:
    """将 (N,2) 的 xy 点映射到最近的顶点，返回 (N,3)。

    若传入 pts2d 为 None，返回 None；若 pts2d 已为 (N,3) 则原样返回拷贝。
    仅按 XY 最近邻匹配，不做空间插值。
    """
    if pts2d is None:
        return None
    pts = np.asarray(pts2d)
    if pts.ndim != _XY_NDIM:
        logger.error(f"pts2d has ndim={pts.ndim}, expected 2.")
        raise ValueError("pts2d must be (N,2) or (N,3)")
    if pts.shape[1] == _XYZ_DIM:
        return pts.copy()
    verts_xy = np.asarray(vertices)[:, :2]
    out = np.zeros((pts.shape[0], 3), dtype=np.float64)
    for i, p in enumerate(pts):
        dists = np.sum((verts_xy - p[:2]) ** 2, axis=1)
        idx = int(np.argmin(dists))
        out[i] = np.asarray(vertices)[idx]
    return out
