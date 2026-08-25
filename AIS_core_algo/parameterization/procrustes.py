"""Procrustes 相似变换工具。

计算两个点集之间的最优相似变换（缩放 + 旋转 + 平移），任意维度（2D/3D）通用。
参数化管线、可视化面板与 landmark 补全（prediction）共用同一实现。

注意：缩放取源/目标点集范数比（RMS 比），非 Umeyama 奇异值公式——参数化 landmark
近共线分布下两公式分歧可达 ~1.4（见 2026-08 回归记录），此处必须保留范数比口径。
"""

from __future__ import annotations

import numpy as np

# 源点集范数下限（退化保护，缩放截断为大数）
_MIN_NORM = 1e-10


def compute_procrustes(
    src: np.ndarray,
    tgt: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Procrustes 相似变换：s·R·src + t ≈ tgt。

    通过 SVD 求解最优旋转、缩放和平移，将源点集对齐到目标点集。

    Args:
        src: (M, D) 源点集。
        tgt: (M, D) 目标点集。

    Returns:
        (s, R, t): 缩放因子、旋转矩阵、平移向量，
        满足 s * R @ src + t ≈ tgt（也即 s·src@R.T + t，行向量等价）。
    """
    sc, tc = src - src.mean(0), tgt - tgt.mean(0)
    scale = float(np.linalg.norm(tc) / max(np.linalg.norm(sc), _MIN_NORM))
    u, _, vt = np.linalg.svd(sc.T @ tc)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = tgt.mean(0) - scale * src.mean(0) @ rotation.T
    return scale, rotation, translation
