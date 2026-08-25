"""Procrustes 相似变换工具。

计算两个二维点集之间的最优相似变换（缩放 + 旋转 + 平移），
供参数化管线和可视化面板共用。
"""

import numpy as np


def compute_procrustes(
    src_2d: np.ndarray,
    tgt_2d: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Procrustes 相似变换：s·R·src + t = tgt。

    通过 SVD 求解最优旋转、缩放和平移，将源点集对齐到目标点集。

    Args:
        src_2d: (M, 2) 源点集。
        tgt_2d: (M, 2) 目标点集。

    Returns:
        (s, R, t): 缩放因子、旋转矩阵、平移向量，
        满足 s * R @ src + t ≈ tgt。
    """
    sc, tc = src_2d - src_2d.mean(0), tgt_2d - tgt_2d.mean(0)
    s = np.linalg.norm(tc) / max(np.linalg.norm(sc), 1e-10)
    U, _, Vt = np.linalg.svd(sc.T @ tc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = tgt_2d.mean(0) - s * src_2d.mean(0) @ R.T
    return s, R, t
