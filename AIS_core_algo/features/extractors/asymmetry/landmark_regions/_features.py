"""Landmark-region feature computation (landmark_regions 计算区).

区域分类、左右不对称特征计算。多边形定义见同包 :mod:`._regions`，
候选区域特征矩阵见 :mod:`._features_candidates`。

Public: classify_by_region, compute_region_asymmetry,
compute_curvature_asymmetry, compute_candidate_asymmetry,
compute_candidate_asymmetry_pairwise, compute_region_features。
"""

from __future__ import annotations

import numpy as np

from ._regions import (
    _LEFT_POLYGONS,
    _N_REGIONS,
    _points_in_polygon,
    _POLYGON_REGIONS,
    _RIGHT_POLYGONS,
    _seed_centroids,
)


def classify_by_region(
    uv_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify UV vertices into four anatomical regions with left / right sides.

    Each of the four anatomical regions (shoulder, thoracic, lumbar, pelvic)
    is split at the spine midline (U = 0) into a left and right sub-region.

    Args:
        uv_coords: (N, 2) array of UV coordinates from the harmonic
            parameterisation.

    Returns:
        labels: (N,) int32 — region ID per vertex (``-1`` if unclassifiable).
        sides: (N,) int32 — 0 = left, 1 = right (``-1`` if unclassifiable).
    """
    N = len(uv_coords)
    labels = np.full(N, -1, dtype=np.int32)
    sides = np.full(N, -1, dtype=np.int32)

    # --- 第一阶段：多边形包含测试 ---
    # 按左右两侧分别测试每个顶点是否落在对应解剖区域的多边形内
    for rid in _POLYGON_REGIONS:
        # 左侧
        mask = _points_in_polygon(uv_coords, _LEFT_POLYGONS[rid])
        mask &= labels == -1
        labels[mask] = rid
        sides[mask] = 0

        # 右侧
        mask = _points_in_polygon(uv_coords, _RIGHT_POLYGONS[rid])
        mask &= labels == -1
        labels[mask] = rid
        sides[mask] = 1

    # --- 第二阶段：对未被任何多边形覆盖的顶点做最近质心分配 ---
    _unassigned_fallback(uv_coords, labels, sides)

    return labels, sides


def _unassigned_fallback(
    uv_coords: np.ndarray,
    labels: np.ndarray,
    sides: np.ndarray,
) -> None:
    """Assign any remaining ``-1`` vertices via nearest-centroid.

    Centroid candidates come first from already-labelled vertices; if none
    exist yet, the template landmark positions themselves are used as seed
    centroids.

    This is an in-place operation on *labels* and *sides*.
    """
    unassigned = labels == -1
    if not unassigned.any():
        return

    # 收集已分类顶点的区域质心作为参照
    centroids: dict[tuple[int, int], np.ndarray] = {}
    for rid in _POLYGON_REGIONS:
        for sd in (0, 1):
            mask = (labels == rid) & (sides == sd)
            if mask.any():
                centroids[(rid, sd)] = uv_coords[mask].mean(axis=0)

    # 如果尚未有任何顶点被分类（极端情况），使用模板骨架点位置作为种子质心
    if not centroids:
        centroids = _seed_centroids()

    keys = list(centroids.keys())
    c_array = np.array([centroids[k] for k in keys], dtype=np.float64)

    for idx in np.where(unassigned)[0]:
        dists = np.linalg.norm(c_array - uv_coords[idx], axis=1)
        best = keys[int(dists.argmin())]
        labels[idx] = best[0]
        sides[idx] = best[1]
def compute_region_asymmetry(
    values: np.ndarray,
    labels: np.ndarray,
    sides: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Compute left-right asymmetry per region for an arbitrary per-vertex quantity.

    .. math::

        R_i = |\\bar{v}_i^\\text{left} - \\bar{v}_i^\\text{right}|
        \\quad
        \\text{global} = \\sum_i w_i \\cdot R_i

    Args:
        values: (N,) array of per-vertex scalar values (e.g. mean curvature,
            Gaussian curvature, surface height, …).
        labels: (N,) int — region ID per vertex (from :func:`classify_by_region`).
        sides: (N,) int — 0 = left, 1 = right.
        weights: (R,) optional per-region weights.  Defaults to equal weights.

    Returns:
        global_asymmetry: Float scalar — weighted sum of per-region values.
        per_region: (R,) array — unweighted asymmetry per region.
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    sides = np.asarray(sides, dtype=np.int32)

    if weights is None:
        weights = np.ones(_N_REGIONS, dtype=np.float64) / _N_REGIONS
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != _N_REGIONS:
        raise ValueError(f"weights must have length {_N_REGIONS}, got {len(weights)}")

    per_region = np.zeros(_N_REGIONS, dtype=np.float64)

    for rid in range(_N_REGIONS):
        reg_mask = labels == rid
        left = reg_mask & (sides == 0)
        right = reg_mask & (sides == 1)

        if left.sum() == 0 or right.sum() == 0:
            continue

        per_region[rid] = abs(float(values[left].mean() - values[right].mean()))

    global_asymmetry = float(np.dot(weights, per_region))
    return global_asymmetry, per_region


def compute_curvature_asymmetry(
    curv_mean: np.ndarray,
    curv_gauss: np.ndarray,
    labels: np.ndarray,
    sides: np.ndarray,
    weights: np.ndarray | None = None,
    lambda_m: float = 1.0,
    lambda_g: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Compute curvature-based asymmetry index (paper §3.4.3, Eq. 10 analogue).

    .. math::

        \\text{AI} =
        \\sum_i w_i \\bigl(
            \\lambda_M |\\bar{\\kappa}_{M,i}^L - \\bar{\\kappa}_{M,i}^R|
            + \\lambda_G |\\bar{\\kappa}_{G,i}^L - \\bar{\\kappa}_{G,i}^R|
        \\bigr)

    This is the landmark-region analogue of
    :func:`features.extractors.asymmetry.asymmetric_index.compute_asymmetric_index`.

    Args:
        curv_mean: (N,) mean curvature per vertex.
        curv_gauss: (N,) Gaussian curvature per vertex.
        labels: (N,) region ID per vertex.
        sides: (N,) 0 = left, 1 = right.
        weights: (4,) per-region weights.  Defaults to equal weights.
        lambda_m: Weight for the mean-curvature term.
        lambda_g: Weight for the Gaussian-curvature term.

    Returns:
        ai_global: float — global curvature asymmetry index.
        ai_per_region: (4,) array — per-region curvature asymmetry values.
    """
    ai_mean, _ = compute_region_asymmetry(curv_mean, labels, sides)
    ai_gauss, _ = compute_region_asymmetry(curv_gauss, labels, sides)

    # Re-weight: per-region = λ_M·R_M + λ_G·R_G, then global = Σ w_i · per-region_i
    values = np.asarray(curv_mean, dtype=np.float64)
    curv_mean_reg = np.zeros(_N_REGIONS, dtype=np.float64)
    curv_gauss_reg = np.zeros(_N_REGIONS, dtype=np.float64)

    for rid in range(_N_REGIONS):
        reg_mask = labels == rid
        left = reg_mask & (sides == 0)
        right = reg_mask & (sides == 1)
        if left.sum() == 0 or right.sum() == 0:
            continue
        curv_mean_reg[rid] = abs(float(values[left].mean() - values[right].mean()))
        curv_gauss_reg[rid] = abs(float(curv_gauss[left].mean() - curv_gauss[right].mean()))

    if weights is None:
        weights = np.ones(_N_REGIONS, dtype=np.float64) / _N_REGIONS
    weights = np.asarray(weights, dtype=np.float64)

    per_region = lambda_m * curv_mean_reg + lambda_g * curv_gauss_reg
    ai_global = float(np.dot(weights, per_region))

    return ai_global, per_region
