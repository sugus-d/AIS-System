"""Landmark-region feature computation (landmark_regions 计算区).

区域分类、左右不对称特征计算。多边形定义见同包 :mod:`._regions`。

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
    build_candidate_polygons,
    SEG_LUMBAR,
    SEG_SHOULDER,
    SEG_THORACIC,
)

_MIN_POLYGON_POINTS = 3
"""多边形内有效顶点数下限，低于该值视为区域不可计算。"""


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
def compute_candidate_asymmetry(
    uv_coords: np.ndarray,
    heights: np.ndarray,
    curv_mean: np.ndarray,
    curv_gauss: np.ndarray,
    roughness: np.ndarray | None = None,
    normal_angle: np.ndarray | None = None,
    vertex_normals: np.ndarray | None = None,
    verbose: bool = False,
    version: str = "bilateral",
) -> tuple[np.ndarray, list[str]]:
    """Compute |Δmean| for every candidate region × every available measure.

    ... (see docstring)
    """
    uv = np.asarray(uv_coords, dtype=np.float64)

    # Active measures: 5 scalar + 3 vector-derived
    measures: list[tuple[str, np.ndarray]] = [
        ("height", heights),
        ("mean_curv", curv_mean),
        ("gauss_curv", curv_gauss),
    ]
    if roughness is not None:
        measures.append(("roughness", roughness))
    if normal_angle is not None:
        measures.append(("normal_angle", normal_angle))
    Q = len(measures)

    # Vector-derived measure (normal_vector_cos only)
    has_vectors = vertex_normals is not None
    Q_vec = 1 if has_vectors else 0
    Q_total = Q + Q_vec

    candidates = build_candidate_polygons(version=version)
    R = len(candidates)
    features = np.zeros((R, Q_total), dtype=np.float64)
    feature_names: list[str] = []

    # V2: bilateral pairs — independent left / right polygons
    for ci, cand in enumerate(candidates):
        left_mask = _points_in_polygon(uv, cand["left_polygon"])
        right_mask = _points_in_polygon(uv, cand["right_polygon"])
        if int(left_mask.sum()) < _MIN_POLYGON_POINTS or int(right_mask.sum()) < _MIN_POLYGON_POINTS:
            feature_names += [f"{cand['name']}_{m}" for m, _ in measures]
            if has_vectors:
                feature_names += [
                    f"{cand['name']}_normal_vector_cos",
                ]
            continue

        # Scalar measures
        for qi, (m_name, vals) in enumerate(measures):
            v = np.asarray(vals, dtype=np.float64)
            features[ci, qi] = abs(float(v[left_mask].mean() - v[right_mask].mean()))
            feature_names.append(f"{cand['name']}_{m_name}")

        # Vector-derived measure: cos of angle between mean normals
        if has_vectors and vertex_normals is not None:
            nl = vertex_normals[left_mask].mean(axis=0)
            nr = vertex_normals[right_mask].mean(axis=0)
            nl /= np.linalg.norm(nl)
            nr /= np.linalg.norm(nr)
            cos_a = float(np.clip(np.dot(nl, nr), -1, 1))

            features[ci, Q] = cos_a
            feature_names += [
                f"{cand['name']}_normal_vector_cos",
            ]

    return features, feature_names


def compute_candidate_asymmetry_pairwise(
    uv_coords: np.ndarray,
    heights: np.ndarray,
    curv_mean: np.ndarray,
    curv_gauss: np.ndarray,
    roughness: np.ndarray | None = None,
    normal_angle: np.ndarray | None = None,
    vertex_normals: np.ndarray | None = None,
    vertices: np.ndarray | None = None,
    verbose: bool = False,
    version: str = "bilateral",
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise |Δ| for every bilateral region pair × every available measure.

    Instead of |mean(L) - mean(R)|, this computes:
      1. For each vertex in the left polygon, mirror its UV across U=0 → (-u, v).
      2. Find the corresponding vertex in the right polygon via nearest UV
         (on the right polygon's vertices).
      3. |value_left - value_right_mirrored|  (for scalars)
      4. For vectors (normal_vector/cos/sin): reflect the right normal across a plane
         whose normal is the local left→right axis (P_right - P_left), then compute
         the angle between n_left and the reflected n_right.  This is more robust
         than world X-axis mirroring because it follows the subject's actual pose.
      5. Average all pointwise differences → asymmetry.

    Args:
        uv_coords: (N, 2) UV coordinates.
        heights: (N,) per-vertex height.
        curv_mean: (N,) mean curvature.
        curv_gauss: (N,) Gaussian curvature.
        roughness: (N,) optional roughness.
        normal_angle: (N,) optional normal angle (deg from vertical).
        vertex_normals: (N, 3) optional, required for normal-vector angle.
        vertices: (N, 3) optional 3D vertex positions, required for local-axis mirror.
        verbose: If True, print progress.
        version: Region definition version.

    Returns:
        features: (N_candidates, M) — mean pairwise |Δ| per row and column.
        feature_names: list[str] — ``"{name}_{measure}"``.
    """
    uv = np.asarray(uv_coords, dtype=np.float64)

    # Active measures
    measure_arrays: list[tuple[str, np.ndarray]] = [
        ("height", heights),
        ("mean_curv", curv_mean),
        ("gauss_curv", curv_gauss),
    ]
    if roughness is not None:
        measure_arrays.append(("roughness", roughness))
    if normal_angle is not None:
        measure_arrays.append(("normal_angle", normal_angle))
    Q = len(measure_arrays)

    # Vector-angle measure (requires vertex_normals)
    has_vectors = vertex_normals is not None
    Q_vec = 1 if has_vectors else 0
    Q_total = Q + Q_vec

    candidates = build_candidate_polygons(version=version)
    R = len(candidates)
    features = np.zeros((R, Q_total), dtype=np.float64)
    feature_names: list[str] = []

    for ci, cand in enumerate(candidates):
        left_mask = _points_in_polygon(uv, cand["left_polygon"])
        right_mask = _points_in_polygon(uv, cand["right_polygon"])
        left_idx = np.where(left_mask)[0]
        right_idx_set = set(np.where(right_mask)[0])

        if len(left_idx) < 3 or len(right_idx_set) < 3:  # noqa: PLR2004
            for m_name, _ in measure_arrays:
                feature_names.append(f"{cand['name']}_{m_name}")
            if has_vectors:
                feature_names += [
                    f"{cand['name']}_normal_vector_cos",
                ]
            continue

        # Build UV -> vertex index mapping for the right polygon
        uv_right = uv[right_mask]
        right_order = np.where(right_mask)[0]

        # Precompute mirrored UV for all left vertices
        mirrored_u = -uv[left_idx, 0]
        mirrored_v = uv[left_idx, 1]

        # For each left vertex, find nearest UV in right polygon
        diffs_scalar = {m: np.zeros(len(left_idx), dtype=np.float64) for m, _ in measure_arrays}
        diffs_cos = np.zeros(len(left_idx), dtype=np.float64) if has_vectors else None

        for pi, (lvi, mu, mv) in enumerate(zip(left_idx, mirrored_u, mirrored_v, strict=False)):
            # Find nearest vertex in right polygon by UV distance
            d = np.hypot(uv_right[:, 0] - mu, uv_right[:, 1] - mv)
            rvi = right_order[int(d.argmin())]
            d_min = d.min()

            # Only pair if the mirrored UV falls within a reasonable tolerance
            _UV_TOLERANCE = 0.05  # ~1 % of UV width
            if d_min > _UV_TOLERANCE:
                continue

            # Scalar differences
            for m_name, arr in measure_arrays:
                diffs_scalar[m_name][pi] = abs(float(arr[lvi]) - float(arr[rvi]))

            # Vector angle (3D normals) — cos of reflected angle
            if has_vectors and vertex_normals is not None and vertices is not None:
                n_left = vertex_normals[lvi]
                n_right = vertex_normals[rvi]
                # 局部左右轴：右点→左点的连线方向（替代世界 X 轴翻转）
                local_axis = vertices[rvi] - vertices[lvi]
                local_norm = np.linalg.norm(local_axis)
                _NORMAL_EPS = 1e-8  # 防止零向量除
                if local_norm > _NORMAL_EPS:
                    local_axis = local_axis / local_norm
                    # 将右侧法向量关于垂直于 local_axis 的平面做镜像
                    n_reflected = n_right - 2 * np.dot(n_right, local_axis) * local_axis
                else:
                    n_reflected = n_right
                cos_a = float(np.clip(np.dot(n_left, n_reflected), -1, 1))
                diffs_cos[pi] = cos_a

        # Mean of pairwise differences
        for qi, (m_name, _) in enumerate(measure_arrays):
            vals = diffs_scalar[m_name]
            features[ci, qi] = float(vals[vals > 0].mean()) if (vals > 0).any() else 0.0
            feature_names.append(f"{cand['name']}_{m_name}__pw")

        if has_vectors and diffs_cos is not None:
            features[ci, Q] = float(diffs_cos[diffs_cos != 0].mean()) if (diffs_cos != 0).any() else 0.0
            feature_names += [
                f"{cand['name']}_normal_vector_cos__pw",
            ]

    return features, feature_names


def compute_region_features(
    heights: np.ndarray,
    curv_mean: np.ndarray,
    curv_gauss: np.ndarray,
    labels: np.ndarray,
    sides: np.ndarray,
    region_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute raw |Δmean| feature matrix: R regions × Q measures.

    For each region, computes the absolute difference of mean values between
    left and right sides for each of the three per-vertex measures (height,
    mean curvature, Gaussian curvature).

    .. math::

        F_{r,q} = \\left|\\bar{v}_{r,q}^{\\text{left}}
        - \\bar{v}_{r,q}^{\\text{right}}\\right|

    where :math:`r` indexes a region row and :math:`q` indexes a measure
    column.

    By default the Pelvic region (ID 3) is excluded from the feature matrix
    since its bottom-unbounded polygon geometry makes its asymmetry signal
    unreliable.

    Args:
        heights: (N,) per-vertex height values.
        curv_mean: (N,) mean curvature per vertex.
        curv_gauss: (N,) Gaussian curvature per vertex.
        labels: (N,) region ID per vertex (from :func:`classify_by_region`).
        sides: (N,) 0 = left, 1 = right.
        region_ids: (R,) region IDs to include.  Defaults to
            ``[Shoulder, Thoracic, Lumbar]`` (Pelvic excluded).

    Returns:
        features: (R, 3) array — **|Δmean|** per region (rows) and per
            measure (columns: height, mean curvature, Gaussian curvature).
        feature_names: list of 3 column-name strings.

    Raises:
        ValueError: If any input array has a different length from the
            others.
    """
    arrays = [heights, curv_mean, curv_gauss, labels, sides]
    N = len(arrays[0])
    for arr in arrays[1:]:
        if len(arr) != N:
            raise ValueError(f"all input arrays must have the same length; got lengths {[len(a) for a in arrays]}")

    if region_ids is None:
        region_ids = np.array([SEG_SHOULDER, SEG_THORACIC, SEG_LUMBAR], dtype=np.int32)
    region_ids = np.asarray(region_ids, dtype=np.int32)
    R = len(region_ids)
    Q = 3  # height, curv_mean, curv_gauss

    features = np.zeros((R, Q), dtype=np.float64)

    # 对每个区域计算 |Δmean|：三种测量值（高度、平均曲率、高斯曲率）各自取左右均值的绝对值差
    for i, rid in enumerate(region_ids):
        reg_mask = labels == rid
        left = reg_mask & (sides == 0)
        right = reg_mask & (sides == 1)

        # 单侧顶点缺失时跳过，保持该行特征为零
        if left.sum() == 0 or right.sum() == 0:
            continue

        for j, vals in enumerate([heights, curv_mean, curv_gauss]):
            features[i, j] = abs(float(vals[left].mean() - vals[right].mean()))

    feature_names = ["height", "curv_mean", "curv_gauss"]

    return features, feature_names
