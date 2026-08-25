"""左右不对称差值计算（DM 和 PW 两种方案）。

DM (direct-mean):
    |mean(左侧遮罩) - mean(右侧遮罩)|
    对标量测量直接差绝对值；对法向量，分别平均左右侧后算向量夹角。

PW (pairwise):
    将左顶点按 UV 镜射 (u -> -u)，在右侧找最近邻配对，
    逐点 |值_L - 值_R| 后平均。
    对法向量：将右侧法向量关于局部左右轴做反射后算夹角。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# DM — 直接左右均值差
# ---------------------------------------------------------------------------


def compute_asymmetry_dm(
    scalar_measures: dict[str, np.ndarray],
    left_masks: list[np.ndarray],
    right_masks: list[np.ndarray],
    region_names: list[str],
    vertex_normals: np.ndarray | None = None,
    min_vertices: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """计算 DM 模式 |Δmean| 特征。

    Args:
        scalar_measures: 标量测量字典，键为测量名，值为 (N,) ndarray。
            典型键值: height, mean_curv, gauss_curv, roughness, normal_angle。
        left_masks: R 个 (N,) bool 数组，各区左遮罩。
        right_masks: R 个 (N,) bool 数组，各区右遮罩。
        region_names: 区域名称列表 (R,)。
        vertex_normals: (N, 3) 可选，存在时额外计算 normal_vector/cos/sin。
        min_vertices: 单侧最少顶点数，不足时该区域跳过。

    Returns:
        (features, feature_names):
            features: (R, Q) — Q 为测量数（标量 + 3 向量衍生）。
            feature_names: 列名列表。
    """
    measures_list = list(scalar_measures.items())
    R = len(region_names)

    has_vectors = vertex_normals is not None
    Q_scalar = len(measures_list)
    Q_vec = 1 if has_vectors else 0
    Q = Q_scalar + Q_vec

    features = np.zeros((R, Q), dtype=np.float64)
    fnames: list[str] = []

    for ri in range(R):
        lm = left_masks[ri]
        rm = right_masks[ri]

        if int(lm.sum()) < min_vertices or int(rm.sum()) < min_vertices:
            fnames += [f"{region_names[ri]}_{m}" for m, _ in measures_list]
            if has_vectors:
                fnames += [
                    f"{region_names[ri]}_normal_vector_cos",
                ]
            continue

        # 标量测量：|mean(left) - mean(right)|
        for qi, (m_name, vals) in enumerate(measures_list):
            v = np.asarray(vals, dtype=np.float64)
            features[ri, qi] = abs(float(v[lm].mean() - v[rm].mean()))
            fnames.append(f"{region_names[ri]}_{m_name}")

        # 法向量衍生：左右侧法向量夹角 cos
        if has_vectors and vertex_normals is not None:
            nl = vertex_normals[lm].mean(axis=0)
            nr = vertex_normals[rm].mean(axis=0)
            nln = nl / max(np.linalg.norm(nl), 1e-12)
            nrn = nr / max(np.linalg.norm(nr), 1e-12)
            cos_a = float(np.clip(np.dot(nln, nrn), -1, 1))

            features[ri, Q_scalar] = cos_a
            fnames += [
                f"{region_names[ri]}_normal_vector_cos",
            ]

    return features, fnames


# ---------------------------------------------------------------------------
# PW — 逐顶点配对差
# ---------------------------------------------------------------------------


def compute_asymmetry_pw(
    scalar_measures: dict[str, np.ndarray],
    uv: np.ndarray,
    left_masks: list[np.ndarray],
    right_masks: list[np.ndarray],
    region_names: list[str],
    vertex_normals: np.ndarray | None = None,
    vertices: np.ndarray | None = None,
    min_vertices: int = 3,
    uv_tolerance: float = 0.05,
    normal_eps: float = 1e-8,
) -> tuple[np.ndarray, list[str]]:
    """计算 PW 模式逐顶点配对不对称特征。

    对每个区域：
    1. 将左顶点按 UV 镜射 (u -> -u, v -> v)
    2. 在右侧找 UV 最近邻配对
    3. |value_left - value_right| → 平均

    Args:
        scalar_measures: 同 ``compute_asymmetry_dm``。
        uv: (N, 2) UV 坐标。
        left_masks: R 个 (N,) bool 数组。
        right_masks: R 个 (N,) bool 数组。
        region_names: (R,) 区域名称。
        vertex_normals: (N, 3) 可选，用于法向量衍生指标。
        vertices: (N, 3) 可选，用于局部左右轴反射镜射。
        min_vertices: 单侧最少顶点数。
        uv_tolerance: 镜射 UV 与右 UV 的最近距离阈值，超过则丢弃该配对。
        normal_eps: 法向量反射时防零除。

    Returns:
        (features, feature_names):
            features: (R, Q) 矩阵。
            feature_names: 列名列表（含 ``__pw`` 后缀）。
    """
    measures_list = list(scalar_measures.items())
    R = len(region_names)

    has_vectors = vertex_normals is not None and vertices is not None
    Q_scalar = len(measures_list)
    Q_vec = 1 if has_vectors else 0
    Q = Q_scalar + Q_vec

    features = np.zeros((R, Q), dtype=np.float64)
    fnames: list[str] = []

    for ri in range(R):
        left_idx = np.where(left_masks[ri])[0]
        right_mask_arr = right_masks[ri]
        right_idx_set = set(np.where(right_mask_arr)[0])

        if len(left_idx) < min_vertices or len(right_idx_set) < min_vertices:
            fnames += [f"{region_names[ri]}_{m}__pw" for m, _ in measures_list]
            if has_vectors:
                fnames += [
                    f"{region_names[ri]}_normal_vector_cos__pw",
                ]
            continue

        uv_right = uv[right_mask_arr]
        right_order = np.where(right_mask_arr)[0]
        mirrored_u = -uv[left_idx, 0]
        mirrored_v = uv[left_idx, 1]

        # 预分配各测量的差数组
        diffs_scalar = {m: np.full(len(left_idx), np.nan, dtype=np.float64) for m, _ in measures_list}
        diffs_cos = np.full(len(left_idx), np.nan, dtype=np.float64) if has_vectors else None

        for pi, (lvi, mu, mv) in enumerate(zip(left_idx, mirrored_u, mirrored_v, strict=True)):
            d = np.hypot(uv_right[:, 0] - mu, uv_right[:, 1] - mv)
            best_idx = int(d.argmin())
            if d[best_idx] > uv_tolerance:
                continue
            rvi = right_order[best_idx]

            # 标量差
            for m_name, arr in measures_list:
                diffs_scalar[m_name][pi] = abs(float(arr[lvi]) - float(arr[rvi]))

            # 法向量夹角 cos（局部左右轴反射）
            if has_vectors and vertex_normals is not None and vertices is not None:
                n_left = vertex_normals[lvi]
                n_right = vertex_normals[rvi]
                local_axis = vertices[rvi] - vertices[lvi]
                local_norm = np.linalg.norm(local_axis)
                if local_norm > normal_eps:
                    local_axis = local_axis / local_norm
                    n_reflected = n_right - 2 * np.dot(n_right, local_axis) * local_axis
                else:
                    n_reflected = n_right
                cos_a = float(np.clip(np.dot(n_left, n_reflected), -1, 1))
                diffs_cos[pi] = cos_a

        # 平均差值（跳过 NaN）
        for qi, (m_name, _) in enumerate(measures_list):
            vals = diffs_scalar[m_name]
            valid = np.isfinite(vals) & (vals > 0)
            features[ri, qi] = float(vals[valid].mean()) if valid.any() else 0.0
            fnames.append(f"{region_names[ri]}_{m_name}__pw")

        if has_vectors and diffs_cos is not None:
            valid = np.isfinite(diffs_cos)
            features[ri, Q_scalar] = float(diffs_cos[valid].mean()) if valid.any() else 0.0
            fnames += [
                f"{region_names[ri]}_normal_vector_cos__pw",
            ]

    return features, fnames
