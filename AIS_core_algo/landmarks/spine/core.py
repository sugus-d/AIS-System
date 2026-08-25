"""脊柱点推导和中线拟合：基于谷底检测 + 两阶段过滤。"""

import numpy as np
from scipy.interpolate import UnivariateSpline

from ..constants import Spine

_ASYMMETRY_WINDOW_THRESHOLD = 20.0  # 左右点 Y 差超过该值（mm）用更宽采样窗口
_MIN_FIT_POINTS = 3                 # 二次曲线拟合最少点数
_CONCAVE_CURVATURE = 0.001          # 二次项系数大于该值判定为凹陷（U 形）
_CONVEX_CURVATURE = -0.001          # 二次项系数小于该值判定为凸起（∩ 形）


def fit_spine_midline(
    vertices: np.ndarray,
    mid_x: float,
    n_bins: int = Spine.N_BINS,
    poly_deg: int = Spine.POLY_DEG,
) -> np.ndarray:
    """通过 X-Z 剖面谷底检测 + 两阶段离群值过滤拟合脊柱中线。

    WHY: 逐 bin 谷底检测 + MAD+残差两步过滤，避免解剖凹陷干扰和离群值拉偏。
    先用 MAD 排除 X 坐标离群点（侧胸壁噪声），再用样条残差过滤剔除 Z 剖面异常点。

    Args:
        vertices: 网格顶点数组 (N, 3)。
        mid_x: 体中线 X 参考值，用于谷底选择。
        n_bins: Y 方向分箱数量。
        poly_deg: 样条拟合的多项式阶数。

    Returns:
        midline: (200, 3) 平滑中线。
    """
    y: np.ndarray = vertices[:, 1]
    y_min: float = float(y.min())
    y_max: float = float(y.max())
    x_range: float = float(vertices[:, 0].max() - vertices[:, 0].min())
    # WHY: 25%-75% X 范围排除肩部和侧胸壁干扰，仅保留躯干中央区域
    x_lo: float = float(vertices[:, 0].min()) + Spine.X_LO_RATIO * x_range
    x_hi: float = float(vertices[:, 0].min()) + Spine.X_HI_RATIO * x_range

    bin_edges: np.ndarray = np.linspace(y_min, y_max, n_bins + 1)
    mid_points: list[list[float]] = []

    for i in range(n_bins):
        mask: np.ndarray = (y >= bin_edges[i]) & (y < bin_edges[i + 1])
        if int(mask.sum()) < Spine.MIN_BIN_POINTS:
            continue
        bin_vertices: np.ndarray = vertices[mask]
        central_mask: np.ndarray = (bin_vertices[:, 0] >= x_lo) & (bin_vertices[:, 0] <= x_hi)
        if int(central_mask.sum()) < Spine.MIN_BIN_POINTS:
            continue
        central_vertices: np.ndarray = bin_vertices[central_mask]

        # 按 X 排序构建 Z(X) 剖面，滑动平均抑制噪声
        sort_idx: np.ndarray = np.argsort(central_vertices[:, 0])
        sorted_x: np.ndarray = central_vertices[sort_idx, 0]
        sorted_z: np.ndarray = central_vertices[sort_idx, 2]
        window: int = max(Spine.MIN_SMOOTHING_WINDOW, len(sorted_x) // Spine.WINDOW_DIVISOR)
        if window >= len(sorted_x):
            continue
        # 用滑动平均平滑 Z(X) 剖面以防止局部毛刺被误判为谷底
        smoothed_z: np.ndarray = np.convolve(sorted_z, np.ones(window) / window, mode="valid")
        offset: int = window // 2

        # 找局部最小值: Z 小于左右各 step 个邻居
        step: int = max(Spine.VALLEY_STEP_MIN, len(smoothed_z) // Spine.VALLEY_STEP_DIVISOR)
        if len(smoothed_z) <= 2 * step:
            continue
        valleys: list[tuple[int, float]] = []
        for j in range(step, len(smoothed_z) - step):
            if smoothed_z[j] < smoothed_z[j - step] and smoothed_z[j] < smoothed_z[j + step]:
                valley_x: float = float(sorted_x[j + offset])
                valleys.append((j + offset, valley_x))
        if not valleys:
            continue
        # 选择最接近体中线 mid_x 的谷底（脊柱沟应在体中线附近）
        best_offset: int
        best_offset, _ = min(valleys, key=lambda v: abs(v[1] - mid_x))
        candidate_y: float = (bin_edges[i] + bin_edges[i + 1]) / 2.0
        mid_points.append([float(sorted_x[best_offset]), candidate_y, float(sorted_z[best_offset])])

    if len(mid_points) < poly_deg + 2:
        return np.zeros((0, 3))

    pts: np.ndarray = np.array(mid_points)

    # Stage 1: MAD 过滤（基于 X 坐标）——排除因肩胛骨偏移导致的 X 离群点
    median_x: float = float(np.median(pts[:, 0]))
    mad: float = max(float(np.median(np.abs(pts[:, 0] - median_x))), Spine.MAD_FLOOR)
    rejected_mad: np.ndarray = np.zeros(len(pts), dtype=bool)
    pts_stage1: np.ndarray = pts
    if len(pts) >= poly_deg + 2:
        mad_mask: np.ndarray = np.abs(pts[:, 0] - median_x) <= Spine.MAD_THRESHOLD * mad
        if int(mad_mask.sum()) >= poly_deg + 2:
            rejected_mad = ~mad_mask
            pts_stage1 = pts[mad_mask]

    # Stage 2: 第一次样条拟合（平滑因子较大，容许一定偏差）
    # WHY: 先拟合 X(Y) 用于残差过滤，Z 拟合在 Stage 4 最终输出时才做。
    y_s1: np.ndarray = pts_stage1[:, 1]
    s1: float = len(pts_stage1) * Spine.STAGE1_S_FACTOR
    spl_x1 = UnivariateSpline(y_s1, pts_stage1[:, 0], k=poly_deg, s=s1)

    # Stage 3: 残差过滤——剔除与样条偏差过大的点
    residuals: np.ndarray = np.abs(pts_stage1[:, 0] - spl_x1(pts_stage1[:, 1]))
    sigma: float = float(np.std(residuals))
    rejected_residual: np.ndarray = np.zeros(len(pts), dtype=bool)
    pts_clean: np.ndarray = pts_stage1
    if sigma > Spine.SIGMA_EPSILON and len(pts_stage1) >= poly_deg + 2:
        residual_mask: np.ndarray = residuals <= Spine.RESIDUAL_SIGMA_THRESHOLD * sigma
        if int(residual_mask.sum()) >= poly_deg + 2:
            stage1_indices: np.ndarray = np.where(~rejected_mad)[0]
            rejected_residual[stage1_indices[~residual_mask]] = True
            pts_clean = pts_stage1[residual_mask]

    # Fallback: clean 点不足时跳过残差过滤，防止过度剔除
    clean_mask: np.ndarray = ~(rejected_mad | rejected_residual)
    n_clean: int = int(clean_mask.sum())
    if n_clean < Spine.CLEAN_POINT_LIMIT and int((~rejected_mad).sum()) >= Spine.CLEAN_POINT_LIMIT:
        pts_clean = pts_stage1
        rejected_residual = np.zeros(len(pts), dtype=bool)

    # Stage 4: 第二次拟合（离群值已剔除，平滑因子更小以贴合真实曲线）
    y_clean: np.ndarray = pts_clean[:, 1]
    spl_x2 = UnivariateSpline(y_clean, pts_clean[:, 0], k=poly_deg, s=len(pts_clean) * Spine.STAGE2_S_FACTOR)
    spl_z2 = UnivariateSpline(y_clean, pts_clean[:, 2], k=poly_deg, s=len(pts_clean) * Spine.STAGE2_S_FACTOR)

    # WHY: 限制到候选点 Y 范围防外推——样条在无数据区域的外推不可靠
    if len(pts_clean) >= poly_deg + 2 and float(pts_clean[:, 1].max() - pts_clean[:, 1].min()) >= Spine.Y_RANGE_MIN:
        y_smooth_min: float = float(pts_clean[:, 1].min())
        y_smooth_max: float = float(pts_clean[:, 1].max())
    else:
        y_smooth_min, y_smooth_max = y_min, y_max
    y_smooth: np.ndarray = np.linspace(y_smooth_min, y_smooth_max, Spine.MIDLINE_N_POINTS)
    return np.column_stack([spl_x2(y_smooth), y_smooth, spl_z2(y_smooth)])


def derive_spine_points(
    vertices: np.ndarray,
    bilateral_pairs: list[np.ndarray],
    curvature: np.ndarray | None = None,
) -> np.ndarray:
    """对每对 bilateral landmark，在横截面局部窗口检测凹凸形状来选脊柱点。

    WHY: 每对 landmark 独立处理，用横截面二次曲线拟合判断局部
    U 形（凹陷）/ ∩ 形（凸起）/ 平坦，据此选谷底、峰顶或中点。
    不再调用 fit_spine_midline，避免下段离群点拉偏样条。

    Args:
        vertices: 网格顶点数组 (N, 3)。
        bilateral_pairs: 若干 (2,3) 的左右成对点列表。
        curvature: 平均曲率数组 (N,)，用于辅助选择棘突。

    Returns:
        (K, 3) 的脊柱点数组，与 bilateral_pairs 一一对应。
    """
    spine_pts: list[np.ndarray] = []
    n_pairs: int = len(bilateral_pairs)

    for i, pair in enumerate(bilateral_pairs):
        left_pt: np.ndarray = pair[0]
        right_pt: np.ndarray = pair[1]

        # -- Y 窗口采样：在左右点 Y 中点附近取局部带状区域
        y_center: float = (float(left_pt[1]) + float(right_pt[1])) / 2.0
        y_diff: float = abs(float(left_pt[1]) - float(right_pt[1]))
        y_margin: float = 15.0 if y_diff > _ASYMMETRY_WINDOW_THRESHOLD else 10.0

        x_lo: float = float(left_pt[0])
        x_hi: float = float(right_pt[0])
        if x_lo > x_hi:
            x_lo, x_hi = x_hi, x_lo

        mask: np.ndarray = (
            (vertices[:, 1] >= y_center - y_margin)
            & (vertices[:, 1] <= y_center + y_margin)
            & (vertices[:, 0] >= x_lo)
            & (vertices[:, 0] <= x_hi)
        )
        candidate_verts: np.ndarray = vertices[mask]
        if len(candidate_verts) == 0:
            spine_pts.append((left_pt + right_pt) / 2.0)
            continue

        # -- 局部窗口形状检测：用二次曲线拟合判断凹陷/凸起/平坦
        mid_x: float = (x_lo + x_hi) / 2.0
        span: float = x_hi - x_lo
        local_half: float = span * 0.15

        local_mask: np.ndarray = (candidate_verts[:, 0] >= mid_x - local_half) & (
            candidate_verts[:, 0] <= mid_x + local_half
        )
        local_verts: np.ndarray = candidate_verts[local_mask]

        if len(local_verts) < _MIN_FIT_POINTS:
            # 点数不足拟合 → 取最接近中线的顶点
            idx = int(np.argmin(np.abs(candidate_verts[:, 0] - mid_x)))
            spine_pts.append(candidate_verts[idx])
            continue

        coeffs: np.ndarray = np.polyfit(local_verts[:, 0], local_verts[:, 2], 2)
        a: float = float(coeffs[0])

        # 曲率查找：将 local_verts 映射回原始顶点索引
        curv_vals: np.ndarray | None = None
        if curvature is not None:
            cand_idx_arr: np.ndarray = np.where(mask)[0]
            local_orig_idx: np.ndarray = cand_idx_arr[local_mask]
            curv_vals = curvature[local_orig_idx]

        if a > _CONCAVE_CURVATURE:
            # 局部凹陷（U 形）：倾向 Z 最小 + 曲率最负（最深沟）
            if i == n_pairs - 1:
                # WHY: 最后一对（waist）使用加权组合，强调 X 邻近中线
                local_z: np.ndarray = local_verts[:, 2]
                local_x: np.ndarray = local_verts[:, 0]
                prox: np.ndarray = np.abs(local_x - mid_x) / max(local_half, 1e-6)
                if curv_vals is not None:
                    c_range: float = max(float(np.abs(curv_vals).max()), 1e-6)
                    c_score: np.ndarray = (curv_vals - curv_vals.min()) / c_range
                    combined: np.ndarray = local_z * 0.15 + prox * 0.70 + c_score * 0.15
                else:
                    combined = local_z * 0.3 + prox * 0.7
                best_idx = int(np.argmin(combined))
            else:
                if curv_vals is not None:
                    # 非最后一对：组合评分（曲率 + 连续 + 居中）
                    c_norm: np.ndarray = (curv_vals - curv_vals.min()) / max(
                        float(curv_vals.max() - curv_vals.min()), 1e-6
                    )
                    prev_x: float = float(spine_pts[-1][0]) if i > 0 and len(spine_pts) >= i else mid_x
                    cont_score: np.ndarray = 1.0 - np.abs(local_verts[:, 0] - prev_x) / max(span, 1e-6)
                    center_score: np.ndarray = 1.0 - np.abs(local_verts[:, 0] - mid_x) / max(local_half, 1e-6)
                    scores: np.ndarray = c_norm * 0.5 + cont_score * 0.3 + center_score * 0.2
                    best_idx = int(np.argmax(scores))
                else:
                    best_idx = int(np.argmin(local_verts[:, 2]))
            spine_pts.append(local_verts[best_idx])
        elif a < _CONVEX_CURVATURE:
            # 局部凸起（∩ 形）：倾向 Z 最大 + 曲率最正（最凸脊）
            if curv_vals is not None:
                c_norm = (curv_vals - curv_vals.min()) / max(float(curv_vals.max() - curv_vals.min()), 1e-6)
                prev_x = float(spine_pts[-1][0]) if i > 0 and len(spine_pts) >= i else mid_x
                cont_score = 1.0 - np.abs(local_verts[:, 0] - prev_x) / max(span, 1e-6)
                scores = c_norm * 0.7 + cont_score * 0.3
                best_idx = int(np.argmax(scores))
            else:
                best_idx = int(np.argmax(local_verts[:, 2]))
            spine_pts.append(local_verts[best_idx])
        else:
            # 平坦：取 X 最接近中线的点
            idx = int(np.argmin(np.abs(local_verts[:, 0] - mid_x)))
            spine_pts.append(local_verts[idx])

    return np.array(spine_pts)
