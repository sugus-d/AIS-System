"""Pipeline coordination steps for axilla detection.

Contains derivative-based analysis path and limited-points contour search.
"""

import numpy as np

from landmarks.angle import compute_lateral_angle_at_point
from landmarks.contour import extract_lower_boundary_per_integer_x
from landmarks.signal_ops import compute_derivatives_from_xy, find_flat_region_x, smooth_contour
from utils.logger import logger

from ..constants import Axilla
from ._search import _detect_few_points, _search_d2_candidates, _search_fallback_candidates


def _detect_single_side(
    contour: np.ndarray,
    nr_pt: np.ndarray,
    has_arm: bool,
    side_name: str,
    x_lo: float,
    x_hi: float,
    y_roi_lo: float,
    y_roi_hi: float,
    distance: float,
    y_range: float,
    y_ref: float,
) -> tuple[np.ndarray, dict]:
    """单侧腋窝检测：预处理 → 导数分析 → 候选选择。

    Returns:
        (best_pt, side_meta)：best_pt 为腋窝点；side_meta 仅含功能字段
        （has_arm / arm_boundary_x / smoothed_clipped），供肩转检测与单臂矫正使用。
    """
    best_pt_init = np.array([float(nr_pt[0]), float(nr_pt[1]) - y_range * 0.30])

    # === Step 1: Contour Preprocessing ===
    smooth = smooth_contour(contour)
    # 下边界多往下扩展一个 padding 以保留腋窝底部过渡区域
    # 上边界直接卡在 y_roi_hi（颈根下方），不过度向上浪费搜索范围
    clip_y_lo = y_roi_lo - Axilla.CLIP_Y_PAD_RATIO * y_range
    clip_y_hi = y_roi_hi
    x_range_width = abs(x_hi - x_lo)
    # 外侧方向（left→向左，right→向右）额外扩展搜索宽度
    # 因为手臂外扩时轮廓外侧可能超出颈腰中线限定的边界
    if side_name == "left":
        clip_x_lo = x_lo - Axilla.CLIP_X_PAD_RATIO * x_range_width
        clip_x_hi = x_hi
    else:
        clip_x_lo = x_lo
        clip_x_hi = x_hi + Axilla.CLIP_X_PAD_RATIO * x_range_width

    mask = (
        (smooth[:, 1] >= clip_y_lo)
        & (smooth[:, 1] <= clip_y_hi)
        & (smooth[:, 0] >= clip_x_lo)
        & (smooth[:, 0] <= clip_x_hi)
    )
    clipped = smooth[mask]
    lower_sv = extract_lower_boundary_per_integer_x(clipped)

    new_dbg: dict = {
        "arm_boundary_x": None,
        "smoothed_clipped": clipped.tolist(),
    }
    best_pt = best_pt_init

    # Dispatch to branch handlers based on point count
    # 点越多精度越高：极少点→原始轮廓搜索，有限点→CCWA/无手臂横向搜索，
    # 足够点→完整导数分析（二阶导筛选凹陷+夹角评分）
    if len(lower_sv) < Axilla.MIN_POINTS_FALLBACK:
        best_pt = _detect_few_points(
            clipped,
            smooth,
            side_name,
            distance,
            has_arm,
            best_pt_init,
            y_ref,
            y_roi_lo,
            y_roi_hi,
        )
    elif len(lower_sv) < Axilla.MIN_POINTS_DERIV:
        best_pt = _detect_limited_points(
            contour,
            side_name,
            x_lo,
            x_hi,
            y_roi_lo,
            y_roi_hi,
            has_arm,
            distance,
            best_pt_init,
            y_ref,
        )
    else:
        best_pt = _detect_derivative_path(
            lower_sv,
            smooth,
            side_name,
            has_arm,
            x_lo,
            x_hi,
            distance,
            new_dbg,
            y_ref,
            y_roi_lo,
            y_roi_hi,
        )

    logger.info(f"[AXILLA] {side_name}: best_pt=({best_pt[0]:.1f}, {best_pt[1]:.1f})")

    side_meta = {
        "has_arm": has_arm,
        "arm_boundary_x": new_dbg["arm_boundary_x"],
        "smoothed_clipped": new_dbg.get("smoothed_clipped", []),
    }
    return best_pt, side_meta


def _detect_limited_points(
    contour: np.ndarray,
    side_name: str,
    x_lo: float,
    x_hi: float,
    y_roi_lo: float,
    y_roi_hi: float,
    has_arm: bool,
    distance: float,
    initial_best_pt: np.ndarray,
    y_ref: float,
) -> np.ndarray:
    """Handle len(lower_sv) < Axilla.MIN_POINTS_DERIV: CCWA or lateral on raw contour."""
    best_score = float("inf") if has_arm else (float("inf") if side_name == "left" else float("-inf"))
    best_pt = initial_best_pt.copy()
    no_arm_candidates: list[np.ndarray] = []
    for k in range(len(contour)):
        y = float(contour[k, 1])
        if y < y_roi_lo or y > y_roi_hi:
            continue
        cand = contour[k].copy()
        if not (x_lo <= float(cand[0]) <= x_hi):
            continue
        _, angle_deg, _, _, _, _ = compute_lateral_angle_at_point(contour, cand, distance=distance)
        if has_arm:
            if Axilla.CW_ANGLE_MIN < angle_deg < Axilla.CW_ANGLE_MAX and angle_deg < best_score:
                best_score = angle_deg
                best_pt = cand.copy()
        else:
            no_arm_candidates.append(cand)

    if not has_arm and len(no_arm_candidates) > 0:
        margin = Axilla.LATERAL_MARGIN
        # 左/右侧分别按最小/最大 effective_x 择优
        best_score = float("inf") if side_name == "left" else float("-inf")
        best_pt = no_arm_candidates[0].copy()
        for pt_arr in no_arm_candidates:
            x_val = float(pt_arr[0])
            y_val = float(pt_arr[1])
            # y_delta 为 y_ref - y_val（正值表示候选点低于参考）
            # 无手臂时没有 V 形凹陷，外侧轮廓会平滑向下延伸
            # 如果不加 Y 高度惩罚，算法可能选到腰部附近的错误低点
            y_delta = max(0.0, y_ref - y_val)
            y_penalty = Axilla.NOARM_Y_WEIGHT * y_delta
            # 限制单点最大惩罚，防止极端低位点被错误优先
            y_penalty = min(y_penalty, getattr(Axilla, "NOARM_MAX_PENALTY", 30.0))
            # 如果候选点比参考低超过允许的最大下沉，跳过该候选
            if y_delta > getattr(Axilla, "NOARM_MAX_DROP", 80.0):
                continue
            if side_name == "left":
                # 对左侧：有效 X 越左越好（腋窝在体外侧），加惩罚使之向右偏移
                effective_x = x_val + y_penalty
                if effective_x < best_score - margin:
                    best_score = effective_x
                    best_pt = pt_arr.copy()
                elif abs(effective_x - best_score) <= margin and y_val < best_pt[1]:
                    best_pt = pt_arr.copy()
            else:
                # 对右侧：有效 X 越右越好，减惩罚使之向左偏移
                effective_x = x_val - y_penalty
                if effective_x > best_score + margin:
                    best_score = effective_x
                    best_pt = pt_arr.copy()
                elif abs(effective_x - best_score) <= margin and y_val < best_pt[1]:
                    best_pt = pt_arr.copy()
    return best_pt


def _detect_derivative_path(
    lower_sv: np.ndarray,
    smooth: np.ndarray,
    side_name: str,
    has_arm: bool,
    x_lo: float,
    x_hi: float,
    distance: float,
    new_dbg: dict,
    y_ref: float,
    y_roi_lo: float,
    y_roi_hi: float,
) -> np.ndarray:
    """Derivative analysis path: derivatives → arm boundary → candidate search."""
    derivative_x = lower_sv[:, 0]
    # 一阶导定位手臂外侧的剧烈斜率变化，二阶导检测 V 形凹陷的曲率负值
    # 两者联合使用才能区分腋窝凹陷和其他轮廓抖动
    dydx = compute_derivatives_from_xy(derivative_x, lower_sv[:, 1], derv_order=1)
    d2ydx2 = compute_derivatives_from_xy(derivative_x, lower_sv[:, 1], derv_order=2)

    # Step 2: Find arm boundary — 从平坦区末端定位手臂外侧边界 X
    ab_x = find_flat_region_x(lower_sv, dydx, d2ydx2, side_name)

    if ab_x is not None:
        if side_name == "left" and ab_x < x_hi:
            x_lo = ab_x
        elif side_name == "right" and ab_x > x_lo:
            x_hi = ab_x
        logger.info(f"[AXILLA] {side_name}: arm_boundary_x={ab_x:.1f}, search_region=({x_lo:.1f}, {x_hi:.1f})")
    new_dbg["arm_boundary_x"] = ab_x

    # Step 3: d<0 candidates → max cosine
    # 腋窝 V 形凹陷处二阶导显著为负（< D2_NEG_THRESHOLD）
    # 同时一阶导不能太大（|dydx| < DYDX_ARM_MAX），排除手臂外侧的陡峭斜坡
    search_mask = (lower_sv[:, 0] >= x_lo) & (lower_sv[:, 0] <= x_hi)
    search_pts = lower_sv[search_mask]
    search_d2 = d2ydx2[search_mask]
    search_dydx = dydx[search_mask]

    if len(search_pts) == 0:
        return np.array([0.0, 0.0])

    neg_mask = (search_d2 < Axilla.D2_NEG_THRESHOLD) & (np.abs(search_dydx) < Axilla.DYDX_ARM_MAX)

    if neg_mask.sum() > 0:
        # 有符合条件的二阶导负值候选 → 用余弦夹角（max cosine）从凹陷中选最优
        return _search_d2_candidates(
            search_pts,
            neg_mask,
            smooth,
            distance,
            x_lo,
            x_hi,
        )
    else:
        # 无二阶导负值候选 → 退回到夹角筛选（有手臂用 CCWA，无手臂用 Y 锚点横向搜索）
        return _search_fallback_candidates(
            search_pts,
            smooth,
            side_name,
            has_arm,
            distance,
            x_lo,
            x_hi,
            y_ref,
            y_roi_lo,
            y_roi_hi,
        )
