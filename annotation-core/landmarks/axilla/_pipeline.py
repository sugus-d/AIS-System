"""Pipeline coordination steps for axilla detection.

Contains derivative-based analysis path and limited-points contour search.
"""

import numpy as np

from utils.angle import compute_lateral_angle_at_point, compute_lateral_angle_profile
from utils.contour import extract_lower_boundary_per_integer_x
from utils.logger import logger
from utils.signal_ops import compute_derivatives_from_xy, find_flat_region_x, smooth_contour

from ..constants import Axilla
from ._search import _detect_few_points, _search_d2_candidates, _search_fallback_candidates
from .debug import build_side_debug


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
    nr_y: float,
    y_range: float,
    y_ref: float,
) -> tuple[np.ndarray, dict]:
    """单侧腋窝检测：预处理 → 导数分析 → 候选选择 → debug 组装。"""
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
        "dydx": [],
        "d2ydx2": [],
        "cos_values": [],
        "arm_boundary_x": None,
        "lower_boundary": lower_sv.tolist(),
        "smoothed_clipped": clipped.tolist(),
        "selection_mode": "unknown",
        "n_lower_points": int(len(lower_sv)),
        "n_search_points": 0,
        "n_d2_neg": 0,
        "d2_threshold": Axilla.D2_NEG_THRESHOLD,
    }
    band_candidates: list[dict] = []
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
            new_dbg,
            band_candidates,
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
            new_dbg,
            band_candidates,
            best_pt_init,
            y_ref,
        )
    else:
        best_pt, x_lo, x_hi = _detect_derivative_path(
            lower_sv,
            smooth,
            side_name,
            has_arm,
            x_lo,
            x_hi,
            distance,
            new_dbg,
            band_candidates,
            y_ref,
            y_roi_lo,
            y_roi_hi,
        )

    # 在最佳候选点上重新计算夹角（cos/cwa），两个分支用不同的度量标准筛选
    # 但最终输出需要统一的夹角值供调试和可视化使用，不能直接用分支中的中间变量
    best_cos, best_cwa, best_left_pt, best_right_pt, ldist, rdist = compute_lateral_angle_at_point(
        smooth, best_pt, distance=distance
    )

    logger.info(
        f"[AXILLA] {side_name}: best_pt=({best_pt[0]:.1f}, {best_pt[1]:.1f}), cos={best_cos:.3f}, cwa={best_cwa:.1f}deg"
    )

    side_debug = build_side_debug(
        side_name=side_name,
        has_arm=has_arm,
        distance=distance,
        nr_y=nr_y,
        y_roi_lo=y_roi_lo,
        y_roi_hi=y_roi_hi,
        x_lo=x_lo,
        x_hi=x_hi,
        best_pt=[float(best_pt[0]), float(best_pt[1])],
        best_cos=best_cos,
        best_cwa=best_cwa,
        best_left_pt=[float(best_left_pt[0]), float(best_left_pt[1])],
        best_right_pt=[float(best_right_pt[0]), float(best_right_pt[1])],
        ldist=ldist,
        rdist=rdist,
        band_candidates=band_candidates,
        derivative_x=new_dbg.get("derivative_x"),
        dydx=new_dbg["dydx"],
        d2ydx2=new_dbg["d2ydx2"],
        cos_profile_points=new_dbg.get("cos_profile_points"),
        cos_profile_values=new_dbg.get("cos_profile_values"),
        cos_values=new_dbg["cos_values"],
        arm_boundary_x=new_dbg["arm_boundary_x"],
        lower_boundary=new_dbg["lower_boundary"],
        smoothed_clipped=new_dbg.get("smoothed_clipped", []),
        selection_mode=new_dbg["selection_mode"],
        n_lower_points=new_dbg["n_lower_points"],
        n_search_points=new_dbg["n_search_points"],
        n_d2_neg=new_dbg["n_d2_neg"],
    )
    return best_pt, side_debug


def _detect_limited_points(
    contour: np.ndarray,
    side_name: str,
    x_lo: float,
    x_hi: float,
    y_roi_lo: float,
    y_roi_hi: float,
    has_arm: bool,
    distance: float,
    new_dbg: dict,
    band_candidates: list[dict],
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
        cos_c, angle_deg, lp, rp, ld, rd = compute_lateral_angle_at_point(contour, cand, distance=distance)
        band_candidates.append(
            {
                "x": float(cand[0]),
                "y": float(cand[1]),
                "cwa": angle_deg,
                "ccwa": angle_deg,
                "angle_deg": angle_deg,
                "cos": cos_c,
                "left_pt": [float(lp[0]), float(lp[1])],
                "right_pt": [float(rp[0]), float(rp[1])],
            }
        )
        if has_arm:
            if Axilla.CW_ANGLE_MIN < angle_deg < Axilla.CW_ANGLE_MAX and angle_deg < best_score:
                best_score = angle_deg
                best_pt = cand.copy()
        else:
            no_arm_candidates.append(cand)

    if not has_arm and len(no_arm_candidates) > 0:
        new_dbg["selection_mode"] = "no_arm_y_anchor_limited"
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
    elif has_arm:
        new_dbg["selection_mode"] = "contour_ccwa"
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
    band_candidates: list[dict],
    y_ref: float,
    y_roi_lo: float,
    y_roi_hi: float,
) -> tuple[np.ndarray, float, float]:
    """Derivative analysis path: derivatives → arm boundary → candidate search."""
    derivative_x = lower_sv[:, 0]
    # 一阶导定位手臂外侧的剧烈斜率变化，二阶导检测 V 形凹陷的曲率负值
    # 两者联合使用才能区分腋窝凹陷和其他轮廓抖动
    dydx = compute_derivatives_from_xy(derivative_x, lower_sv[:, 1], derv_order=1)
    d2ydx2 = compute_derivatives_from_xy(derivative_x, lower_sv[:, 1], derv_order=2)

    new_dbg["derivative_x"] = derivative_x.tolist()
    new_dbg["dydx"] = dydx.tolist()
    new_dbg["d2ydx2"] = d2ydx2.tolist()

    # Step 2: Find arm boundary — 从平坦区末端定位手臂外侧边界 X
    ab_x = find_flat_region_x(lower_sv, dydx, d2ydx2, side_name)

    if ab_x is not None:
        if side_name == "left" and ab_x < x_hi:
            x_lo = ab_x
        elif side_name == "right" and ab_x > x_lo:
            x_hi = ab_x
        logger.info(f"[AXILLA] {side_name}: arm_boundary_x={ab_x:.1f}, search_region=({x_lo:.1f}, {x_hi:.1f})")
    new_dbg["arm_boundary_x"] = ab_x

    # Step 2.5: Cosine profile for visualization
    cos_pts, cos_vals = compute_lateral_angle_profile(lower_sv, distance=distance)
    new_dbg["cos_profile_points"] = cos_pts.tolist()
    new_dbg["cos_profile_values"] = cos_vals.tolist()

    # Step 3: d<0 candidates → max cosine
    # 腋窝 V 形凹陷处二阶导显著为负（< D2_NEG_THRESHOLD）
    # 同时一阶导不能太大（|dydx| < DYDX_ARM_MAX），排除手臂外侧的陡峭斜坡
    search_mask = (lower_sv[:, 0] >= x_lo) & (lower_sv[:, 0] <= x_hi)
    search_pts = lower_sv[search_mask]
    search_d2 = d2ydx2[search_mask]
    search_dydx = dydx[search_mask]
    new_dbg["n_search_points"] = int(len(search_pts))
    band_candidates.clear()

    if len(search_pts) == 0:
        new_dbg["selection_mode"] = "empty_search_region"
        best_pt = np.array([0.0, 0.0])
        return best_pt, x_lo, x_hi

    neg_mask = (search_d2 < Axilla.D2_NEG_THRESHOLD) & (np.abs(search_dydx) < Axilla.DYDX_ARM_MAX)
    new_dbg["n_d2_neg"] = int(neg_mask.sum())

    if neg_mask.sum() > 0:
        # 有符合条件的二阶导负值候选 → 用余弦夹角（max cosine）从凹陷中选最优
        return _search_d2_candidates(
            search_pts,
            search_d2,
            neg_mask,
            smooth,
            side_name,
            distance,
            new_dbg,
            band_candidates,
            x_lo,
            x_hi,
        )
    else:
        # 无二阶导负值候选 → 退回到夹角筛选（有手臂用 CCWA，无手臂用 Y 锚点横向搜索）
        return _search_fallback_candidates(
            search_pts,
            search_d2,
            search_dydx,
            smooth,
            side_name,
            has_arm,
            distance,
            new_dbg,
            band_candidates,
            x_lo,
            x_hi,
            y_ref,
            y_roi_lo,
            y_roi_hi,
        )
