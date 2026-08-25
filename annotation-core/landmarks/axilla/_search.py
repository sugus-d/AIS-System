"""Candidate search strategies for axilla detection.

Contains d<0 candidate selection, CCWA/lateral fallback, and few-point bailout.
"""

import numpy as np

from utils.angle import compute_lateral_angle_at_point
from utils.logger import logger

from ..constants import Axilla

_MIN_X_SPAN_MM = 2.0  # 轮廓 X 跨度低于该值视为近垂直（直接选最低点）


def _detect_few_points(
    clipped: np.ndarray,
    smooth: np.ndarray,
    side_name: str,
    distance: float,
    has_arm: bool,
    new_dbg: dict,
    band_candidates: list[dict],
    fallback_best_pt: np.ndarray,
    y_ref: float = 0.0,
    y_roi_lo: float = 0.0,
    y_roi_hi: float = 0.0,
) -> np.ndarray:
    """Handle len(lower_sv) < MIN_POINTS_FALLBACK: contour lateral or no-data."""
    if not has_arm and len(clipped) > 0:
        new_dbg["selection_mode"] = "no_arm_y_anchor_few"
        for k in range(len(clipped)):
            pt = clipped[k]
            cos_val, angle_deg, lp, rp, ld, rd = (
                compute_lateral_angle_at_point(smooth, pt, distance=distance)
            )
            band_candidates.append({
                "x": float(pt[0]),
                "y": float(pt[1]),
                "cwa": angle_deg,
                "ccwa": angle_deg,
                "angle_deg": angle_deg,
                "cos": cos_val,
                "left_pt": [float(lp[0]), float(lp[1])],
                "right_pt": [float(rp[0]), float(rp[1])],
            })
        # Y-anchored: effective_X = X penalized for being below y_ref
        y_mask = (clipped[:, 1] > y_roi_lo) & (clipped[:, 1] < y_roi_hi)
        if y_mask.sum() == 0:
            y_mask = np.ones(len(clipped), dtype=bool)
        in_range = clipped[y_mask]
        # 如果轮廓 X 跨度极小（近乎垂直），直接选最低点
        x_span = float(in_range[:, 0].max() - in_range[:, 0].min())
        if x_span < _MIN_X_SPAN_MM:
            # 近垂直轮廓 → 选最低点作为 axilla
            best_idx = int(np.argmin(in_range[:, 1]))  # 最小 Y = 最低
            return in_range[best_idx].copy()
        if side_name == "left":
            effective = np.array([
                float(pt[0]) + Axilla.NOARM_Y_WEIGHT * max(0.0, y_ref - float(pt[1]))
                for pt in in_range
            ])
            best_idx = int(np.argmin(effective))
        else:
            effective = np.array([
                float(pt[0]) - Axilla.NOARM_Y_WEIGHT * max(0.0, y_ref - float(pt[1]))
                for pt in in_range
            ])
            best_idx = int(np.argmax(effective))
        return in_range[best_idx].copy()
    else:
        new_dbg["selection_mode"] = "no_data_fallback"
        band_candidates.clear()
        return fallback_best_pt


def _search_d2_candidates(
    search_pts: np.ndarray,
    search_d2: np.ndarray,
    neg_mask: np.ndarray,
    smooth: np.ndarray,
    side_name: str,
    distance: float,
    new_dbg: dict,
    band_candidates: list[dict],
    x_lo: float,
    x_hi: float,
) -> tuple[np.ndarray, float, float]:
    """d<0 candidates exist → select by max cosine."""
    new_dbg["selection_mode"] = "d2_neg_max_cos"

    best_cos = -1.0
    best_pt = search_pts[0].copy()
    cos_vals_list: list[float] = []
    for i in range(len(search_pts)):
        pt = search_pts[i]
        is_cand = bool(neg_mask[i])
        cos_val, _, _, _, _, _ = compute_lateral_angle_at_point(
            smooth, pt, distance=distance
        )
        band_candidates.append({
            "x": float(pt[0]),
            "y": float(pt[1]),
            "cos": cos_val,
            "d2ydx2": float(search_d2[i]),
            "is_candidate": is_cand,
        })
        if is_cand:
            cos_vals_list.append(cos_val)
            if cos_val > best_cos:
                best_cos = cos_val
                best_pt = pt.copy()

    new_dbg["cos_values"] = cos_vals_list
    return best_pt, x_lo, x_hi


def _search_fallback_candidates(
    search_pts: np.ndarray,
    search_d2: np.ndarray,
    search_dydx: np.ndarray,
    smooth: np.ndarray,
    side_name: str,
    has_arm: bool,
    distance: float,
    new_dbg: dict,
    band_candidates: list[dict],
    x_lo: float,
    x_hi: float,
    y_ref: float = 0.0,
    y_roi_lo: float = 0.0,
    y_roi_hi: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    """No d<0 candidates → CCWA (arm) or Y-anchored lateral (armless) fallback."""
    new_dbg["selection_mode"] = (
        "ccwa_fallback" if has_arm else "no_arm_y_anchor_fallback"
    )

    best_score = (
        float("inf")
        if has_arm
        else (float("inf") if side_name == "left" else float("-inf"))
    )
    best_pt = search_pts[0].copy()
    no_arm_candidates: list[np.ndarray] = []
    ccwa_valid_count = 0
    for i in range(len(search_pts)):
        pt = search_pts[i]
        cos_val, angle_deg, lp, rp, ld, rd = compute_lateral_angle_at_point(
            smooth, pt, distance=distance
        )
        band_candidates.append({
            "x": float(pt[0]),
            "y": float(pt[1]),
            "cwa": angle_deg,
            "ccwa": angle_deg,
            "angle_deg": angle_deg,
            "cos": cos_val,
            "left_pt": [float(lp[0]), float(lp[1])],
            "right_pt": [float(rp[0]), float(rp[1])],
            "d2ydx2": float(search_d2[i]),
            "is_candidate": False,
        })
        if has_arm:
            if Axilla.CW_ANGLE_MIN < angle_deg < Axilla.CW_ANGLE_MAX and angle_deg < best_score:
                best_score = angle_deg
                best_pt = pt.copy()
                ccwa_valid_count += 1
        else:
            no_arm_candidates.append(pt.copy())

    # has_arm=True but CCWA filtering yielded zero valid candidates → fall through to no-arm lateral y-anchor
    if has_arm and ccwa_valid_count == 0:
        logger.info(
            f"[AXILLA] {side_name}: CCWA fallback → no-arm y-anchor "
            f"(searched {len(search_pts)} pts, none in CW_ANGLE range)"
        )
        new_dbg["selection_mode"] = "ccwa_to_noarm_y_anchor"
        for i in range(len(search_pts)):
            no_arm_candidates.append(search_pts[i].copy())
    if len(no_arm_candidates) > 0:
        cand_pts = np.array(no_arm_candidates)
        y_mask = (cand_pts[:, 1] > y_roi_lo) & (cand_pts[:, 1] < y_roi_hi)
        if y_mask.sum() > 0:
            cand_pts = cand_pts[y_mask]
        margin = Axilla.LATERAL_MARGIN
        # 左/右侧分别按最小/最大 effective_x 择优
        best_score = float("inf") if side_name == "left" else float("-inf")
        best_pt = cand_pts[0].copy()
        for j in range(len(cand_pts)):
            pt_arr = cand_pts[j]
            x_val = float(pt_arr[0])
            y_val = float(pt_arr[1])
            # y_delta 为 y_ref - y_val（正值表示候选点低于参考）
            y_delta = max(0.0, y_ref - y_val)
            # 原始 y_penalty
            y_penalty = Axilla.NOARM_Y_WEIGHT * y_delta
            # 限制单点最大惩罚，防止极端低位点被错误优先
            y_penalty = min(y_penalty, getattr(Axilla, 'NOARM_MAX_PENALTY', 30.0))
            # 如果候选点比参考低超过允许的最大下沉，跳过该候选
            if y_delta > getattr(Axilla, 'NOARM_MAX_DROP', 80.0):
                continue
            if side_name == "left":
                effective_x = x_val + y_penalty
                if effective_x < best_score - margin:
                    best_score = effective_x
                    best_pt = pt_arr.copy()
                elif abs(effective_x - best_score) <= margin and y_val < best_pt[1]:
                    best_pt = pt_arr.copy()
            else:
                effective_x = x_val - y_penalty
                if effective_x > best_score + margin:
                    best_score = effective_x
                    best_pt = pt_arr.copy()
                elif abs(effective_x - best_score) <= margin and y_val < best_pt[1]:
                    best_pt = pt_arr.copy()
    return best_pt, x_lo, x_hi
