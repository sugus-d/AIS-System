"""Debug payload builder for axilla detection."""

from ..constants import Axilla


def build_side_debug(
    side_name: str,
    has_arm: bool,
    distance: float,
    nr_y: float,
    y_roi_lo: float,
    y_roi_hi: float,
    x_lo: float,
    x_hi: float,
    best_pt: list[float],
    best_cos: float,
    best_cwa: float,
    best_left_pt: list[float],
    best_right_pt: list[float],
    ldist: float,
    rdist: float,
    band_candidates: list[dict],
    *,
    derivative_x: list[float] | None = None,
    dydx: list[float],
    d2ydx2: list[float],
    cos_profile_points: list | None = None,
    cos_profile_values: list | None = None,
    cos_values: list[float],
    arm_boundary_x: float | None,
    lower_boundary: list,
    smoothed_clipped: list,
    selection_mode: str,
    n_lower_points: int,
    n_search_points: int,
    n_d2_neg: int,
) -> dict:
    """Build a single-side axilla debug dict.

    Returns the same structure as the original debug[side_name] dict.
    """
    # CCWA（Complementary ClockWise Angle）用于可视化时统一用 0~360 范围表示方向
    # CWA 可能 > 180，CCWA 换算成从外侧看的补角，方便目视判断凹角大小
    best_ccwa = (360.0 - best_cwa) % 360.0
    return {
        "has_arm": has_arm,
        "distance": distance,
        "band_lo": y_roi_lo,
        "band_hi": y_roi_hi,
        "x_lo": x_lo,
        "x_hi": x_hi,
        "nr_y": nr_y,
        "best_pt": best_pt,
        "best_cwa": best_cwa,
        "best_ccwa": best_ccwa,
        "best_angle_deg": best_cwa,
        "best_cos": best_cos,
        "angle_left": best_left_pt,
        "angle_right": best_right_pt,
        "angle_left_dist": ldist,
        "angle_right_dist": rdist,
        "band_candidates": band_candidates,
        "derivative_x": derivative_x if derivative_x is not None else [],
        "dydx": dydx,
        "d2ydx2": d2ydx2,
        "cos_profile_points": cos_profile_points if cos_profile_points is not None else [],
        "cos_profile_values": cos_profile_values if cos_profile_values is not None else [],
        "cos_values": cos_values,
        "arm_boundary_x": arm_boundary_x,
        "lower_boundary": lower_boundary,
        "smoothed_clipped": smoothed_clipped,
        "selection_mode": selection_mode,
        "n_lower_points": n_lower_points,
        "n_search_points": n_search_points,
        "n_d2_neg": n_d2_neg,
        "d2_threshold": Axilla.D2_NEG_THRESHOLD,
    }
