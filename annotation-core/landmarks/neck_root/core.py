"""Neck root detection via histogram-mode width + per-candidate derivative + long-axis angle.

Use histogram mode bin as anchor, filter candidates by first-derivative threshold,
select by long-axis angle closest to 45° (within 20°~70°) with full fallback chain.
"""

import numpy as np

from utils.angle import compute_lateral_angle_at_point
from utils.contour import (
    extract_longest_contiguous_segment_in_box,
    search_segment_indices,
)
from utils.logger import logger
from utils.profile import build_width_profile_lines
from utils.signal_ops import (
    compute_derivatives_from_xy,
    normalize_xy,
    select_points_by_derivative,
    smooth_contour,
)

from .._validate_utils import long_axis_angle
from ..constants import AngleCandidate, NeckRoot
from .debug import build_neck_root_debug
from .hist import compute_histogram_mode_width


def detect_neck_root_strips(
    waist_points: np.ndarray,
    left_contour: np.ndarray,
    right_contour: np.ndarray,
    is_debug: bool = NeckRoot.DEBUG_MODE,
) -> tuple[np.ndarray, dict]:
    """按宽度模态、导数阈值和侧向角度检测颈根。

    Args:
        waist_points: 腰部左右点。
        left_contour: 左侧轮廓。
        right_contour: 右侧轮廓。


    Returns:
        tuple[np.ndarray, dict]: 颈根点和调试字典。
    """
    logger.info(
        f"waist_points=({waist_points[0, 0]:.1f},{waist_points[1, 0]:.1f}) "
        f"contour_left_len={len(left_contour)} contour_right_len={len(right_contour)}"
    )

    # 1. 预处理：先裁剪轮廓的上半部分再平滑
    # WHY: 颈根必然位于躯干上半段，裁剪掉下半段能减少后续宽度剖面计算中的干扰；
    # 平滑则消除局部锯齿，使导数计算更稳定。
    waist_w = float(waist_points[1, 0] - waist_points[0, 0])
    left_contour, right_contour = _clip_and_smooth_top_contours(left_contour, right_contour, waist_points)

    # 2-4. 宽度剖面 → 直方图模态 → 候选段索引
    # WHY：颈根在解剖上是脖子变宽的起始位置，用宽度直方图的众数 bin 作为
    # 颈宽典型值 W_mode，再从 W_mode 向下搜索宽度 ≥ 1.5×W_mode 的行作为下界，
    # 上下界之间的轮廓段就是颈根候选搜索空间。
    left_seg_idxs, right_seg_idxs, mode_idx, bin_info, mode_width = _compute_candidate_segments(
        left_contour, right_contour, waist_w
    )
    # 5. 在候选段上计算一阶导数（直接在当前段点上计算，不再额外做一次段内平滑）
    # WHY：颈根所在位置是脖子窄茎到肩膀的过渡区，该处轮廓方向迅速变化，
    # 一阶导数可以量化这种"变化强度"。左侧 X 递增→导数正，右侧 X 递减→导数负。
    left_seg_pts = left_contour[np.asarray(left_seg_idxs, dtype=int), :2]
    right_seg_pts = right_contour[np.asarray(right_seg_idxs, dtype=int), :2]
    left_x, left_y = normalize_xy(left_seg_pts)
    right_x, right_y = normalize_xy(right_seg_pts)
    left_d = compute_derivatives_from_xy(left_x, left_y, derv_order=1)
    right_d = compute_derivatives_from_xy(right_x, right_y, derv_order=1)

    # 6. 导数阈值筛选 → 候选点打包 → 最优点选择
    # WHY：左侧轮廓向外扩散（X 递增）时导数正且大，右侧向内收拢时导数负且小。
    # 用 LEFT_DERIV_THRESHOLD=0.3（左侧保留大于 0.3 的点）和
    # RIGHT_DERIV_THRESHOLD=-0.3（右侧保留小于 -0.3 的点）过滤掉过渡不明显的区域。
    # 筛选后为每个候选点计算侧向角和长轴转角，打包成 AngleCandidate 结构。
    left_keep = select_points_by_derivative(left_seg_pts, left_x, left_d, NeckRoot.LEFT_DERIV_THRESHOLD)
    right_keep = select_points_by_derivative(
        right_seg_pts,
        right_x,
        right_d,
        NeckRoot.RIGHT_DERIV_THRESHOLD,
        keep_greater=False,
    )
    left_candidates = _build_angle_candidates(left_contour, left_keep)
    right_candidates = _build_angle_candidates(right_contour, right_keep)
    if not left_candidates or not right_candidates:
        logger.warning(
            "No neck-root candidates after derivative filtering: "
            f"left={len(left_candidates)} right={len(right_candidates)}"
        )
        raise ValueError("Failed to detect neck root candidates.")
    # WHY：候选点中可能存在多个导数满足条件的点，需要进一步用长轴转角筛选。
    # 颈根在解剖上是窄茎→肩部的过渡区，长轴转角应在 20°~70° 之间（接近垂直的窄茎
    # 段 < 15°，到肩部后 > 80°）。选最接近 55° 的候选作为最优，因为 55° 大致位于
    # 过渡区中间，对大多数体态最稳定。左右两侧独立选择。
    left_best = _select_best_by_long_axis(
        candidates=left_candidates,
        contour=left_contour,
        seg_pts=left_seg_pts,
        seg_x=left_x,
        seg_d=left_d,
        side="left",
    )
    right_best = _select_best_by_long_axis(
        candidates=right_candidates,
        contour=right_contour,
        seg_pts=right_seg_pts,
        seg_x=right_x,
        seg_d=right_d,
        side="right",
    )
    neck_root = np.vstack([left_best.point[:2], right_best.point[:2]])

    neck_debug = build_neck_root_debug(
        is_debug=is_debug,
        waist_points=waist_points,
        left_c=left_contour,
        right_c=right_contour,
        bin_info=bin_info,
        mode_idx=mode_idx,
        mode_width=mode_width,
        waist_w=waist_w,
        left_candidates=left_candidates,
        right_candidates=right_candidates,
        left_x=left_x,
        left_d=left_d,
        right_x=right_x,
        right_d=right_d,
        left_best=left_best,
        right_best=right_best,
    )
    return neck_root, neck_debug


# --- Helper functions (private) ---


def _build_angle_candidates(
    contour: np.ndarray,
    points: np.ndarray,
) -> list[AngleCandidate]:
    candidates: list[AngleCandidate] = []
    for pt in points:
        _, angle_deg, left_pt, right_pt, left_dist, right_dist = compute_lateral_angle_at_point(
            contour, pt, distance=NeckRoot.ANGLE_SAMPLE_DISTANCE
        )
        axis_deg = long_axis_angle(contour, float(pt[0]), float(pt[1]), arc_len=NeckRoot.LONG_AXIS_ARC_LEN)
        candidates.append(
            AngleCandidate(
                point=pt.copy(),
                angle_deg=float(angle_deg),
                axis_deg=float(axis_deg),
                left_pt=left_pt.copy(),
                right_pt=right_pt.copy(),
                left_dist=float(left_dist),
                right_dist=float(right_dist),
            )
        )
    return candidates


def _select_best_by_long_axis(
    candidates: list[AngleCandidate],
    contour: np.ndarray,
    seg_pts: np.ndarray,
    seg_x: np.ndarray,
    seg_d: np.ndarray,
    side: str,
) -> AngleCandidate:
    """在导数筛选后的候选点中按长轴角最接近 45° 选择颈根。

    WHY：设计三条 fallback 路径来覆盖解剖变异——
    正常体态下导数筛选已经能产出一批候选点，只需从中挑长轴角最合适的即可。
    但存在颈椎前凸/后凸、耸肩姿势等导致导数信号不佳的情况，
    此时需要逐步放宽搜索范围（候选段 → 整条轮廓）确保总能找到合理位置。

    Primary: 从 candidates 中过滤出 long_axis ∈ [20°,70°] 的候选，
    选长轴角最接近 45°（中间值）的点。

    Fallback 链：
    1. 无满足角度条件的候选 → 沿候选段从上往下找第一个导数+角度都满足的点
    2. 段内无有效点 → 沿整条裁剪后轮廓搜索

    Returns:
        AngleCandidate: 选中的最佳候选点。

    Raises:
        ValueError: 所有 fallback 均失败时。
    """
    # --- Primary: 长轴角最接近 45° 的候选 ---
    # WHY：选择 55° 作为目标而不是 45° 是因为颈根过渡区在多数体态下
    # 倾向于 > 45°（接近肩部），55° 能更稳定地落在过渡区中部。
    valid = [c for c in candidates if NeckRoot.LONG_AXIS_ANGLE_MIN <= c.axis_deg <= NeckRoot.LONG_AXIS_ANGLE_MAX]
    if valid:
        best = min(valid, key=lambda c: abs(c.axis_deg - 55.0))
        logger.info(
            f"Selected {side} neck root from {len(valid)}/{len(candidates)} "
            f"angle-valid candidates, best axis_deg={best.axis_deg:.1f}°"
        )
        return best

    logger.warning(
        f"No {side} candidate meets long_axis_angle constraint, falling back "
        f"to contour scan (n_candidates={len(candidates)})"
    )

    # --- Fallback 1: 候选段从上往下扫描 ---
    # WHY：如果导数筛选后的候选点全都不满足长轴转角条件（通常是导数阈值过松，
    # 保留了太多非颈根区域点），则在原始的候选段内沿 Y 递减方向搜索第一个
    # 同时满足导数阈值和长轴转角条件的点，这比在原 candidates 中硬选更可靠。
    result = _find_first_valid_point(seg_pts, contour, seg_x, seg_d, side, "segment fallback")
    if result is not None:
        return result

    # --- Fallback 2: 整条裁剪后轮廓扫描 ---
    # WHY：候选段内仍无有效点时，说明裁剪范围（TOP_RATIO=0.3）可能因体态特殊
    # 而把颈根区域裁掉了部分。此时放宽到整条已裁剪轮廓上重新扫描，确保覆盖所有可能位置。
    full_x, full_y = normalize_xy(contour)
    full_d = compute_derivatives_from_xy(full_x, full_y, derv_order=1)
    result = _find_first_valid_point(contour[..., :2], contour, full_x, full_d, side, "full contour fallback")
    if result is not None:
        return result

    raise ValueError(f"Failed to detect {side} neck root candidate (all fallbacks exhausted).")


def _find_first_valid_point(
    search_pts: np.ndarray,
    contour: np.ndarray,
    x_arr: np.ndarray,
    d_arr: np.ndarray,
    side: str,
    source: str,
) -> AngleCandidate | None:
    """在 search_pts 中从 Y 最大(最高)往下扫,找第一个导数+长轴角都满足的点。

    WHY：从最高点（Y 最大）开始往下扫，因为颈根必然位于轮廓的上端区域。
    对于正常体态，第一个满足条件的点就是最接近窄茎上端过渡区的位置，
    不会跑到肩膀更下方去。
    """
    threshold = NeckRoot.LEFT_DERIV_THRESHOLD if side == "left" else NeckRoot.RIGHT_DERIV_THRESHOLD
    keep_greater = side == "left"
    order = np.argsort(search_pts[:, 1])[::-1]

    for idx in order:
        pt = search_pts[idx]
        deriv_val = float(np.interp(float(pt[0]), x_arr, d_arr, left=np.nan, right=np.nan))
        if not np.isfinite(deriv_val):
            continue
        if keep_greater and deriv_val <= threshold:
            continue
        if not keep_greater and deriv_val >= threshold:
            continue
        axis_deg = long_axis_angle(contour, float(pt[0]), float(pt[1]), arc_len=NeckRoot.LONG_AXIS_ARC_LEN)
        if not (NeckRoot.LONG_AXIS_ANGLE_MIN <= axis_deg <= NeckRoot.LONG_AXIS_ANGLE_MAX):
            continue
        logger.info(f"Found valid {side} neck root via {source} at ({pt[0]:.1f}, {pt[1]:.1f})")
        _, angle_deg, left_pt, right_pt, left_dist, right_dist = compute_lateral_angle_at_point(
            contour, pt, distance=NeckRoot.ANGLE_SAMPLE_DISTANCE
        )
        return AngleCandidate(
            point=pt.copy(),
            angle_deg=float(angle_deg),
            axis_deg=float(axis_deg),
            left_pt=left_pt.copy(),
            right_pt=right_pt.copy(),
            left_dist=float(left_dist),
            right_dist=float(right_dist),
        )
    return None


def _clip_and_smooth_top_contours(
    left_c: np.ndarray,
    right_c: np.ndarray,
    waist_points: np.ndarray,
    sigma: float = NeckRoot.CONTOUR_SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    """裁剪左右轮廓的上半区域并对结果做高斯平滑。

    Args:
        left_c: 左侧轮廓点数组，形状 (N, 3)。
        right_c: 右侧轮廓点数组，形状 (N, 3)。
        waist_points: 腰部左右点数组，形状 (2, 3)，用于定义左右裁剪边界。
        sigma: 高斯核的标准差，控制平滑程度。

    Returns:
        tuple[np.ndarray, np.ndarray]: 裁剪并平滑后的左右轮廓，均为 (M, 2)。
    """

    contour_y_min = min(left_c[:, 1].min(), right_c[:, 1].min())
    contour_y_max = max(left_c[:, 1].max(), right_c[:, 1].max())
    y_range = contour_y_max - contour_y_min
    waist_x_l, waist_x_r = waist_points[:, 0]

    # WHY：裁剪只保留轮廓顶部 30% 的区域（TOP_RATIO=0.3），因为颈根一定在躯干上半段。
    # 左侧用 waist_x_l 作为 X 内侧边界（防止裁到对侧），右侧用 waist_x_r 同理。
    # 用 extract_longest_contiguous_segment_in_box 确保取到的是连续段，
    # 避免偶尔的孤立噪声点被错误包含。
    clip_y_min = contour_y_max - NeckRoot.TOP_RATIO * y_range
    left_idxs = extract_longest_contiguous_segment_in_box(
        left_c,
        x_min=float(waist_x_l),
        x_max=float(left_c[:, 0].max()),
        y_min=float(clip_y_min),
        y_max=float(contour_y_max),
        closed=False,
    )
    right_idxs = extract_longest_contiguous_segment_in_box(
        right_c,
        x_min=float(right_c[:, 0].min()),
        x_max=float(waist_x_r),
        y_min=float(clip_y_min),
        y_max=float(contour_y_max),
        closed=False,
    )
    if len(left_idxs) == 0 or len(right_idxs) == 0:
        raise ValueError("No contour points in top region after clipping: check waist points and contour data")

    # WHY：用高斯平滑消除局部噪声杂波，mode="nearest" 防止边界处出现外推伪影。
    # sigma=0.5 是轻量平滑，保留轮廓大体形状的同时滤掉单点毛刺。
    left_c = smooth_contour(left_c[left_idxs, :2], sigma=sigma, mode="nearest")
    right_c = smooth_contour(right_c[right_idxs, :2], sigma=sigma, mode="nearest")
    return left_c, right_c


def _compute_candidate_segments(
    left_contour: np.ndarray,
    right_contour: np.ndarray,
    waist_w: float,
) -> tuple[np.ndarray, np.ndarray, int, list[dict], float]:
    """返回候选段索引、模态 bin 信息和模态宽度。

    WHY：两步定位法——
    1. 先通过宽度直方图的众数 bin 确定颈宽典型值 W_mode 及其 Y 位置（mode_y），
       这个 Y 位置是颈根候选区的上界。
    2. 从 mode_y 向下搜索，找到第一条宽度 ≥ 1.5×W_mode 的行作为下界，
       因为从颈部到肩膀宽度会显著增大，1.5 倍意味着已经进入肩部区域。
    上下界之间的轮廓段就是颈根候选搜索空间。
    """
    lines = build_width_profile_lines(left_contour, right_contour)
    mode_idx, bin_info = compute_histogram_mode_width(lines, waist_w)
    mode_bin = bin_info[mode_idx]
    mode_width = float(mode_bin["width"])  # W_mode
    mode_y = float(mode_bin["y"])
    if mode_bin["x_left"] is None or mode_bin["x_right"] is None:
        logger.warning("Mode bin is empty while building neck-root candidate segments.")
        raise ValueError("Failed to build neck-root candidate segments.")
    top_left = np.array([mode_bin["x_left"], mode_y])
    top_right = np.array([mode_bin["x_right"], mode_y])

    lower_bound_width = mode_width * NeckRoot.LOWER_BOUND_WIDTH_RATIO
    lower_bound_mask = (lines[:, 3] >= lower_bound_width) & (lines[:, 2] < mode_y)
    if not np.any(lower_bound_mask):
        logger.warning("No contour rows found below the mode-width lower bound while building neck root.")
        raise ValueError("Failed to locate a lower-bound contour row for neck root.")
    lower_bound_line = lines[lower_bound_mask][np.argmax(lines[lower_bound_mask][:, 2])]
    bottom_left = np.array([lower_bound_line[0], lower_bound_line[2]])
    bottom_right = np.array([lower_bound_line[1], lower_bound_line[2]])

    left_seg_idxs = search_segment_indices(left_contour, bottom_left, top_left)
    right_seg_idxs = search_segment_indices(right_contour, top_right, bottom_right)
    return left_seg_idxs, right_seg_idxs, mode_idx, bin_info, mode_width
