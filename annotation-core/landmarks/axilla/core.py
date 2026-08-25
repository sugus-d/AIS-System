"""Axilla detection via derivative-based analysis.

Step 1: Contour Preprocessing — Gaussian smooth, Y clip, lower boundary.
Step 2: Find Arm Boundary (outer X bound) via derivative thresholds.
Step 3: Find d < 0 candidates — top 25 % most negative.
Step 4: Select axilla point via max cosine.
"""

import numpy as np

from utils.logger import logger

from ..constants import Axilla
from ._pipeline import _detect_single_side
from .arms import _has_arms

_MAX_ARM_ASYMMETRY_MM = 20.0  # 左右腋窝 Y 高度差超过该值触发单臂矫正


def detect_axilla_strips(
    left_c: np.ndarray,
    right_c: np.ndarray,
    widths: np.ndarray,
    y_cen: np.ndarray,
    neck_root: np.ndarray,
    y_min: float,
    y_range: float,
    waist_points: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """通过导数分析在轮廓带（strips）中检测腋窝（axilla）点。

    Args:
        left_c: 左侧轮廓 (N,2).
        right_c: 右侧轮廓 (N,2).
        widths: 每行的宽度数组，用于判定手臂存在性。
        y_cen: 轮廓中心 Y，用于定位带中心。
        neck_root: 颈根左右两点数组 shape (2,2) 或 (2,3)，用于定位搜索带。
        y_min: 轮廓最小 Y，用于归一化。
        y_range: 轮廓 Y 范围（max-min），用于决定搜索带大小。
        waist_points: 腰部左右点，用于限定外侧边界。

    Returns:
        axilla_out: numpy.ndarray, shape (2,2)，左右腋窝点（x,y）。
        debug: dict，包含中间计算与候选点信息，便于可视化与调试。
    """
    has_left_arm, has_right_arm = _has_arms(left_c, right_c, widths, y_cen, y_min, y_range)

    logger.info(
        f"[AXILLA] left_c={len(left_c)} pts, right_c={len(right_c)} pts, "
        f"y_range={y_range:.1f}, "
        f"has_left_arm={has_left_arm}, has_right_arm={has_right_arm}"
    )

    nr_y = float(np.mean(neck_root[:, 1]))
    # 颈根到腰部按比例切割搜索带：Y_ROI_HI_RATIO 控制上界，Y_ROI_LO_RATIO 控制下界
    # 腋窝必然位于颈根下方的过渡区域，用固定比例裁剪可排除颈部和腰腹无关区域
    y_roi_hi = nr_y - y_range * Axilla.Y_ROI_HI_RATIO
    y_roi_lo = nr_y - y_range * Axilla.Y_ROI_LO_RATIO
    waist_y = float(np.mean(waist_points[:, 1]))
    # y_ref 作为无手臂侧的 Y 锚点参考：取搜索带中线和腰部中点之间的位置
    # 无手臂时轮廓外侧缺少 V 形凹陷，需要用 Y 高度惩罚来引导候选点落在此区间
    y_ref = waist_y + 0.5 * (y_roi_hi - waist_y)

    nrL_x = float(neck_root[0, 0])
    nrR_x = float(neck_root[1, 0])
    wL_x = float(waist_points[0, 0])
    wR_x = float(waist_points[1, 0])
    mid_x = (wL_x + wR_x) / 2.0
    left_mid = (wL_x + nrL_x) / 2.0
    right_mid = (nrR_x + wR_x) / 2.0
    # 外侧边界以腰部为基准向外扩展：对左而言更左，对右而言更右
    # 腋窝一定在颈根和腰部之间，但手臂外扩时轮廓会延伸到腰部外侧，不能直接卡死腰线
    outer_L = wL_x + Axilla.OUTER_BOUND_RATIO * (wL_x - mid_x)
    outer_R = wR_x + Axilla.OUTER_BOUND_RATIO * (wR_x - mid_x)

    results = []
    debug = {"left": {}, "right": {}}
    for contour, nr_pt, has_arm, side_name in [
        (left_c, neck_root[0], has_left_arm, "left"),
        (right_c, neck_root[1], has_right_arm, "right"),
    ]:
        if side_name == "left":
            x_lo = outer_L
            x_hi = left_mid
        else:
            x_lo = right_mid
            x_hi = outer_R

        best_pt, side_debug = _detect_single_side(
            contour=contour,
            nr_pt=nr_pt,
            has_arm=has_arm,
            side_name=side_name,
            x_lo=x_lo,
            x_hi=x_hi,
            y_roi_lo=y_roi_lo,
            y_roi_hi=y_roi_hi,
            distance=10.0,
            nr_y=nr_y,
            y_range=y_range,
            y_ref=y_ref,
        )
        results.append(best_pt)
        debug[side_name] = side_debug

    axilla_out = np.zeros((2, 2), dtype=np.float64)
    for i, pt2 in enumerate(results):
        axilla_out[i, :2] = pt2[:2]

    # 当单侧手臂缺失时，无手臂侧的轮廓没有 V 形腋窝凹陷
    # 导数分析容易选到比正常位置低很多的点，导致左右高度严重不对称（>20mm）
    # 此处用有手臂侧的 Y 高度矫正无手臂侧：取加权平均值作为目标 Y，重新在 20mm 带内搜 X 最外侧点
    dy_ax = abs(axilla_out[0, 1] - axilla_out[1, 1])
    if has_left_arm != has_right_arm and dy_ax > _MAX_ARM_ASYMMETRY_MM:
        no_arm_idx = 1 if has_left_arm else 0
        arm_ax_y = float(axilla_out[1 - no_arm_idx, 1])
        no_arm_name = "right" if no_arm_idx else "left"
        target_y = arm_ax_y * 0.6 + float(axilla_out[no_arm_idx, 1]) * 0.4
        sc = np.array(debug[no_arm_name].get("smoothed_clipped", []))
        if len(sc) > 0:
            x_lo_n, x_hi_n = (outer_L, left_mid) if no_arm_idx == 0 else (right_mid, outer_R)
            x_bound = x_hi_n * 1.15 if no_arm_idx else x_lo_n * 0.85
            y_lo_n, y_hi_n = target_y - 10.0, target_y + 10.0
            mask = (
                (sc[:, 1] >= y_lo_n)
                & (sc[:, 1] <= y_hi_n)
                & (sc[:, 0] >= (x_lo_n if no_arm_idx == 0 else x_lo_n))
                & (sc[:, 0] <= (x_bound if no_arm_idx else x_hi_n))
            )
            if mask.sum() > 0:
                candidates = sc[mask]
                best = (
                    candidates[np.argmin(candidates[:, 0])]
                    if no_arm_idx == 0
                    else candidates[np.argmax(candidates[:, 0])]
                )
                axilla_out[no_arm_idx, :2] = best[:2]

    logger.info(f"[AXILLA] left_arm={'Y' if has_left_arm else 'N'} right_arm={'Y' if has_right_arm else 'N'}")
    return axilla_out, debug
