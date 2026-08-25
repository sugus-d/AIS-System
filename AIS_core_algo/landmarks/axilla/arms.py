"""Arm presence detection for axilla landmark."""

import numpy as np

from ..constants import Axilla

_MIN_SHOULDER_POINTS = 3  # 肩部区间最少采样点数（太少不判定手臂）


def _has_arms(
    left_c: np.ndarray,
    right_c: np.ndarray,
    widths: np.ndarray,
    y_cen: np.ndarray,
    y_min: float,
    y_range: float,
) -> tuple[bool, bool]:
    """Detect if arms are present on each side.

    Returns (has_left_arm, has_right_arm).
    """
    # 用 Y 方向归一化位置（0=底部, 1=顶部）确定腰部与肩部的垂直区间
    frac = (y_cen - y_min) / y_range
    waist_mask = (frac >= Axilla.WAIST_FRAC_LO) & (frac <= Axilla.WAIST_FRAC_HI)
    shoulder_mask = (frac >= Axilla.SHOULDER_FRAC_LO) & (frac <= Axilla.SHOULDER_FRAC_HI)

    waist_w = float(np.median(widths[waist_mask])) if waist_mask.sum() > 0 else 1.0

    has_left = has_right = False
    # 需要足够的肩部数据点和合理的腰部宽度才开始判定
    if shoulder_mask.sum() > _MIN_SHOULDER_POINTS and waist_w > 1:
        waist_y_lo = y_cen[waist_mask].min() if waist_mask.sum() > 0 else y_min
        waist_y_hi = y_cen[waist_mask].max() if waist_mask.sum() > 0 else y_min
        shoulder_y_lo = y_cen[shoulder_mask].min() if shoulder_mask.sum() > 0 else y_min
        shoulder_y_hi = y_cen[shoulder_mask].max() if shoulder_mask.sum() > 0 else y_min

        # 肩部区域的外侧扩散距离：有手臂时肩膀向外大幅凸出，远大于腰部
        sh_left = left_c[(left_c[:, 1] >= shoulder_y_lo) & (left_c[:, 1] <= shoulder_y_hi)]
        sh_right = right_c[(right_c[:, 1] >= shoulder_y_lo) & (right_c[:, 1] <= shoulder_y_hi)]
        left_extent = float(np.abs(sh_left[:, 0].min())) if len(sh_left) > 0 else 0.0
        right_extent = float(np.abs(sh_right[:, 0].max())) if len(sh_right) > 0 else 0.0

        # 腰部区域的外侧距离作为基准参照（无手臂干扰的躯干宽度）
        waist_left_pts = left_c[(left_c[:, 1] >= waist_y_lo) & (left_c[:, 1] <= waist_y_hi)]
        waist_right_pts = right_c[(right_c[:, 1] >= waist_y_lo) & (right_c[:, 1] <= waist_y_hi)]
        waist_left = float(np.abs(waist_left_pts[:, 0].min())) if len(waist_left_pts) > 0 else 1.0
        waist_right = float(np.abs(waist_right_pts[:, 0].max())) if len(waist_right_pts) > 0 else 1.0

        # 肩部扩散距离超出腰部的比例超过阈值说明存在手臂
        # 无手臂时肩部不会比腰部宽出显著比例
        has_left = (left_extent / max(waist_left, 1.0)) >= Axilla.ARM_EXTENT_RATIO
        has_right = (right_extent / max(waist_right, 1.0)) >= Axilla.ARM_EXTENT_RATIO
    return has_left, has_right
