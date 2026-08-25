"""Waist: left/right contour points at minimum-width Y-level."""

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Search window half-width (mm) around initial target Y for independent search
_WINDOW_HALF_WIDTH = 30.0
# Arc length (mm) for local vertical-angle computation
_ARC_LENGTH = 15.0
_MIN_REGION_POINTS = 5       # 搜索区间采样点数下限（太少回退保底）


def detect_waist(
    left_c: np.ndarray,
    right_c: np.ndarray,
    widths: np.ndarray,
    y_cen: np.ndarray,
    y_min: float,
    y_range: float,
) -> np.ndarray:
    """Detect waist points at trunk minimum-width Y-level.

    WHY: Full-curve smoothing with mode='nearest' avoids boundary reflection
    artifacts from previous window-only approach. Expanded search range
    [0.22, 0.55] excludes mesh truncation and neck/shoulder narrowing zones.

    Side-independent search within ±30mm of the minimum-width Y handles
    asymmetric cases (S0069, S0107) where left/right waist Y differ.

    Args:
        left_c: Left contour points (N, 2).
        right_c: Right contour points (N, 2).
        widths: Width per y-level.
        y_cen: Y-center values per level.
        y_min: Minimum Y of mesh.
        y_range: Y range of mesh.

    Returns:
        waist_points (2, 2)。
    """
    # 对整条宽度曲线做高斯平滑（mode='nearest' 避免边界反射），
    # 比局域窗口法更稳定地定位最窄处
    w_full_smooth = gaussian_filter1d(widths, sigma=5, mode="nearest")
    # 搜索范围 [0.22, 0.55]：排除网格截断区和颈肩自然缩窄区（0.55 以上），
    # 以及胸部以下开始扩宽的区域（0.22 以下）
    y_lo = y_min + 0.22 * y_range
    y_hi = y_min + 0.55 * y_range
    mask = (y_cen >= y_lo) & (y_cen <= y_hi)

    # 点数不够说明搜索区间异常（如截断网格），退回到轮廓中间位置作为保底
    if mask.sum() < _MIN_REGION_POINTS:
        mid = len(left_c) // 3
        return np.stack([left_c[mid], right_c[mid]])

    w_region = w_full_smooth[mask]
    min_idx = int(np.argmin(w_region))
    target_y = y_cen[mask][min_idx]

    # 左右独立搜索最垂直轮廓点：腰线在解剖上左右 Y 可不同
    # （如 S0069, S0107 因体态不对称导致左右腰高不同），
    # 统一搜索窗口围绕 target_y 各向两侧扩展 ±30mm
    li = _pick_most_vertical(left_c, target_y)
    ri = _pick_most_vertical(right_c, target_y)
    return np.stack([left_c[li], right_c[ri]])


def _vertical_angle_at(contour: np.ndarray, idx: int, arc_len: float = _ARC_LENGTH) -> float:
    """沿轮廓前后各 arc_len mm 的夹角与垂直方向的偏差(°)，0°=垂直。

    WHY：腰线最细处轮廓近乎垂直，用局部弧长夹角而非全局切线可以平滑局部噪声。
    实现复用 ``landmarks._validate_utils.long_axis_angle``（idx 取该点坐标，等价）。
    """
    from landmarks._validate_utils import long_axis_angle

    x, y = contour[idx, :2]
    return long_axis_angle(contour, x, y, arc_len=arc_len)


def _pick_most_vertical(
    contour: np.ndarray,
    target_y: float,
) -> int:
    """在 target_y ± _WINDOW_HALF_WIDTH 窗口内选长轴角最小（最垂直）的轮廓点索引。

    WHY：腰线最窄处的垂直度是定位金标准。
    在 target_y 附近用 ±30mm 窗口搜索，既包容左右不对称又不过度偏离最窄 Y。

    Returns:
        最垂直点的索引。窗口内无点（极端不对称）时退回到 Y 最接近的点。
    """
    y_vals = contour[:, 1]
    window_mask = np.abs(y_vals - target_y) <= _WINDOW_HALF_WIDTH
    candidates = np.where(window_mask)[0]
    if len(candidates) == 0:
        # 窗口内无点（极端不对称），退回到 Y 最接近的点
        return int(np.argmin(np.abs(y_vals - target_y)))
    angles = [_vertical_angle_at(contour, i) for i in candidates]
    best_local = int(np.argmin(angles))
    return int(candidates[best_local])
