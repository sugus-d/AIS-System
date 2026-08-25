"""腰部（waist + waist_lower）对称特征。

waist_lower 从侧向轮廓中 waist Y 以下的点推导。
"""

from __future__ import annotations

import numpy as np


def _find_lower_point(
    contour: np.ndarray,
    ref_y: float,
    offset: float = 80.0,
) -> np.ndarray:
    """在轮廓上找 waist Y 以下 offset mm 附近的 3D 点。

    沿轮廓搜索 Y 值最接近 ref_y - offset 的点，
    因为轮廓是 (X, Y) 2D，返回时补 Z=0（由调用方负责 3D 映射）。

    Args:
        contour: (N, 2) 轮廓点 [X, Y]。
        ref_y: 参考 Y（waist 平均 Y）。
        offset: 向下偏移量（mm）。

    Returns:
        (3,) 近似 3D 点 [X, Y, 0]。
    """
    target_y: float = ref_y - offset
    # 限制搜索范围：只能在 waist 以下
    below: np.ndarray = contour[contour[:, 1] <= ref_y]
    if len(below) == 0:
        # 回退：取轮廓最低点
        idx: int = int(np.argmin(contour[:, 1]))
        return np.array([float(contour[idx, 0]), float(contour[idx, 1]), 0.0])
    idx = int(np.argmin(np.abs(below[:, 1] - target_y)))
    return np.array([float(below[idx, 0]), float(below[idx, 1]), 0.0])


def extract_waist(
    landmarks: dict,
    trunk_length: float,
) -> dict[str, float]:
    """计算左右腰部 + waist_lower 对称特征。

    注意 waist_lower 点坐标为近似值（从侧向轮廓推算），
    非精确 lift_to_mesh 结果。

    Args:
        landmarks: 完整的 landmark 字典（需包含 lateral_profiles）。
        trunk_length: 躯干长度（mm）。

    Returns:
        waist_* 和 waist_lower_* 共 10 个特征。
    """
    result: dict[str, float] = {}

    # ── waist ──
    pts: np.ndarray = landmarks["waist"]  # (2, 3)
    diff: np.ndarray = pts[1] - pts[0]
    dist_3d: float = float(np.linalg.norm(diff))
    h_diff: float = float(diff[2])
    v_diff: float = float(diff[1])
    slope: float = float(np.degrees(np.arctan2(abs(v_diff), abs(diff[0]))))
    v_ratio: float = abs(v_diff) / trunk_length if trunk_length > 0 else 0.0
    result.update(
        {
            "waist_distance_3d": dist_3d,
            "waist_anterior_diff": h_diff,
            "waist_vertical_diff": v_diff,
            "waist_slope_angle": slope,
            "waist_v_diff_ratio": v_ratio,
        }
    )

    # ── waist_lower：从侧向轮廓推导 ──
    profiles: dict | None = landmarks.get("lateral_profiles")
    if profiles is not None:
        left_c: np.ndarray = profiles["left_contour"]  # (N, 2)
        right_c: np.ndarray = profiles["right_contour"]
        wa_y: float = float(np.mean(pts[:, 1]))
        left_lower: np.ndarray = _find_lower_point(left_c, wa_y)
        right_lower: np.ndarray = _find_lower_point(right_c, wa_y)
    else:
        # ponytail: 无轮廓时用 waist 点向下偏移 80mm 近似
        wa_y = float(np.mean(pts[:, 1]))
        left_lower = np.array([float(pts[0, 0]), wa_y - 80.0, float(pts[0, 2])])
        right_lower = np.array([float(pts[1, 0]), wa_y - 80.0, float(pts[1, 2])])

    ldiff: np.ndarray = right_lower - left_lower
    ldist_3d: float = float(np.linalg.norm(ldiff))
    lh_diff: float = float(ldiff[2])
    lv_diff: float = float(ldiff[1])
    lslope: float = float(np.degrees(np.arctan2(abs(lv_diff), abs(ldiff[0]))))
    lv_ratio: float = abs(lv_diff) / trunk_length if trunk_length > 0 else 0.0
    result.update(
        {
            "waist_lower_distance_3d": ldist_3d,
            "waist_lower_anterior_diff": lh_diff,
            "waist_lower_vertical_diff": lv_diff,
            "waist_lower_slope_angle": lslope,
            "waist_lower_v_diff_ratio": lv_ratio,
        }
    )

    return result
