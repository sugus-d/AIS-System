"""肩胛峰（scapular_peaks）对称特征。"""

from __future__ import annotations

import numpy as np


def extract_scapular(
    landmarks: dict,
    trunk_length: float,
) -> dict[str, float]:
    """计算肩胛峰左右对称特征。

    Args:
        landmarks: 完整的 landmark 字典。
        trunk_length: 躯干长度（mm），用于 v_diff_ratio 归一化。

    Returns:
        scapular_peaks_distance_3d, _anterior_diff, _vertical_diff,
        _slope_angle, _v_diff_ratio。
    """
    pts: np.ndarray = landmarks["scapular_peaks"]  # (2, 3) [left, right]
    diff: np.ndarray = pts[1] - pts[0]

    dist_3d: float = float(np.linalg.norm(diff))
    h_diff: float = float(diff[2])
    v_diff: float = float(diff[1])
    slope: float = float(np.degrees(np.arctan2(abs(v_diff), abs(diff[0]))))
    v_ratio: float = abs(v_diff) / trunk_length if trunk_length > 0 else 0.0

    return {
        "scapular_peaks_distance_3d": dist_3d,
        "scapular_peaks_anterior_diff": h_diff,
        "scapular_peaks_vertical_diff": v_diff,
        "scapular_peaks_slope_angle": slope,
        "scapular_peaks_v_diff_ratio": v_ratio,
    }
