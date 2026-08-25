"""肩臂转点（shoulder_transition）对称特征。"""

from __future__ import annotations

import numpy as np


def extract_shoulder(
    landmarks: dict,
    trunk_length: float,
) -> dict[str, float]:
    """计算肩臂转点左右对称特征。

    Args:
        landmarks: 完整的 landmark 字典。
        trunk_length: 躯干长度（mm），用于 v_diff_ratio 归一化。

    Returns:
        shoulder_transition_distance_3d, _anterior_diff, _vertical_diff,
        _slope_angle, _v_diff_ratio。
    """
    left: np.ndarray = np.asarray(landmarks["shoulder_transition_L"])
    right: np.ndarray = np.asarray(landmarks["shoulder_transition_R"])
    diff: np.ndarray = right - left

    dist_3d: float = float(np.linalg.norm(diff))
    h_diff: float = float(diff[2])
    v_diff: float = float(diff[1])
    slope: float = float(np.degrees(np.arctan2(abs(v_diff), abs(diff[0]))))
    v_ratio: float = abs(v_diff) / trunk_length if trunk_length > 0 else 0.0

    return {
        "shoulder_transition_distance_3d": dist_3d,
        "shoulder_transition_anterior_diff": h_diff,
        "shoulder_transition_vertical_diff": v_diff,
        "shoulder_transition_slope_angle": slope,
        "shoulder_transition_v_diff_ratio": v_ratio,
    }
