"""颈根（neck_root）对称特征。"""

from __future__ import annotations

import numpy as np


def extract_neck_root(
    landmarks: dict,
    trunk_length: float,
) -> dict[str, float]:
    """计算颈根左右对称特征。

    Args:
        landmarks: 完整的 landmark 字典。
        trunk_length: 躯干长度（mm），用于 v_diff_ratio 归一化。

    Returns:
        neck_root_distance_3d, _anterior_diff, _vertical_diff,
        _slope_angle, _v_diff_ratio。
    """
    left: np.ndarray = np.asarray(landmarks["neck_root_L"])
    right: np.ndarray = np.asarray(landmarks["neck_root_R"])
    diff: np.ndarray = right - left

    dist_3d: float = float(np.linalg.norm(diff))
    h_diff: float = float(diff[2])  # Z (anterior-posterior)
    v_diff: float = float(diff[1])  # Y (inferior-superior)
    # 水平夹角：90° 完全垂直，0° 完全水平
    slope: float = float(np.degrees(np.arctan2(abs(v_diff), abs(diff[0]))))
    v_ratio: float = abs(v_diff) / trunk_length if trunk_length > 0 else 0.0

    return {
        "neck_root_distance_3d": dist_3d,
        "neck_root_anterior_diff": h_diff,
        "neck_root_vertical_diff": v_diff,
        "neck_root_slope_angle": slope,
        "neck_root_v_diff_ratio": v_ratio,
    }
