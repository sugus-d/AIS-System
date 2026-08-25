"""脊柱（spine）分段特征 — 从 4 个脊柱点推导 6 点序列。

landmarks 中只有 4 个脊柱点（P0-P3，对应 4 个 bilateral pair），
P5 和 P4 通过插值/外推从已有点推导：
  P5  = P2 与 P3 的中点
  P4  = P3 以下延长 P2→P3 方向 1/3 距离

特征名与 CSV ``features_schemeA.csv`` 中 ``spine_*`` 列一致。
"""

from __future__ import annotations

import numpy as np

# 基础脊柱点数量（P0, P1, P2, P3）
_N_SPINE_BASE: int = 4
# 累积曲率需要的段落数（P0-P1, P3-P4）
_N_ANGLE_PAIRS: int = 2


def _derive_p5p4(
    spine_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """从 4 个基础脊柱点推导 P5 和 P4。

    Args:
        spine_pts: (4, 3) — [P0, P1, P2, P3]。

    Returns:
        (P5, P4) 各为 (3,)。
    """
    p2: np.ndarray = spine_pts[2]  # axilla 水平
    p3: np.ndarray = spine_pts[3]  # waist 水平
    # P5: P2 与 P3 的中点
    p5: np.ndarray = (p2 + p3) / 2.0
    # P4: P3 以下延长 P2→P3 方向 1/3
    vec: np.ndarray = p3 - p2
    p4: np.ndarray = p3 + vec / 3.0
    return p5, p4


def _seg_features(
    p_start: np.ndarray,
    p_end: np.ndarray,
    seg_name: str,
    height: float | None,
) -> dict[str, float]:
    """计算单个脊柱段落的 4 个特征。

    Args:
        p_start: 起点 (3,)。
        p_end: 终点 (3,)。
        seg_name: 如 ``"P0_P1"``。
        height: 身高（mm），可选。

    Returns:
        {spine_{seg_name}_length, _len_ratio, _angle_vertical, _lateral_deviation}。
    """
    vec: np.ndarray = p_end - p_start
    seg_len: float = float(np.linalg.norm(vec))

    # 与垂直方向 (Y axis) 的夹角
    vertical: np.ndarray = np.array([0.0, 1.0, 0.0])
    cos_vert: float = float(np.dot(vec, vertical) / (np.linalg.norm(vec) + 1e-12))
    angle_vert: float = float(np.degrees(np.arccos(np.clip(cos_vert, -1.0, 1.0))))

    # 侧向偏移：线段中点 X 坐标的绝对值
    mid_x: float = float((p_start[0] + p_end[0]) / 2.0)
    lat_dev: float = abs(mid_x)

    result: dict[str, float] = {
        f"spine_{seg_name}_length": seg_len,
        f"spine_{seg_name}_len_ratio": 0.0,
        f"spine_{seg_name}_angle_vertical": angle_vert,
        f"spine_{seg_name}_lateral_deviation": lat_dev,
    }
    if height is not None and height > 0:
        result[f"spine_{seg_name}_len_ratio"] = float(seg_len / height)
    return result


def extract_spine(
    landmarks: dict,
    height: float | None = None,
) -> dict[str, float]:
    """计算脊柱分段特征。

    从 4 个基础脊柱点（P0-P3）推导 P5、P4，
    计算 5 个段（P0-P1, P1-P2, P2-P5, P5-P3, P3-P4）的特征，
    以及总长、曲率、最大侧向偏移。

    Args:
        landmarks: 完整的 landmark 字典（需包含 ``spine_points``）。
        height: 身高（mm），可选。提供时计算 ``len_ratio``。

    Returns:
        dict: ``spine_*`` 共计 20+ 个特征。
    """
    # 从 landmark 获取 4 个基础脊柱点
    spine_pts: np.ndarray = landmarks["spine_points"]  # (4, 3)
    if len(spine_pts) < _N_SPINE_BASE:
        return {}  # 输入不足

    p0, p1, p2, p3 = spine_pts[0], spine_pts[1], spine_pts[2], spine_pts[3]
    p5, p4 = _derive_p5p4(spine_pts)

    # 6 点完整序列
    pts: dict[str, np.ndarray] = {
        "P0": p0,
        "P1": p1,
        "P2": p2,
        "P5": p5,
        "P3": p3,
        "P4": p4,
    }

    # 5 个段落
    segments: list[tuple[str, str, str]] = [
        ("P0_P1", "P0", "P1"),
        ("P1_P2", "P1", "P2"),
        ("P2_P5", "P2", "P5"),
        ("P5_P3", "P5", "P3"),
        ("P3_P4", "P3", "P4"),
    ]

    features: dict[str, float] = {}
    segment_angles: list[float] = []

    for seg_name, key_start, key_end in segments:
        feat: dict[str, float] = _seg_features(
            pts[key_start],
            pts[key_end],
            seg_name,
            height,
        )
        features.update(feat)
        if seg_name in ("P0_P1", "P3_P4"):
            segment_angles.append(feat[f"spine_{seg_name}_angle_vertical"])

    # 总长
    features["spine_P0_P3_length"] = float(np.linalg.norm(p3 - p0))
    features["spine_P0_P4_length"] = float(np.linalg.norm(p4 - p0))
    if height is not None and height > 0:
        features["spine_P0_P3_len_ratio"] = float(np.linalg.norm(p3 - p0) / height)
        features["spine_P0_P4_len_ratio"] = float(np.linalg.norm(p4 - p0) / height)
    else:
        features["spine_P0_P3_len_ratio"] = 0.0
        features["spine_P0_P4_len_ratio"] = 0.0

    # 累积曲率：顶段 P0-P1 与底段 P3-P4 的角度差
    if len(segment_angles) >= _N_ANGLE_PAIRS:
        features["spine_curvature_P0P1_vs_P3P4"] = float(abs(segment_angles[1] - segment_angles[0]))
    else:
        features["spine_curvature_P0P1_vs_P3P4"] = 0.0

    # 最大侧向偏移：所有 6 个点的 X 绝对值的最大值
    x_vals: np.ndarray = np.abs(np.array([pts[k][0] for k in ["P0", "P1", "P2", "P5", "P3", "P4"]]))
    features["spine_max_lateral_deviation"] = float(x_vals.max())

    return features
