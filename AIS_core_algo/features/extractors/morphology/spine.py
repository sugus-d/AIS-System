"""脊柱（spine）分段特征 — 从 4 个脊柱语义点推导 6 点序列。

landmarks 中只有 4 个基础脊柱点（neck_root/scapular/axilla/waist 水平），
thoracic 和 waist_lower 通过插值/外推从已有点推导：
  thoracic    = axilla 与 waist 的中点
  waist_lower = waist 以下延长 axilla→waist 方向 1/3 距离

特征列名用段语义缩写（SPINE_SEG_SEMANTIC 单源），如
``spine_neck_scapular_length``（原 spine_P0_P1_length）。
"""

from __future__ import annotations

import numpy as np

from landmarks.constants import SPINE_SEG_SEMANTIC

# 基础脊柱语义键（4 个，其余 2 点由推导得到）
_SPINE_BASE_KEYS = (
    "neck_root_spine_point",  # P0 ↔ neck_root
    "scapular_spine_point",  # P1 ↔ scapular_peaks
    "axilla_spine_point",  # P2 ↔ axilla
    "waist_spine_point",  # P3 ↔ waist
)
# 推导键（thoracic = axilla/waist 中点；waist_lower = waist 下延）
_THORACIC_KEY = "thoracic_spine_point"
_WAIST_LOWER_KEY = "waist_lower_spine_point"
# 累积曲率需要的段落数（顶段 neck-scapular + 底段 waist-waistlower）
_N_ANGLE_PAIRS = 2


def _derive_p5p4(
    spine_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """从 4 个基础脊柱点推导 thoracic(P5) 和 waist_lower(P4)。

    Args:
        spine_pts: (4, 3) — [neck_root, scapular, axilla, waist]。

    Returns:
        (thoracic, waist_lower) 各为 (3,)。
    """
    axilla_pt: np.ndarray = spine_pts[2]  # axilla 水平
    waist_pt: np.ndarray = spine_pts[3]  # waist 水平
    # thoracic: axilla 与 waist 的中点
    thoracic: np.ndarray = (axilla_pt + waist_pt) / 2.0
    # waist_lower: waist 以下延长 axilla→waist 方向 1/3
    vec: np.ndarray = waist_pt - axilla_pt
    waist_lower: np.ndarray = waist_pt + vec / 3.0
    return thoracic, waist_lower


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
        seg_name: 段语义名，如 ``"neck_scapular"``。
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

    从 4 个基础脊柱点推导 thoracic、waist_lower，
    计算 5 个段（neck-scapular, scapular-axilla, axilla-thoracic,
    thoracic-waist, waist-waistlower）的特征，以及总长、曲率、最大侧向偏移。

    Args:
        landmarks: 扁平语义键 landmark 字典（含 4 个 *_spine_point）。
        height: 身高（mm），可选。提供时计算 ``len_ratio``。

    Returns:
        dict: ``spine_*`` 共计 20+ 个特征。
    """
    # 从语义键取 4 个基础脊柱点
    if not all(key in landmarks for key in _SPINE_BASE_KEYS):
        return {}  # 输入不足
    spine_pts: np.ndarray = np.array([landmarks[key] for key in _SPINE_BASE_KEYS], dtype=np.float64)
    if len(spine_pts) < len(_SPINE_BASE_KEYS):
        return {}  # 输入不足

    nr_pt, sc_pt, ax_pt, wa_pt = spine_pts
    th_pt, wl_pt = _derive_p5p4(spine_pts)

    # 6 点完整序列（语义键）
    pts: dict[str, np.ndarray] = {
        "neck_root_spine_point": nr_pt,
        "scapular_spine_point": sc_pt,
        "axilla_spine_point": ax_pt,
        _THORACIC_KEY: th_pt,
        "waist_spine_point": wa_pt,
        _WAIST_LOWER_KEY: wl_pt,
    }

    # 5 个段落（段语义名 → 起止语义键）
    segments: list[tuple[str, str, str]] = [
        ("neck_scapular", "neck_root_spine_point", "scapular_spine_point"),
        ("scapular_axilla", "scapular_spine_point", "axilla_spine_point"),
        ("axilla_thoracic", "axilla_spine_point", _THORACIC_KEY),
        ("thoracic_waist", _THORACIC_KEY, "waist_spine_point"),
        ("waist_waistlower", "waist_spine_point", _WAIST_LOWER_KEY),
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
        if seg_name in ("neck_scapular", "waist_waistlower"):
            segment_angles.append(feat[f"spine_{seg_name}_angle_vertical"])

    # 总长
    features["spine_neck_waist_length"] = float(np.linalg.norm(wa_pt - nr_pt))
    features["spine_neck_waistlower_length"] = float(np.linalg.norm(wl_pt - nr_pt))
    if height is not None and height > 0:
        features["spine_neck_waist_len_ratio"] = float(np.linalg.norm(wa_pt - nr_pt) / height)
        features["spine_neck_waistlower_len_ratio"] = float(np.linalg.norm(wl_pt - nr_pt) / height)
    else:
        features["spine_neck_waist_len_ratio"] = 0.0
        features["spine_neck_waistlower_len_ratio"] = 0.0

    # 累积曲率：顶段 neck-scapular 与底段 waist-waistlower 的角度差
    if len(segment_angles) >= _N_ANGLE_PAIRS:
        features["spine_curvature_neck_scapular_vs_waist_waistlower"] = float(
            abs(segment_angles[1] - segment_angles[0])
        )
    else:
        features["spine_curvature_neck_scapular_vs_waist_waistlower"] = 0.0

    # 最大侧向偏移：所有 6 个点的 X 绝对值的最大值
    x_vals: np.ndarray = np.abs(np.array([pts[key][0] for key in pts]))
    features["spine_max_lateral_deviation"] = float(x_vals.max())

    return features


# 供迁移/外部核对：旧 P 名特征 → 语义名（SPINE_SEG_SEMANTIC 驱动）
def legacy_feature_name(old_name: str) -> str:
    """spine_P0_P1_length 等旧 P 名特征 → 语义名。非 spine_P 名原样返回。"""
    if not old_name.startswith("spine_"):
        return old_name
    for p_name, semantic in SPINE_SEG_SEMANTIC.items():
        if p_name in old_name:
            return old_name.replace(p_name, semantic)
    return old_name
