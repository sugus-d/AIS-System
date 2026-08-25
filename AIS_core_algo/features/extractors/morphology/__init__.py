"""形态学特征提取 — 从 landmarks dict 计算固定 31 个几何特征。

每个子模块的 ``extract_xxx(landmarks, trunk_length)`` 为纯函数，
``extract_morphology()`` 汇总并过滤到 ``_KEEP_FEATURES`` 列表。
"""

from __future__ import annotations

import numpy as np

from .axilla import extract_axilla
from .neck_root import extract_neck_root
from .scapular import extract_scapular
from .shoulder import extract_shoulder
from .spine import extract_spine
from .waist import extract_waist

# 固定产出特征列表（经 |r|>0.9 去重后剩余，见 .claude/docs/morphology-feature-reference.md）
_KEEP_FEATURES = frozenset({
    "neck_root_distance_3d", "neck_root_anterior_diff",
    "neck_root_vertical_diff", "neck_root_slope_angle",
    "shoulder_transition_distance_3d", "shoulder_transition_anterior_diff",
    "shoulder_transition_vertical_diff", "shoulder_transition_slope_angle",
    "scapular_peaks_distance_3d", "scapular_peaks_anterior_diff",
    "scapular_peaks_vertical_diff", "scapular_peaks_slope_angle",
    "axilla_distance_3d", "axilla_anterior_diff",
    "axilla_vertical_diff", "axilla_slope_angle",
    "waist_distance_3d", "waist_anterior_diff",
    "waist_vertical_diff", "waist_slope_angle",
    "spine_neck_scapular_length", "spine_neck_scapular_angle_vertical",
    "spine_neck_scapular_lateral_deviation",
    "spine_scapular_axilla_length", "spine_scapular_axilla_angle_vertical",
    "spine_axilla_thoracic_angle_vertical",
    "spine_waist_waistlower_len_ratio", "spine_waist_waistlower_lateral_deviation",
    "spine_curvature_neck_scapular_vs_waist_waistlower",
    "width_waist_axilla_ratio",
    "trunk_length_ratio",
})


def _pair_midpoint(landmarks: dict, name: str) -> np.ndarray:
    """左右语义键 → 中点 (3,)。"""
    return np.mean(
        np.array([landmarks[f"{name}_L"], landmarks[f"{name}_R"]], dtype=np.float64),
        axis=0,
    )


def _compute_trunk_length(landmarks: dict) -> float:
    """计算躯干长度 = neck_root 平均 Y - waist 平均 Y。"""
    nr: np.ndarray = _pair_midpoint(landmarks, "neck_root")
    wa: np.ndarray = _pair_midpoint(landmarks, "waist")
    return float(nr[1] - wa[1])


def _compute_widths(landmarks: dict) -> dict[str, float]:
    """计算 waist-to-axilla 宽度比（唯一保留的宽度特征）。"""
    ax_w = float(np.linalg.norm(np.asarray(landmarks["axilla_R"]) - np.asarray(landmarks["axilla_L"])))
    wa_w = float(np.linalg.norm(np.asarray(landmarks["waist_R"]) - np.asarray(landmarks["waist_L"])))
    return {"width_waist_axilla_ratio": float(wa_w / ax_w) if ax_w > 0 else 0.0}


def _compute_asymmetry(landmarks: dict) -> dict[str, float]:
    """计算肩部斜度（唯一保留的不对称指标）。"""
    nr: np.ndarray = np.array([landmarks["neck_root_L"], landmarks["neck_root_R"]], dtype=np.float64)
    return {
        "shoulder_slope": float(abs(nr[1, 1] - nr[0, 1])),
    }


def extract_morphology(
    landmarks: dict,
    height: float | None = None,
) -> dict[str, float]:
    """从 landmarks dict 提取形态学特征（固定 31D）。

    只输出 ``_KEEP_FEATURES`` 中列出的特征。
    删除的特征（_v_diff_ratio、_len_ratio、waist_lower_*、shoulder_slope 等）
    见 `.claude/docs/morphology-feature-reference.md`。

    Args:
        landmarks: ``extract_landmarks()`` 返回的 landmark 字典。
        height: 身高（mm），可选。提供时计算 ``trunk_length_ratio``。

    Returns:
        dict: 扁平化的特征名→值字典，固定 31 个特征。
    """
    trunk_length: float = _compute_trunk_length(landmarks)
    features: dict[str, float] = {}

    features.update(extract_neck_root(landmarks, trunk_length))
    features.update(extract_shoulder(landmarks, trunk_length))
    features.update(extract_scapular(landmarks, trunk_length))
    features.update(extract_axilla(landmarks, trunk_length))
    features.update(extract_waist(landmarks, trunk_length))
    features.update(extract_spine(landmarks, height))

    features.update(_compute_widths(landmarks))
    features.update(_compute_asymmetry(landmarks))

    features["trunk_length"] = trunk_length
    if height is not None and height > 0:
        features["trunk_length_ratio"] = float(trunk_length / height)
    else:
        features["trunk_length_ratio"] = 0.0

    # 只保留 _KEEP_FEATURES 中的特征
    return {k: v for k, v in features.items() if k in _KEEP_FEATURES}
