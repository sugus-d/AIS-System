"""特征展示名 — CI/形态/临床/region 特征的瀑布图与重要性图可读标签。

从 ci_decompose 拆出（等价重构）：展示名是独立展示层，与 CI 反解解耦；
region 归属判定复用 region_areas._get_region_display。
"""

from __future__ import annotations

from features.utils.region_areas import _get_region_display

# CI 特征展示标签（v0.1.0 与 manual 两种命名统一映射）
_CI_DISPLAY = {
    "ci_normal_angle_dm": "Normal Angle Asymmetric Index",
    "ci_normal_vector_cos_pw": "Normal Angle Asymmetric Index",
    "ci_height_pw": "Height Asymmetric Index",
    "ci_mean_curv_dm": "Curvature Asymmetric Index",
    "normal_angle_pw": "Normal Angle Asymmetric Index",
    "normal_vector_cos_pw": "Normal Angle Asymmetric Index",
    "height_dm": "Height Asymmetric Index",
    "mean_curv_dm": "Curvature Asymmetric Index",
    "ci10_normal": "Normal-Targeted Asymmetric Index",
    "ci20_mild": "Mild-Targeted Asymmetric Index",
}

_MEASURE_LABEL = {
    "normal_angle": "Normal Angle",
    "normal_vector_cos": "Normal Angle",
    "normal_vector": "Normal Angle",
    "gauss_curv": "Curvature",
    "mean_curv": "Curvature",
    "roughness": "Roughness",
    "height": "Height",
}

_MORPH_LABEL = {
    "spine_neck_scapular_length": "Spine Length (Neck-Scapula)",
    "spine_neck_scapular_angle_vertical": "Spine Vertical Angle (Neck-Scapula)",
    "spine_neck_scapular_lateral_deviation": "Spine Lateral Deviation (Neck-Scapula)",
    "spine_scapular_axilla_length": "Spine Length (Scapula-Axilla)",
    "spine_scapular_axilla_angle_vertical": "Spine Vertical Angle (Scapula-Axilla)",
    "spine_axilla_thoracic_angle_vertical": "Spine Vertical Angle (Axilla-Waist)",
    "spine_waist_waistlower_len_ratio": "Spine Length Ratio (Waist-Pelvis)",
    "spine_waist_waistlower_lateral_deviation": "Spine Lateral Deviation (Waist-Pelvis)",
    "spine_curvature_neck_scapular_vs_waist_waistlower": "Spine Curvature Ratio (Neck vs Waist)",
    "neck_root_slope_angle": "Neck Slope Angle",
    "neck_root_vertical_diff": "Neck Vertical Diff",
    "shoulder_transition_vertical_diff": "Shoulder Vertical Diff",
    "shoulder_transition_slope_angle": "Shoulder Slope Angle",
    "scapular_peaks_slope_angle": "Scapular Slope Angle",
    "scapular_peaks_distance_3d": "Scapular Distance",
    "scapular_peaks_anterior_diff": "Scapular Anterior Diff",
    "axilla_vertical_diff": "Axilla Vertical Diff",
    "waist_slope_angle": "Waist Slope Angle",
    "waist_vertical_diff": "Waist Vertical Diff",
    "waist_distance_3d": "Waist Distance",
}

_CLINICAL_LABEL = {
    "Height": "Height", "Weight": "Weight", "BMI": "BMI",
    "Gender": "Gender", "Height_x_Weight": "H×W",
}


def feature_display_name(name: str) -> str:
    """将原始特征名映射为可读标签。

    优先级：临床 → CI → 形态 → region 特征（"{区域} {测量}"）。
    region 特征区域归属用面积占比判定（region_areas._get_region_display）。
    """
    # 临床特征
    if name in _CLINICAL_LABEL:
        return _CLINICAL_LABEL[name]

    # CI 特征
    if name in _CI_DISPLAY:
        return _CI_DISPLAY[name]

    # 形态特征
    if name in _MORPH_LABEL:
        return _MORPH_LABEL[name]

    # region 不对称特征：测量类型 + 区域前缀
    fn_lower = name.lower()
    measure = "Morph"
    for pat, lbl in _MEASURE_LABEL.items():
        if pat in fn_lower:
            measure = lbl
            break

    region = _get_region_display(name)
    if region:
        return f"{region} {measure}"
    return measure
