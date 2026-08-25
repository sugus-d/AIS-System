"""共享特征列名定义 — 各选择器从这里导入，避免重复。"""

from __future__ import annotations

# 形态学特征 (14 项，全方案共用)
MORPH_COLS_14: list[str] = [
    "neck_root_vertical_diff", "neck_root_slope_angle",
    "shoulder_transition_vertical_diff",
    "scapular_peaks_anterior_diff", "scapular_peaks_vertical_diff",
    "scapular_peaks_slope_angle",
    "axilla_anterior_diff", "axilla_vertical_diff",
    "waist_distance_3d",
    "spine_P0_P1_length", "spine_P3_P4_angle_vertical",
    "spine_curvature_P0P1_vs_P3P4",
    "waist_Y_asymmetry", "axilla_Y_asymmetry",
]

# 区域特征 (25 项，全方案共用)
REGION_COLS_25: list[str] = [
    "wa_wl_p0_p5_normal_angle", "ax_wa_p0_p3_normal_angle",
    "wl_p2_p4_normal_angle__pw", "sp_wa_p0_p1_normal_vector_cos",
    "nr_sp_p0_p2_height", "nr_p0_p5_height",
    "wa_p5_p3_mean_curv__pw", "nr_sp_p0_p1_roughness",
    "st_ax_p0_p1_normal_vector_cos", "ax_wa_p2_p5_normal_vector_cos__pw",
    "ax_wl_p5_p4_normal_angle__pw", "st_ax_p1_p2_normal_angle",
    "wa_wl_p0_p4_gauss_curv", "wa_p5_p4_mean_curv",
    "st_sp_p3_p4_height", "ax_p0_p1_roughness__pw",
    "nr_sp_p1_p3_normal_angle", "sp_ax_p5_p3_roughness__pw",
    "ax_wa_p0_p1_mean_curv", "wl_p1_p5_normal_angle",
    "st_sp_p3_p4_roughness", "sp_ax_p5_p3_gauss_curv",
    "wa_p0_p3_normal_angle",
]


def compute_block_slices(cols: list[str], groups: list[tuple[str, set[str]]]) -> dict[str, slice]:
    """根据列顺序，按已知分组构建 block_slices。"""
    slices: dict[str, slice] = {}
    pos = 0
    for gname, gset in groups:
        count = sum(1 for c in cols[pos:] if c in gset)
        if count == 0:
            continue
        slices[gname] = slice(pos, pos + count)
        pos += count
    return slices
