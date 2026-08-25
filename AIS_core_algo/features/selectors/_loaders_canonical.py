"""Canonical 特征加载器 — _load_canonical_union_64d / _load_canonical_44d。

从 _loaders.py 拆分而来，保持等价。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.selectors._utils import make_data_dict


def _load_canonical_union_64d(version: str) -> dict:
    """并集：morph_region_ci_37d ∪ canonical_44d = 64D。

    CI 的 canonical 前缀对齐（normal_angle_dm → ci_normal_angle_dm），
    2 个 canonical CI 不存在于提取，去重后 union = morph 16 + region 39 + CI 4 + basic 5 = 64D。
    """
    d = Path("results/extraction/features_extraction") / version
    ds = Path("results/extraction/features_selection") / version
    df_b = pd.read_csv(d / "basic.csv").dropna(subset=["max_cobb"])
    df_m_ext = pd.read_csv(d / "morphology.csv").dropna(subset=["max_cobb"])
    df_r_ext = pd.read_csv(d / "region_asymmetry.csv").dropna(subset=["max_cobb"])
    df_m = pd.read_csv(ds / "morphology.csv").dropna(subset=["max_cobb"])
    df_r = pd.read_csv(ds / "region_asymmetry.csv").dropna(subset=["max_cobb"])
    df_c = pd.read_csv(ds / "ci.csv")

    y = df_b["max_cobb"].values.astype(float)
    basic_cols = [c for c in df_b.columns if c not in ("subject_id", "max_cobb")]
    morph_sel = [c for c in df_m.columns if c not in ("subject_id", "max_cobb")]
    region_sel = [c for c in df_r.columns if c not in ("subject_id", "max_cobb")]
    ci_cols = [c for c in df_c.columns if c != "subject_id"]

    # canonical_47d morph 存在于 v0.1.0 提取的
    canon_morph = ["neck_root_vertical_diff", "neck_root_slope_angle",
        "shoulder_transition_vertical_diff", "scapular_peaks_anterior_diff",
        "scapular_peaks_vertical_diff", "scapular_peaks_slope_angle",
        "axilla_anterior_diff", "axilla_vertical_diff", "waist_distance_3d",
        "spine_neck_scapular_length", "spine_curvature_neck_scapular_vs_waist_waistlower"]
    morph_all = sorted(set(morph_sel) | set(canon_morph))

    # canonical_47d region 存在于 v0.1.0 提取的
    canon_region = ["wa_wl_p0_p5_normal_angle", "ax_wa_p0_p3_normal_angle",
        "wl_p2_p4_normal_angle__pw", "sp_wa_p0_p1_normal_vector_cos",
        "nr_sp_p0_p2_height", "nr_p0_p5_height", "wa_p5_p3_mean_curv__pw",
        "nr_sp_p0_p1_roughness", "st_ax_p0_p1_normal_vector_cos",
        "ax_wa_p2_p5_normal_vector_cos__pw", "ax_wl_p5_p4_normal_angle__pw",
        "st_ax_p1_p2_normal_angle", "wa_wl_p0_p4_gauss_curv", "wa_p5_p4_mean_curv",
        "st_sp_p3_p4_height", "ax_p0_p1_roughness__pw", "nr_sp_p1_p3_normal_angle",
        "sp_ax_p5_p3_roughness__pw", "ax_wa_p0_p1_mean_curv", "wl_p1_p5_normal_angle",
        "ax_p5_p3_normal_angle", "st_sp_p3_p4_roughness", "sp_ax_p5_p3_gauss_curv",
        "wa_p0_p3_normal_angle"]
    region_all = sorted(set(region_sel) | set(canon_region))

    # CI union: canonical CI 前缀对齐到 ci_（只取 v0.1.0 中存在的）
    canon_ci = ["ci_normal_angle_dm", "ci_roughness_pw", "ci_gauss_curv_dm"]
    canon_ci_avail = [c for c in canon_ci if c in ci_cols]
    ci_all = sorted(set(ci_cols) | set(canon_ci_avail))

    feature_names = basic_cols + morph_all + region_all + ci_all
    df_all = df_b[["subject_id", "max_cobb"] + basic_cols].merge(
        df_m_ext[["subject_id"] + morph_all], on="subject_id", how="left")
    df_all = df_all.merge(df_r_ext[["subject_id"] + region_all], on="subject_id", how="left")
    df_all = df_all.merge(df_c[["subject_id"] + ci_all], on="subject_id", how="left")

    X = np.nan_to_num(df_all[feature_names].values.astype(float), nan=0.0)
    return make_data_dict(y, X, None, None, None)


def _load_canonical_44d(version: str) -> dict:
    """加载 canonical 固定参考集，从 v0.1.0 提取数据中取可用特征。

    canonical_47d 原数据仅 N=60，从 v0.1.0 提取（N=122）中取它的 45 个特征，
    去掉 v0.1.0 不存在的 3 个（spine_waist_waistlower_angle_vertical, waist_Y_asymmetry, axilla_Y_asymmetry），
    CI 前缀对齐（normal_angle_dm → ci_normal_angle_dm）后 = 42D。
    """
    d = Path("results/extraction/features_extraction") / version
    ds = Path("results/extraction/features_selection") / version
    df_b = pd.read_csv(d / "basic.csv").dropna(subset=["max_cobb"])
    df_m = pd.read_csv(d / "morphology.csv").dropna(subset=["max_cobb"])
    df_r = pd.read_csv(d / "region_asymmetry.csv").dropna(subset=["max_cobb"])
    df_c = pd.read_csv(ds / "ci.csv")

    y = df_b["max_cobb"].values.astype(float)
    basic_cols = [c for c in df_b.columns if c not in ("subject_id", "max_cobb")]

    # canonical_47d morph list, minus ones not in extraction
    canon_morph = ["neck_root_vertical_diff", "neck_root_slope_angle",
        "shoulder_transition_vertical_diff", "scapular_peaks_anterior_diff",
        "scapular_peaks_vertical_diff", "scapular_peaks_slope_angle",
        "axilla_anterior_diff", "axilla_vertical_diff", "waist_distance_3d",
        "spine_neck_scapular_length", "spine_curvature_neck_scapular_vs_waist_waistlower"]
    morph_cols = [c for c in canon_morph if c in df_m.columns]

    # canonical_47d region list, minus ones not in extraction
    canon_region = ["wa_wl_p0_p5_normal_angle", "ax_wa_p0_p3_normal_angle",
        "wl_p2_p4_normal_angle__pw", "sp_wa_p0_p1_normal_vector_cos",
        "nr_sp_p0_p2_height", "nr_p0_p5_height", "wa_p5_p3_mean_curv__pw",
        "nr_sp_p0_p1_roughness", "st_ax_p0_p1_normal_vector_cos",
        "ax_wa_p2_p5_normal_vector_cos__pw", "ax_wl_p5_p4_normal_angle__pw",
        "st_ax_p1_p2_normal_angle", "wa_wl_p0_p4_gauss_curv", "wa_p5_p4_mean_curv",
        "st_sp_p3_p4_height", "ax_p0_p1_roughness__pw", "nr_sp_p1_p3_normal_angle",
        "sp_ax_p5_p3_roughness__pw", "ax_wa_p0_p1_mean_curv", "wl_p1_p5_normal_angle",
        "ax_p5_p3_normal_angle", "st_sp_p3_p4_roughness", "sp_ax_p5_p3_gauss_curv",
        "wa_p0_p3_normal_angle"]
    region_cols = [c for c in canon_region if c in df_r.columns]

    ci_cols = [c for c in df_c.columns if c != "subject_id"]

    feature_names = basic_cols + morph_cols + region_cols + ci_cols
    df_all = df_b[["subject_id", "max_cobb"] + basic_cols].merge(
        df_m[["subject_id"] + morph_cols], on="subject_id", how="left")
    df_all = df_all.merge(df_r[["subject_id"] + region_cols], on="subject_id", how="left")
    df_all = df_all.merge(df_c, on="subject_id", how="left")

    X = np.nan_to_num(df_all[feature_names].values.astype(float), nan=0.0)
    return make_data_dict(y, X, None, None, None)
