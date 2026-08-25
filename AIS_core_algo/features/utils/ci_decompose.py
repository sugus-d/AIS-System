"""CI 反解与特征归组 — 将 CI 合成特征的贡献按类别/区域反解回基础特征。

供特征重要性分析（commands/export/analyze.py）、SHAP 瀑布图（prediction/
report_waterfall.py、commands/export/charts_waterfall.py）与模型包 CI 公式
加载共用。region 面积几何在 region_areas，展示名在 ci_display，本模块只做
反解核心（CI 公式加载、SHAP/重要性反解、按类别/区域聚合、线性归组）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.utils.region_areas import _get_region_distribution
from utils.paths import COMPOSITE_DIR

COMPOSITE_CSV = COMPOSITE_DIR / "results_compressed.csv"

# CI 特征 → 测量类型映射
CI_MEASURE = {
    "ci_normal_angle_dm":      "normal_angle",
    "ci_normal_vector_cos_pw": "normal_vector_cos",
    "ci_height_pw":            "height",
    "ci_mean_curv_dm":         "mean_curv",
}

_MEASURE_TO_CATEGORY = {
    "normal_angle":      "Normal Angle",
    "normal_vector":     "Normal Angle",
    "normal_vector_cos": "Normal Angle",
    "normal_vector_sin": "Normal Angle",
    "height":            "Height",
    "mean_curv":         "Curvature",
    "gauss_curv":        "Curvature",
    "roughness":         "Roughness",
}
_SHAP_EPS = 1e-6
"""视为零贡献的 SHAP 绝对值阈值。"""


def load_ci_formulas() -> dict:
    """从 CSV 加载 CI 公式元数据 {group: {measure, n_feats, r}}；CSV 缺失返回空 dict。"""
    if not COMPOSITE_CSV.exists():
        return {}
    df = pd.read_csv(COMPOSITE_CSV)
    formulas = {}
    for _, r in df.iterrows():
        if r["n_feats"] == 0:
            continue
        group = r["group"]
        method = "pw" if group.endswith("_pw") else "dm"
        measure = group.replace(f"_{method}", "")
        formulas[group] = {"measure": measure, "n_feats": r["n_feats"], "r": r["full_r"]}
    return formulas


def ci_category(ci_name: str) -> str:
    """将 CI 特征名映射到合并类别。

    v0.1.0 特征名带 ``ci_`` 前缀（ci_mean_curv_dm），manual 方案不带
    （mean_curv_pw/height_dm...）——后者按特征名中的测量词做 fallback 归类，
    否则 CI 贡献在瀑布图/重要性分解中被静默丢弃。
    """
    measure = CI_MEASURE.get(ci_name, "")
    if measure:
        return _MEASURE_TO_CATEGORY.get(measure, "Other")
    for token in ("normal_angle", "normal_vector_cos", "normal_vector",
                  "gauss_curv", "mean_curv", "roughness", "height"):
        if token in ci_name:
            return _MEASURE_TO_CATEGORY.get(token, "Other")
    return "Other"


def decompose_ci_shap(shap_values_row: np.ndarray, feature_names: list[str],
                      ci_indices: list[int], group_names: list[str]) -> dict[str, float]:
    """将 CI 特征 SHAP 值分解到类别级贡献。

    Args:
        shap_values_row: 单 subject 的 1D SHAP 值。
        feature_names: 特征名列表。
        ci_indices: feature_names 中 CI 特征的索引。
        group_names: 全部类别名的有序列表。

    Returns:
        类别名 → SHAP 贡献（含 CI 反解）。未归入任何 group_name 的贡献收进
        "Other"（绝不静默丢弃），保证 sum(contrib) == sum(shap_values_row)。
        仅返回非零贡献类别。
    """
    result = {g: 0.0 for g in group_names}
    other = 0.0

    # 直接 region 特征聚合
    for i, name in enumerate(feature_names):
        if i in ci_indices:
            continue
        g = _assign_group(name)
        if g in result:
            result[g] += shap_values_row[i]
        else:
            other += shap_values_row[i]

    # CI 特征分解
    for ci_idx in ci_indices:
        ci_name = feature_names[ci_idx]
        cat = ci_category(ci_name)
        if cat in result:
            result[cat] += shap_values_row[ci_idx]
        else:
            other += shap_values_row[ci_idx]

    # 未归入任何已知类别的贡献收进 "Other"（如 ci10_normal/ci20_mild 单目标 CI 特征），
    # 保证聚合完备——瀑布图 Prediction 与真实模型输出一致，不被静默丢弃。
    if abs(other) > _SHAP_EPS:
        result["Other"] = other
    return {k: v for k, v in result.items() if abs(v) > _SHAP_EPS}


def decompose_waterfall(
    shap_row: np.ndarray,
    expected_val: float,
    feature_names: list[str],
    ci_feature_names: set[str] | list[str] | tuple[str, ...],
    group_names: list[str],
) -> tuple[dict[str, float], float]:
    """单行 SHAP 分解 → (按类别聚合贡献, pred_val)。

    pred_val 用 SHAP 可加性（expected + sum(shap)）取**真实模型输出**，而非
    expected + sum(contrib)——decompose_ci_shap 已保证 sum(contrib)==sum(shap)，
    两者一致；但直接用 sum(shap) 最稳，任何聚合改动都不会让瀑布图显示值偏离模型。

    Args:
        shap_row: (N,) 单 subject 的 SHAP 贡献。
        expected_val: SHAP 基线（explainer.expected_value）。
        feature_names: 模型特征名（与 shap_row 同序）。
        ci_feature_names: CI 合成特征名集合（各方案命名不同，由调用方按原口径传入）。
        group_names: 瀑布图类别列表（含 "Other"）。

    Returns:
        (contrib, pred_val)：contrib 为类别聚合贡献；pred_val 为真实模型输出。
    """
    ci_indices = [i for i, name in enumerate(feature_names) if name in ci_feature_names]
    contrib = decompose_ci_shap(shap_row, feature_names, ci_indices, group_names)
    pred_val = expected_val + float(shap_row.sum())
    return contrib, pred_val


def load_ci_formulas_from_package(comp: dict) -> dict[str, dict]:
    """从模型包读统一 CI 公式 {ci_name: {"columns": [...], "coefs": [...]}}。

    优先用训练时保存的 ``comp["ci_formulas"]``；旧模型包无此字段时从
    ``ci_formula_params`` + ``ci10/ci20_params`` 转换（向后兼容，免重训）。
    两种方案（v0.1.0/manual）的 CI 命名不同（如 mean_curv_dm vs mean_curv_pw），
    统一后按 feature_names 实际出现的名字匹配，反解逻辑无方案差异。

    Args:
        comp: 模型包（v8 的内嵌 composite 或 v0.1.0 顶层）。

    Returns:
        {ci_name: {"columns": 基础 region 特征列, "coefs": 对应系数}}。
    """
    if comp.get("ci_formulas"):
        return comp["ci_formulas"]
    out: dict[str, dict] = {}
    for group, p in (comp.get("ci_formula_params") or {}).items():
        out[group] = {"columns": list(p["columns"]), "coefs": [float(c) for c in p["coef"]]}
    for key, name in (("ci10_params", "ci10_normal"), ("ci20_params", "ci20_mild")):
        p = comp.get(key) or {}
        if not p:
            continue
        columns = p["columns"]
        nz = np.asarray(p["nz"], dtype=int)
        out[name] = {"columns": [columns[i] for i in nz], "coefs": [float(c) for c in p["coef"]]}
    return out


def decompose_ci_importance(ci_importance: dict[str, float], ci_formulas: dict[str, dict]) -> dict[str, float]:
    """CI 特征贡献按 |coef| 比例反解回基础 region 特征。

    复用 export 现成口径（commands/export/analyze.py 原实现，已提升至此统一复用）：
    每个 CI 特征的贡献按其公式系数绝对值比例摊回基础特征。不依赖特征值
    （Permutation Importance / SHAP 单行均适用），保证总和平移不变。

    Args:
        ci_importance: {ci_name: 贡献值}（如 SHAP 值或 Permutation Importance）。
        ci_formulas: 统一 CI 公式（见 load_ci_formulas_from_package）。

    Returns:
        {region_feature: 摊回的贡献}。
    """
    decomposed: dict[str, float] = {}
    for ci_name, imp in ci_importance.items():
        formula = ci_formulas.get(ci_name)
        if not formula:
            continue
        columns = formula["columns"]
        coefs = formula["coefs"]
        total_abs = sum(abs(c) for c in coefs)
        if total_abs == 0:
            continue
        for feat, coef in zip(columns, coefs, strict=True):
            share = imp * abs(coef) / total_abs
            decomposed[feat] = decomposed.get(feat, 0.0) + share
    return decomposed


def aggregate_by_measurement(feature_imp: dict[str, float]) -> dict[str, float]:
    """按测量类型聚合特征贡献（Normal Angle/Curvature/Height/Roughness/Morph/Clinical）。

    未知类别回退到 Morph（对齐 analyze 原口径）。
    """
    groups: dict[str, float] = {}
    for name, imp in feature_imp.items():
        group = _assign_group(name)
        if group == "Other":
            group = "Morph"
        groups[group] = groups.get(group, 0.0) + imp
    return groups


def aggregate_by_region(feature_imp: dict[str, float]) -> dict[str, float]:
    """按 5 个解剖区域 + Morph + Clinical 聚合特征贡献。

    region 特征按 UV 面积比例分摊到区域（Shoulder/Scapula/Axilla/Waist/Pelvis）；
    Morph/Clinical 保持原组。
    """
    regions: dict[str, float] = {}
    for name, imp in feature_imp.items():
        group = _assign_group(name)
        if group == "Other":
            group = "Morph"
        if group in ("Morph", "Clinical"):
            regions[group] = regions.get(group, 0.0) + imp
            continue
        # region 特征：按面积比例分摊
        dist = _get_region_distribution(name)
        if dist:
            for band, ratio in dist.items():
                regions[band] = regions.get(band, 0.0) + imp * ratio
        else:
            regions["Morph"] = regions.get("Morph", 0.0) + imp
    return regions


def linear_contrib_to_groups(
    columns: list[str],
    coefs: list[float],
    feature_df: pd.DataFrame,
    group_names: list[str],
    scale: float,
    feature_means: dict[str, float] | None = None,
) -> dict[str, float]:
    """线性公式（pred = intercept + scale × Σ coef_j × col_j）按列归组的特征贡献。

    用于 ensemble 的 AI/AI8/Ridge 分量（LR）：pred 是其基线 + 特征线性组合。
    提供 ``feature_means``（训练集列均值）时每列贡献 = scale × coef_j × (col_j − mean_j)，
    与基线 `intercept + scale×mean(combo)` 对齐（SHAP expected-value 口径）且完全闭合；
    不提供时回退旧口径（不减均值，基线用截距）——兼容未保存均值的旧模型包。

    Args:
        columns: 公式特征列名。
        coefs: 对应系数。
        feature_df: 含特征列值的 DataFrame。
        group_names: 瀑布图类别列表（含 "Other"）。
        scale: 外层 LR 系数（如 ai8_lr.coef）。
        feature_means: {列名: 训练集列均值}；None 时按 0 处理（旧行为）。

    Returns:
        类别 → 贡献（不含 intercept，intercept 由调用方并入组合基线）。
    """
    result: dict[str, float] = {g: 0.0 for g in group_names}
    other = 0.0
    means = feature_means or {}
    for name, coef in zip(columns, coefs, strict=True):
        if name not in feature_df.columns:
            continue
        contrib = scale * coef * (float(feature_df[name].iloc[0]) - float(means.get(name, 0.0)))
        group = _assign_group(name)
        if group in result:
            result[group] += contrib
        else:
            other += contrib
    if abs(other) > _SHAP_EPS:
        result["Other"] = result.get("Other", 0.0) + other
    return {k: v for k, v in result.items() if abs(v) > _SHAP_EPS}


# ── 特征分组（与 analyze 脚本共享） ──

_MORPH_NAMES = {
    "spine_neck_scapular_length", "spine_neck_scapular_angle_vertical", "spine_neck_scapular_lateral_deviation",
    "spine_scapular_axilla_length", "spine_scapular_axilla_angle_vertical",
    "spine_axilla_thoracic_angle_vertical",
    "spine_waist_waistlower_len_ratio", "spine_waist_waistlower_lateral_deviation",
    "spine_curvature_neck_scapular_vs_waist_waistlower",
    "scapular_peaks_slope_angle", "scapular_peaks_distance_3d",
    "scapular_peaks_anterior_diff", "scapular_peaks_vertical_diff",
    "waist_slope_angle", "waist_vertical_diff", "waist_distance_3d", "waist_anterior_diff",
    "neck_root_slope_angle", "neck_root_vertical_diff", "neck_root_distance_3d", "neck_root_anterior_diff",
    "axilla_vertical_diff", "axilla_distance_3d", "axilla_anterior_diff", "axilla_slope_angle",
    "shoulder_transition_vertical_diff", "shoulder_transition_slope_angle",
    "shoulder_transition_distance_3d", "shoulder_transition_anterior_diff",
    "width_waist_axilla_ratio", "trunk_length_ratio",
}


def _assign_group(name: str) -> str:
    if name in ("Height", "Weight", "BMI", "Gender", "Height_x_Weight"):
        return "Clinical"
    if name in _MORPH_NAMES:
        return "Morph"
    if any(x in name for x in ("normal_angle", "normal_vector_cos", "normal_vector", "normal_vector_sin")):
        return "Normal Angle"
    if "gauss_curv" in name or "mean_curv" in name:
        return "Curvature"
    if "_height" in name or name == "height_dm":
        return "Height"
    if "roughness" in name:
        return "Roughness"
    return "Other"
