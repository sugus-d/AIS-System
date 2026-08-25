"""features.utils.ci_decompose — CI 反解核心、归组、展示名测试。

覆盖拆分后三个模块：ci_decompose（反解核心）、ci_display（展示名）。
region_areas 的蒙特卡洛面积缓存较慢，此处只测纯函数分支，不触发缓存构建。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.utils.ci_decompose import (
    _assign_group,
    aggregate_by_measurement,
    aggregate_by_region,
    ci_category,
    decompose_ci_importance,
    decompose_ci_shap,
    linear_contrib_to_groups,
    load_ci_formulas_from_package,
)
from features.utils.ci_display import feature_display_name

# 瀑布图标准类别列表（含 Other）
_GROUP_NAMES = ["Normal Angle", "Curvature", "Height", "Roughness", "Morph", "Clinical", "Other"]


class TestCiCategory:
    def test_v010_prefixed_names(self):
        assert ci_category("ci_normal_angle_dm") == "Normal Angle"
        assert ci_category("ci_normal_vector_cos_pw") == "Normal Angle"
        assert ci_category("ci_height_pw") == "Height"
        assert ci_category("ci_mean_curv_dm") == "Curvature"

    def test_manual_unprefixed_fallback(self):
        # manual 方案不带 ci_ 前缀，按测量词 fallback 归类
        assert ci_category("mean_curv_pw") == "Curvature"
        assert ci_category("height_dm") == "Height"
        assert ci_category("normal_vector_cos_pw") == "Normal Angle"
        assert ci_category("roughness_pw") == "Roughness"

    def test_unknown_returns_other(self):
        assert ci_category("unrelated_feature") == "Other"


class TestAssignGroup:
    def test_clinical(self):
        assert _assign_group("Height") == "Clinical"
        assert _assign_group("Gender") == "Clinical"

    def test_morph(self):
        assert _assign_group("waist_slope_angle") == "Morph"

    def test_measure_groups(self):
        assert _assign_group("wa_wl_p0_p4_normal_angle__pw") == "Normal Angle"
        assert _assign_group("r_side_mean_curv_pw") == "Curvature"
        assert _assign_group("x_height_dm") == "Height"
        assert _assign_group("roughness_pw") == "Roughness"

    def test_unknown_returns_other(self):
        assert _assign_group("zzz") == "Other"


class TestDecomposeCiShap:
    def test_decomposes_ci_and_keeps_total(self):
        """CI 特征归到类别；未归类特征收进 Other，总贡献守恒。"""
        feature_names = ["r1", "r2", "ci_mean_curv_dm", "ci10_normal"]
        shap_row = np.array([1.0, 2.0, 3.0, 4.0])
        ci_indices = [2, 3]
        result = decompose_ci_shap(shap_row, feature_names, ci_indices, _GROUP_NAMES)
        # r1/r2/ci10_normal 未归入已知组 → Other；ci_mean_curv_dm → Curvature
        assert result["Curvature"] == pytest.approx(3.0)
        assert result["Other"] == pytest.approx(7.0)
        # 聚合完备：sum(contrib) == sum(shap)
        assert sum(result.values()) == pytest.approx(sum(shap_row))

    def test_empty_ci_indices(self):
        feature_names = ["a", "b"]
        result = decompose_ci_shap(np.array([0.5, 0.5]), feature_names, [], _GROUP_NAMES)
        # a/b 未匹配 → Other
        assert result["Other"] == pytest.approx(1.0)


class TestDecomposeCiImportance:
    def test_share_by_abs_coef_ratio(self):
        """贡献按 |coef| 比例摊回基础特征。"""
        ci_formulas = {"ci_mean_curv_dm": {"columns": ["a", "b", "c"], "coefs": [1.0, 2.0, 1.0]}}
        result = decompose_ci_importance({"ci_mean_curv_dm": 4.0}, ci_formulas)
        # total_abs=4：a→1.0, b→2.0, c→1.0
        assert result == {"a": 1.0, "b": 2.0, "c": 1.0}

    def test_skips_missing_formula_and_zero_coef(self):
        assert decompose_ci_importance({"unknown": 5.0}, {}) == {}
        assert decompose_ci_importance({"ci": 5.0}, {"ci": {"columns": ["a"], "coefs": [0.0]}}) == {}


class TestLinearContribToGroups:
    def test_contribution_with_means(self):
        """提供 feature_means 时按 (col − mean) 口径，与 SHAP expected-value 对齐。"""
        feature_df = pd.DataFrame({"r1": [3.0]})
        result = linear_contrib_to_groups(["r1"], [1.0], feature_df, _GROUP_NAMES, 2.0, {"r1": 1.0})
        assert result["Other"] == pytest.approx(4.0)  # 2×1×(3−1)

    def test_without_means_fallback(self):
        """无 feature_means 时按旧口径（不减均值）。"""
        feature_df = pd.DataFrame({"r1": [3.0]})
        result = linear_contrib_to_groups(["r1"], [1.0], feature_df, _GROUP_NAMES, 2.0)
        assert result["Other"] == pytest.approx(6.0)  # 2×1×(3−0)

    def test_skips_missing_column(self):
        feature_df = pd.DataFrame({"other": [1.0]})
        result = linear_contrib_to_groups(["r1"], [1.0], feature_df, _GROUP_NAMES, 1.0)
        assert result == {}


class TestAggregate:
    def test_by_measurement_falls_back_to_morph(self):
        """未知类别回退到 Morph（对齐 analyze 原口径）。"""
        result = aggregate_by_measurement({"waist_slope_angle": 1.0, "zzz": 2.0})
        assert result == {"Morph": 3.0}

    def test_by_region_keeps_morph_clinical(self):
        """Morph/Clinical 保持原组（不触发 region 面积缓存）。"""
        result = aggregate_by_region({"waist_slope_angle": 1.0, "Height": 2.0})
        assert result == {"Morph": 1.0, "Clinical": 2.0}


class TestLoadCiFormulasFromPackage:
    def test_uses_saved_ci_formulas(self):
        comp = {"ci_formulas": {"g1": {"columns": ["a"], "coefs": [1.0]}}}
        assert load_ci_formulas_from_package(comp) == {"g1": {"columns": ["a"], "coefs": [1.0]}}

    def test_legacy_conversion(self):
        """旧模型包无 ci_formulas 时从 ci_formula_params + ci10/20_params 转换。"""
        comp = {
            "ci_formula_params": {"g1": {"columns": ["a", "b"], "coef": [1.0, 2.0]}},
            # coef 保存时已按 nz 过滤（CiTargetSynthesizer.fit: lr.coef_[0][nz]）
            "ci10_params": {"columns": ["c", "d"], "nz": [1], "coef": [3.0]},
        }
        result = load_ci_formulas_from_package(comp)
        assert result["g1"] == {"columns": ["a", "b"], "coefs": [1.0, 2.0]}
        # ci10: nz=[1] → columns[1]="d"，coef 已是过滤后的 [3.0]
        assert result["ci10_normal"] == {"columns": ["d"], "coefs": [3.0]}


class TestFeatureDisplayName:
    def test_priority_clinical_ci_morph(self):
        """展示名优先级：临床 → CI → 形态（region 分支不触发慢缓存）。"""
        assert feature_display_name("Gender") == "Gender"
        assert feature_display_name("ci10_normal") == "Normal-Targeted Asymmetric Index"
        assert feature_display_name("waist_slope_angle") == "Waist Slope Angle"
        assert feature_display_name("Height_x_Weight") == "H×W"
