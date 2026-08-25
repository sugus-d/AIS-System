"""prediction.measures — 分级边界、指数公式应用、体征参数包装测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from prediction.feature_pipeline import _compute_indices
from prediction.measures import _compute_body_params
from utils.constants import classify_cobb


class TestClassify:
    @pytest.mark.parametrize(
        ("cobb", "expected"),
        [
            (0.0, "Normal"),
            (9.99, "Normal"),
            (10.0, "Mild"),
            (19.99, "Mild"),
            (20.0, "Moderate"),
            (39.99, "Moderate"),
            (40.0, "Severe"),
            (100.0, "Severe"),
        ],
    )
    def test_boundaries(self, cobb, expected):
        assert classify_cobb(cobb) == expected


class TestComputeIndices:
    def test_formula_application_and_full_names(self):
        """公式应用 + 全称展开（nai→normal_angle_index 等）。"""
        model_pkg = {
            "asymmetry_cols": ["r1", "r2"],
            "asymmetry_scaler": _IdentityScaler(),
            "asymmetry_formulas": {
                "nai": {"feats": ["r1"], "coefs": [2.0]},
                "ri": {"feats": ["r2"], "coefs": [3.0]},
            },
            "asymmetry_ai": {"nai": 0.5, "ri": 0.5},
        }
        feature_df = pd.DataFrame({"r1": [1.0], "r2": [2.0]})
        result = _compute_indices(feature_df, model_pkg)
        assert result == {
            "normal_angle_index": 2.0,
            "roughness_index": 6.0,
            "asymmetric_index": 4.0,
        }

    def test_missing_region_feature_raises(self):
        model_pkg = {
            "asymmetry_cols": ["r1", "r2"],
            "asymmetry_scaler": _IdentityScaler(),
            "asymmetry_formulas": {"nai": {"feats": ["r1"], "coefs": [1.0]}},
            "asymmetry_ai": {},
        }
        feature_df = pd.DataFrame({"r1": [1.0]})
        with pytest.raises(ValueError, match="缺少 region 特征"):
            _compute_indices(feature_df, model_pkg)


class TestComputeBodyParams:
    def test_returns_9_params_with_info(self):
        """合成扁平 18 键 gt → 9 个键，value 为 float 且带 info 描述。"""
        gt = {
            "shoulder_transition_L": [-20.99, 28.82, -492.99],
            "shoulder_transition_R": [169.69, 40.69, -497.76],
            "scapular_peaks_L": [-1.31, -20.59, -477.08],
            "scapular_peaks_R": [121.42, -24.56, -466.93],
            "waist_lower_L": [-49.70, -320.49, -580.91],
            "waist_lower_R": [197.29, -287.13, -566.56],
            "neck_root_spine_point": [0, 0, 0],
            "scapular_spine_point": [1, 1, 1],
            "axilla_spine_point": [2, 2, 2],
            "waist_spine_point": [3, 3, 3],
            "waist_lower_spine_point": [4, 4, 4],
            "thoracic_spine_point": [5, 5, 5],
        }
        result = _compute_body_params(gt, "S001")
        assert len(result) == 9
        for value in result.values():
            assert isinstance(value["value"], float)
            assert "info" in value
        assert result["Sh.W"]["value"] > 0


class _IdentityScaler:
    """桩：原样返回输入（替代 sklearn scaler 的 transform）。"""

    def transform(self, X):
        return X
