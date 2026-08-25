"""prediction.feature_pipeline — CI 合成、Gender 清洗、per-class 校准、单模型预测测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prediction.feature_pipeline import (
    _add_ci_features,
    _calibrate,
    _compute_ci_target,
    _predict,
    _prepare_feature_df,
)


class _IdentityScaler:
    """桩：原样返回输入（替代 sklearn scaler 的 transform）。"""

    def transform(self, X):
        return X


class _StubModel:
    """桩：固定输出 cobb 值。"""

    def __init__(self, value: float):
        self._value = value

    def predict(self, X):
        return np.array([[self._value]])


class TestComputeCiTarget:
    def test_formula_application(self):
        params = {
            "columns": ["f1", "f2", "f3"],
            "scaler": _IdentityScaler(),
            "nz": [0, 2],
            "coef": [1.0, 2.0],
        }
        feature_df = pd.DataFrame({"f1": [1.0], "f2": [99.0], "f3": [3.0]})
        # X[:, [0,2]] @ [1,2] = 1*1 + 3*2 = 7
        assert _compute_ci_target(feature_df, params) == pytest.approx(7.0)


class TestAddCiFeatures:
    def test_synthesizes_ci_groups_and_targets(self):
        model_pkg = {
            "feature_names": ["f1", "g1", "ci10_normal", "ci20_mild"],
            "ci_formula_params": {
                "g1": {"columns": ["f1", "f2"], "mean": [1.0, 2.0], "std": [1.0, 1.0], "coef": [2.0, 3.0]},
            },
            "ci10_params": {"columns": ["f1"], "scaler": _IdentityScaler(), "nz": [0], "coef": [5.0]},
            "ci20_params": {"columns": ["f1"], "scaler": _IdentityScaler(), "nz": [0], "coef": [7.0]},
        }
        feature_df = pd.DataFrame({"f1": [3.0], "f2": [4.0]})
        result = _add_ci_features(feature_df, model_pkg)
        # g1 = ((3-1)/1)*2 + ((4-2)/1)*3 = 4 + 6 = 10；ci10 = 5*3 = 15；ci20 = 7*3 = 21
        assert result["g1"].iloc[0] == pytest.approx(10.0)
        assert result["ci10_normal"].iloc[0] == pytest.approx(15.0)
        assert result["ci20_mild"].iloc[0] == pytest.approx(21.0)

    def test_missing_group_in_feature_names_skipped(self):
        """ci_formula_params 中的组不在 feature_names 里 → 跳过（不合成）。"""
        model_pkg = {"feature_names": ["f1"], "ci_formula_params": {"g1": {}}}
        feature_df = pd.DataFrame({"f1": [1.0]})
        result = _add_ci_features(feature_df, model_pkg)
        assert "g1" not in result.columns


class TestPrepareFeatureDf:
    def test_string_gender_coerced_to_zero(self):
        df = pd.DataFrame({"Gender": ["Female"], "f1": [1.0]})
        result = _prepare_feature_df(df, {"feature_names": []})
        assert result["Gender"].iloc[0] == 0.0

    def test_numeric_gender_kept(self):
        df = pd.DataFrame({"Gender": ["1"], "f1": [1.0]})
        result = _prepare_feature_df(df, {"feature_names": []})
        assert result["Gender"].iloc[0] == 1.0


class TestCalibrate:
    def test_int_and_str_bias_keys(self):
        """bias 键 int（v0.1.0）与 str（v1.0.0）都兼容。"""
        preds = np.array([15.0])
        assert _calibrate(preds, {1: 2.0})[0] == pytest.approx(13.0)  # clip(15-2, 10, 20)
        assert _calibrate(preds, {"1": 3.0})[0] == pytest.approx(12.0)

    def test_missing_bias_no_change(self):
        preds = np.array([15.0])
        assert _calibrate(preds, {})[0] == pytest.approx(15.0)


class TestPredict:
    def test_single_model(self):
        model_pkg = {
            "feature_names": ["f1"],
            "scaler": _IdentityScaler(),
            "model": _StubModel(30.0),
            "transform_target": False,
        }
        feature_df = pd.DataFrame({"f1": [1.0]})
        assert _predict(feature_df, model_pkg) == {"cobb": 30.0, "severity": "Moderate"}

    def test_empty_feature_names_raises(self):
        model_pkg = {"feature_names": None, "scaler": _IdentityScaler(), "model": _StubModel(1.0)}
        feature_df = pd.DataFrame({"f1": [1.0]})
        with pytest.raises(ValueError, match="feature_names 为空"):
            _predict(feature_df, model_pkg)

    def test_missing_feature_raises(self):
        model_pkg = {
            "feature_names": ["f1", "f2"],
            "scaler": _IdentityScaler(),
            "model": _StubModel(1.0),
            "transform_target": False,
        }
        feature_df = pd.DataFrame({"f1": [1.0]})
        with pytest.raises(ValueError, match="特征缺失"):
            _predict(feature_df, model_pkg)
