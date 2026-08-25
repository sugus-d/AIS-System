"""modeling.ensemble — AI 特征构建、加权集成、AI-LR OOF、校准偏差测试。

拆分后补测：AI 分量层的纯函数（build_ai_feature / build_ensemble_preds /
fit_ai_linear_oof）与 ensemble_train 的 _bias_from_oof。
不触碰文件 I/O 路径（_load_ai_feature / _fit_ai_linear 依赖真实 CSV）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeling.ensemble import build_ai_feature, build_ensemble_preds, fit_ai_linear_oof
from modeling.ensemble_train import _bias_from_oof


class TestBuildAiFeature:
    def test_formula_evaluation(self):
        """AI 特征 = intercept + Σ coef_j × col_j。"""
        formula = {"feats": ["a", "b"], "coefs": [2.0, 3.0], "intercept": 1.0}
        feature_df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        result = build_ai_feature(feature_df, formula)
        assert result[0] == pytest.approx(9.0)  # 1 + 2×1 + 3×2

    def test_missing_column_raises(self):
        """缺列 fail-fast 抛错（eval_linear_formula 显式检查，防静默错算）。"""
        formula = {"feats": ["a", "b"], "coefs": [2.0, 3.0], "intercept": 1.0}
        feature_df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="线性公式缺特征"):
            build_ai_feature(feature_df, formula)


class TestBuildEnsemblePreds:
    def test_alpha_weighting(self):
        primary = np.array([10.0, 20.0])
        ai = np.array([0.0, 10.0])
        assert np.allclose(build_ensemble_preds(primary, ai, 0.6), [6.0, 16.0])
        assert np.allclose(build_ensemble_preds(primary, ai, 1.0), primary)
        assert np.allclose(build_ensemble_preds(primary, ai, 0.0), ai)


class TestFitAiLinearOof:
    def test_oof_recovers_linear_target(self):
        """线性目标在折外预测下应被完美重建（每折 LR 拟合）。"""
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1
        pred = fit_ai_linear_oof(x, y, n_splits=5)
        assert np.allclose(pred, y, atol=1e-8)


class TestBiasFromOof:
    def test_per_class_median_bias(self):
        """per-class 偏差 = 该类 (pred − true) 的中位数。"""
        y_true = np.array([5.0, 6.0, 15.0, 16.0, 25.0])
        y_pred = y_true + np.array([1.0, 2.0, 0.5, 1.5, -1.0])
        bias = _bias_from_oof(y_true, y_pred, [0, 10, 20, 40, np.inf])
        assert bias[0] == pytest.approx(1.5)   # median(1, 2)
        assert bias[1] == pytest.approx(1.0)   # median(0.5, 1.5)
        assert bias[2] == pytest.approx(-1.0)

    def test_empty_class_skipped(self):
        """无样本的类别不产生偏差键。"""
        y_true = np.array([25.0])
        y_pred = np.array([24.0])
        bias = _bias_from_oof(y_true, y_pred, [0, 10, 20, 40, np.inf])
        assert bias == {2: -1.0}
