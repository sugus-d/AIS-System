"""modeling.ensemble_boundary_features — per-class blend、边界钳制、分类器测试。

拆分后补测：特征层的纯变换逻辑（apply_perclass_blend / apply_boundary_clamps /
boundary_oof_probs / fit_boundary_classifiers）。build_refit_ai_feature /
build_ridge_ai_feature 依赖真实 region CSV（文件 I/O），此处不测。
"""

from __future__ import annotations

import numpy as np
import pytest

from modeling.ensemble_boundary_features import (
    apply_boundary_clamps,
    apply_perclass_blend,
    boundary_oof_probs,
    fit_boundary_classifiers,
)


class TestApplyPerclassBlend:
    def test_weighted_blend(self):
        """per-class α blend + Ridge-AI 加权。

        c7=20, ai8=10 → b0=0.48×20+0.52×10=14.8 → digitize(14.8, [10,20,40])=1（Mild）
        → α=0.7 → pbase=0.7×20+0.3×10=17 → final=0.6×17+0.4×12=15.0
        """
        result = apply_perclass_blend(
            np.array([20.0]), np.array([10.0]), np.array([12.0]),
            perclass_alpha=(0.8, 0.7, 0.48, 0.5), beta=0.6, alpha_base=0.48,
        )
        assert result[0] == pytest.approx(15.0)

    def test_class_boundaries_switch_alpha(self):
        """b0 跨类边界时 α 切换（Severe 类 α=0.5 → 多信 AI 分量）。"""
        result = apply_perclass_blend(
            np.array([80.0]), np.array([10.0]), np.array([50.0]),
            perclass_alpha=(0.8, 0.7, 0.48, 0.5), beta=1.0,  # β=1 只看 pbase
        )
        # b0=0.48×80+0.52×10=43.6 ≥ 40 → digitize=3（Severe）→ α=0.5 → 0.5×80+0.5×10=45
        assert result[0] == pytest.approx(45.0)


class TestApplyBoundaryClamps:
    def test_upper_clamp_when_strong_p20(self):
        """Mild 区间 + P(y>20) 强 → 上钳到 tgt20。"""
        result = apply_boundary_clamps(np.array([15.0]), np.array([0.1]), np.array([0.9]))
        assert result[0] == 20.5

    def test_lower_clamp_when_weak_p10(self):
        """Mild 区间 + P(y>10) 弱 → 下钳到 tgt10。"""
        result = apply_boundary_clamps(np.array([15.0]), np.array([0.1]), np.array([0.1]))
        assert result[0] == 9.5

    def test_no_clamp_when_no_evidence(self):
        """无钳制证据 → 保持原值。"""
        result = apply_boundary_clamps(np.array([15.0]), np.array([0.9]), np.array([0.1]))
        assert result[0] == 15.0

    def test_outside_mild_range_untouched(self):
        """不在 Mild 区间（≥20）不参与钳制。"""
        result = apply_boundary_clamps(np.array([25.0]), np.array([0.1]), np.array([0.9]))
        assert result[0] == 25.0


def _balanced_binary_data(n: int = 30, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """构造二值化后两类平衡的数据（交替标签，任何折内都两 class）。

    交替标签保证 KFold 任意折内 y>10 与 y≤10 均存在（Logistic 二分类要求）；
    x0 与标签强相关（区分度来源），其余列纯噪声。
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    high = (np.arange(n) % 2) == 0  # 交替高/低
    X[:, 0] += np.where(high, 3.0, -3.0)
    y = np.where(high, 25.0, 5.0)
    return X, y


class TestBoundaryOofProbs:
    def test_probability_shape_and_range(self):
        """OOF 概率形状与 (N,) 且落在 [0,1]。"""
        X, y = _balanced_binary_data()
        probs = boundary_oof_probs(X, y, n_splits=3)
        assert set(probs) == {"p10", "p20"}
        for values in probs.values():
            assert values.shape == (len(y),)
            assert values.min() >= 0.0 and values.max() <= 1.0

    def test_probability_separates_classes(self):
        """P(y>10) 对 y>10 的样本应显著高于 y<10 的样本（特征可分）。"""
        X, y = _balanced_binary_data()
        probs = boundary_oof_probs(X, y, n_splits=3)
        high = probs["p10"][y > 10]
        low = probs["p10"][y <= 10]
        assert high.mean() - low.mean() > 0.5


class TestFitBoundaryClassifiers:
    def test_classifier_structure(self):
        """返回 p10/p20 各含 scaler + clf。"""
        X, y = _balanced_binary_data()
        classifiers = fit_boundary_classifiers(X, y)
        assert set(classifiers) == {"p10", "p20"}
        for pkg in classifiers.values():
            assert hasattr(pkg["scaler"], "transform")
            assert hasattr(pkg["clf"], "predict_proba")
