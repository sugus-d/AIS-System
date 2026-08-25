"""features.synthesis — 线性公式、CI 合成器、指数合成器测试。

覆盖被 prediction 复用的核心合成器：fit 拟合 → to_params/from_params
序列化 → transform 应用的三段链路（round-trip 一致性）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.synthesis import (
    AsymmetrySynthesizer,
    CiFormulaSynthesizer,
    CiTargetSynthesizer,
    eval_linear_formula,
)


class TestEvalLinearFormula:
    def test_evaluation(self):
        df = pd.DataFrame({"a": [1.0], "b": [2.0]})
        assert eval_linear_formula(df, ["a", "b"], [2.0, 3.0], 1.0)[0] == pytest.approx(9.0)

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError, match="线性公式缺特征"):
            eval_linear_formula(df, ["a", "b"], [2.0, 3.0])


def _region_df(n: int = 40, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """构造 region 特征表：3 组测量（mean_curv/height/normal_angle）× dm/pw 变体。

    每组内与 max_cobb 强相关一列（供筛选），其余为噪声。
    """
    rng = np.random.default_rng(seed)
    y = rng.uniform(5, 40, n)
    cols: dict[str, np.ndarray] = {}
    for measure in ("mean_curv", "height", "normal_angle"):
        for method in ("dm", "pw"):
            suffix = f"_{measure}__pw" if method == "pw" else f"_{measure}"
            signal = rng.normal(size=n) + y / 40 * 2  # 与目标相关
            cols[f"r1{suffix}"] = signal
            cols[f"r2{suffix}"] = rng.normal(size=n)  # 噪声
    df = pd.DataFrame(cols)
    df["subject_id"] = [f"S{i:03d}" for i in range(n)]
    df["max_cobb"] = y
    return df, y


class TestCiFormulaSynthesizer:
    def test_fit_transform_roundtrip(self):
        """fit → to_params → from_params → transform 输出一致（序列化无损）。"""
        df_r, y = _region_df()
        synth = CiFormulaSynthesizer(groups=[("mean_curv", "pw"), ("height", "dm")])
        synth.fit(df_r, y)
        params = synth.to_params()
        assert params, "至少应拟合出 1 组 CI 公式"

        rebuilt = CiFormulaSynthesizer.from_params(params)
        out1 = synth.transform(df_r)
        out2 = rebuilt.transform(df_r)
        assert list(out1.columns) == list(out2.columns)
        for group in out1.columns:
            assert np.allclose(out1[group], out2[group])

    def test_missing_column_alignment(self):
        """transform 缺列时按全列索引 zeros 回填（保持维度）。"""
        df_r, y = _region_df()
        synth = CiFormulaSynthesizer(groups=[("mean_curv", "pw")]).fit(df_r, y)
        params = synth.to_params()
        full_cols = list(params["mean_curv_pw"]["columns"])
        assert full_cols

        partial = df_r.drop(columns=[full_cols[0]])
        out = CiFormulaSynthesizer.from_params(params).transform(partial)
        assert len(out) == len(df_r)  # 行数不变，缺列不抛错


class TestCiTargetSynthesizer:
    def test_fit_transform_roundtrip(self):
        """fit → transform_ndarray 输出形状正确，且与目标标签相关。"""
        rng = np.random.default_rng(42)
        region_cols = [f"c{i}" for i in range(6)]
        X = rng.normal(size=(50, 6))
        X[:, 0] += 2 * rng.binomial(1, 0.5, 50)  # 区分度来源
        target = (X[:, 0] > 0).astype(float)

        synth = CiTargetSynthesizer().fit(region_cols, X, target, C=0.1, thr=0.05)
        out = synth.transform_ndarray(X, region_cols)
        assert out.shape == (50,)
        # 与目标正相关（合成 CI 特征应反映目标）
        assert np.corrcoef(out, target)[0, 1] > 0.3

    def test_params_roundtrip(self):
        """to_params/from_params 序列化后 transform 一致。"""
        rng = np.random.default_rng(7)
        region_cols = [f"c{i}" for i in range(4)]
        X = rng.normal(size=(40, 4))
        X[:, 0] += 2 * rng.binomial(1, 0.5, 40)
        target = (X[:, 0] > 0).astype(float)
        synth = CiTargetSynthesizer().fit(region_cols, X, target, C=0.1, thr=0.05)
        rebuilt = CiTargetSynthesizer.from_params(synth.to_params())
        assert np.allclose(synth.transform_ndarray(X, region_cols), rebuilt.transform_ndarray(X, region_cols))


class TestAsymmetrySynthesizer:
    def test_fit_transform_returns_5_indices(self):
        """fit → transform 输出 5 个指数键（curvature/height/nai/ri/ai）。"""
        df_r, _ = _region_df()
        synth = AsymmetrySynthesizer().fit(df_r)
        out = synth.transform(df_r)
        assert set(out) == {"curvature_index", "height_index", "nai", "ri", "ai"}
        for values in out.values():
            assert values.shape == (len(df_r),)

    def test_params_roundtrip(self):
        rebuilt = AsymmetrySynthesizer.from_params(AsymmetrySynthesizer().fit(_region_df()[0]).to_params())
        df_r, _ = _region_df()
        out1 = AsymmetrySynthesizer().fit(df_r).transform(df_r)
        out2 = rebuilt.transform(df_r)
        for key in out1:
            assert np.allclose(out1[key], out2[key])

    def test_missing_region_feature_raises(self):
        df_r, _ = _region_df()
        synth = AsymmetrySynthesizer().fit(df_r)
        with pytest.raises(ValueError, match="缺少 region 特征"):
            synth.transform(df_r.drop(columns=[df_r.columns[0]]))
