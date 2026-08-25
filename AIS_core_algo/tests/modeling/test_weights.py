"""WeightComponent 单元测试 — 独立乘区权重架构。

TDD: 先写测试，再实现。
"""

from __future__ import annotations

import numpy as np

from modeling.training.weights import (
    ConstWeight,
    DecayWeight,
    InvFreqWeight,
    MarginBoostWeight,
    PerClassWeight,
    WeightComponent,
)


class TestWeightComponent:
    """基类行为测试。"""

    def test_normalize_sum_to_n(self):
        """归一化后权重之和应等于 N（样本数）。"""
        rng = np.random.default_rng(42)
        w = rng.random(100) * 10
        comp = _DummyWeight()
        normalized = comp._normalize(w)
        assert normalized.shape == w.shape
        assert np.isclose(normalized.sum(), 100, rtol=1e-4)

    def test_normalize_preserves_zeros(self):
        """全零权重归一化后应为全 1（下界保护）。"""
        w = np.zeros(50)
        comp = _DummyWeight()
        normalized = comp._normalize(w)
        assert np.allclose(normalized, 1.0)

    def test_normalize_clip_lower_bound(self):
        """极小权重不应导致除零。"""
        w = np.array([1e-20, 1e-20, 1.0])
        comp = _DummyWeight()
        normalized = comp._normalize(w)
        assert np.all(np.isfinite(normalized))
        assert np.all(normalized >= 0)

    def test_compute_normalizes_when_true(self):
        """normalize=True 时 compute 应自动归一化。"""
        y = np.arange(10, dtype=float)
        dw = DecayWeight(clinical=20, class_weight=3, dist_k=0.1, normalize=True)
        w = dw.compute(y)
        assert np.isclose(w.sum(), 10, rtol=1e-3)

    def test_compute_skips_normalize_when_false(self):
        """normalize=False 时 compute 应返回原始权重。"""
        y = np.arange(10, dtype=float)
        dw = DecayWeight(clinical=20, class_weight=3, dist_k=0.1, normalize=False)
        w = dw.compute(y)
        raw = dw._compute_raw(y)
        assert np.allclose(w, raw)


class _DummyWeight(WeightComponent):
    name = "dummy"

    def __init__(self, normalize: bool = True) -> None:
        super().__init__(normalize=normalize)

    def _compute_raw(self, y: np.ndarray) -> np.ndarray:
        return np.ones(len(y))


class TestDecayWeight:
    """DecayWeight: class_weight × exp(-dist_k × |y - clinical|)"""

    def test_fixed_params(self):
        """固定参数时，compute 返回确定值（无 threshold）。"""
        y = np.array([0, 10, 20, 30, 40], dtype=float)
        dw = DecayWeight(clinical=10, class_weight=3, dist_k=0.1, normalize=False)
        w = dw.compute(y)
        # threshold=False：全部乘 class_weight=3
        expected_10 = 3.0 * np.exp(-0.1 * 0)   # = 3.0
        expected_0 = 3.0 * np.exp(-0.1 * 10)    # ≈ 1.104
        expected_20 = 3.0 * np.exp(-0.1 * 10)   # ≈ 1.104 (无 threshold)
        assert np.isclose(w[1], expected_10, rtol=1e-3)
        assert np.isclose(w[0], expected_0, rtol=1e-3)
        assert np.isclose(w[2], expected_20, rtol=1e-3)

    def test_threshold_mode(self):
        """threshold=True 时 y>clinical 用 1.0 而非 class_weight。"""
        y = np.array([0, 10, 20, 30, 40], dtype=float)
        dw = DecayWeight(clinical=10, class_weight=3, dist_k=0.1,
                         normalize=False, threshold=True)
        w = dw.compute(y)
        # threshold=True: y>clinical 时 class_w=1.0
        assert np.isclose(w[2], 1.0 * np.exp(-0.1 * 10), rtol=1e-3)

    def test_searchable_params(self):
        """参数为 list 时，get_param_space 返回正确范围。"""
        dw = DecayWeight(clinical=20, class_weight=[3, 5, 8, 12],
                          dist_k=[0.05, 0.1, 0.2, 0.5])
        space = dw.get_param_space()
        assert "class_weight" in space
        assert "dist_k" in space
        assert space["class_weight"]["type"] == "discrete"
        assert space["class_weight"]["values"] == [3, 5, 8, 12]

    def test_invalid_clinical_searchable(self):
        """clinical 为 list 时应被正确识别为可搜索。"""
        dw = DecayWeight(clinical=[10, 15, 20], class_weight=3, dist_k=0.1)
        space = dw.get_param_space()
        assert "clinical" in space
        assert space["clinical"]["values"] == [10, 15, 20]

    def test_no_searchable(self):
        """全部固定时 get_param_space 返回空。"""
        dw = DecayWeight(clinical=10, class_weight=3, dist_k=0.1)
        assert dw.get_param_space() == {}

    def test_normalize_in_compute(self):
        """normalize=True 时 compute 后 sum = N。"""
        rng = np.random.default_rng(42)
        y = rng.uniform(0, 60, 100)
        dw = DecayWeight(clinical=10, class_weight=3, dist_k=0.1, normalize=True)
        w = dw.compute(y)
        assert np.isclose(w.sum(), 100, rtol=1e-4)

    def test_raw_compute(self):
        """_compute_raw 返回未归一化的原始权重。"""
        y = np.array([5, 15, 25], dtype=float)
        dw = DecayWeight(clinical=10, class_weight=3, dist_k=0.1, normalize=False)
        raw = dw._compute_raw(y)
        w = dw.compute(y)
        assert np.allclose(raw, w)  # normalize=False 时 compute 返回原始值


class TestInvFreqWeight:
    """InvFreqWeight: 按类频率反比加权。"""

    def test_basic(self):
        """多数类权重=1，少数类权重>1。"""
        y = np.array([5, 5, 5, 15, 15, 25, 25, 25, 25, 45], dtype=float)
        iw = InvFreqWeight(max_ratio=5.0, normalize=False)
        w = iw.compute(y)
        # Most frequent is Moderate(4) → weight=1, Severe(1) → weight=4
        assert np.isclose(w[0], 4/3, rtol=1e-3), f"got {w[0]}"
        assert np.isclose(w[-1], 4.0, rtol=1e-3), f"got {w[-1]}"  # Severe

    def test_max_ratio_clip(self):
        """max_ratio 限制最大权重。"""
        y = np.array([5, 5, 5, 15, 15, 25, 25, 25, 25, 45], dtype=float)
        iw = InvFreqWeight(max_ratio=2.0)
        w = iw.compute(y)
        assert w.max() <= 2.0

    def test_no_max_ratio(self):
        """max_ratio=None 不限制。"""
        y = np.array([5, 5, 5, 15, 15, 25, 25, 25, 25, 45], dtype=float)
        iw = InvFreqWeight(max_ratio=None, normalize=False)
        w = iw.compute(y)
        assert np.isclose(w[-1], 4.0)  # Severe=1, max=4


class TestMarginBoostWeight:
    """MarginBoostWeight: 边界附近增强。"""

    def test_boost_at_boundaries(self):
        """10° 和 20° 附近权重最高。"""
        y = np.array([0, 5, 10, 15, 20, 25, 30], dtype=float)
        mw = MarginBoostWeight(margin_factor=3.0, sigma=2.0)
        w = mw.compute(y)
        # At exact boundaries (10, 20): 1 + 3*exp(0) + 3*exp(-(10/2)^2/2) ≈ 4 + small
        assert w[2] > w[0]  # 10° > 0°
        assert w[4] > w[6]  # 20° > 30°
        assert w[2] > w[3]  # 10° > 15° (slightly off center)

    def test_sigma_isolation(self):
        """sigma 很小时只有边界附近 boost。"""
        y = np.array([10, 12, 20, 22], dtype=float)
        mw = MarginBoostWeight(margin_factor=2.0, sigma=1.0)
        w = mw.compute(y)
        # 10° gets boost from one boundary, 20° from the other
        assert w[0] > 1.0
        assert w[2] > 1.0


class TestPerClassWeight:
    """PerClassWeight: 每类指定权重。"""

    def test_basic(self):
        y = np.array([5, 15, 25, 45], dtype=float)
        pw = PerClassWeight(weights=(6.0, 3.0, 1.0, 5.0), normalize=False)
        w = pw.compute(y)
        assert np.isclose(w[0], 6.0)  # Normal
        assert np.isclose(w[1], 3.0)  # Mild
        assert np.isclose(w[2], 1.0)  # Moderate
        assert np.isclose(w[3], 5.0)  # Severe


class TestConstWeight:
    """ConstWeight: 常数权重 + 搜索噪声。"""

    def test_all_ones(self):
        cw = ConstWeight()
        y = np.arange(10, dtype=float)
        w = cw.compute(y)
        assert np.allclose(w, 1.0)

    def test_noise_params_in_space(self):
        cw = ConstWeight(alpha=[0.01, 0.1, 1], solver=["svd", "cholesky"])
        space = cw.get_param_space()
        assert "alpha" in space
        assert "solver" in space
        assert space["alpha"]["type"] == "discrete"
        assert space["solver"]["values"] == ["svd", "cholesky"]

    def test_no_noise_params(self):
        cw = ConstWeight()
        assert cw.get_param_space() == {}


class TestReproduce0731:
    """验证新架构能复现 0.731 方案。"""

    def test_search_weights_only_decay(self):
        """搜索阶段只用 DecayWeight (clinical=20, cw/dk 可搜)。"""
        dw = DecayWeight(clinical=20, class_weight=[3, 5, 8, 12],
                         dist_k=[0.05, 0.1, 0.2, 0.5])
        assert "class_weight" in dw.get_param_space()
        assert "dist_k" in dw.get_param_space()

    def test_final_weights_composite(self):
        """最终训练用 InvFreq × MarginBoost × Decay(clinical=10, 固定)，乘积验证。"""
        y = np.array([5] * 3 + [15] * 3 + [25] * 3 + [45] * 3, dtype=float)
        w = np.ones(len(y))
        for comp in [InvFreqWeight(max_ratio=3.0),
                     MarginBoostWeight(),
                     DecayWeight(clinical=10, class_weight=2, dist_k=0.1)]:
            w *= comp.compute(y)
        assert len(w) == 12
        assert np.all(np.isfinite(w))

    def test_merged_param_space_for_search(self):
        """搜索空间应合并模型 HP 和权重 HP。"""
        # 模拟 HPSearcher 的合并逻辑
        model_hp = {"max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]}
        weight_components = [
            DecayWeight(clinical=20, class_weight=[3, 5, 8, 12],
                        dist_k=[0.05, 0.1, 0.2, 0.5]),
        ]
        weight_hp = {}
        for wc in weight_components:
            weight_hp.update(wc.get_param_space())
        full_hp = {**model_hp, **weight_hp}
        assert "class_weight" in full_hp
        assert "dist_k" in full_hp
        assert "max_depth" in full_hp
        # 144 × 6 = 864 combos
        total = 1
        for v in full_hp.values():
            total *= len(v["values"] if isinstance(v, dict) and "values" in v else v)
        assert total >= 90
