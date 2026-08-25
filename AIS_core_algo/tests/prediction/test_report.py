"""prediction.report — 自适应色限策略测试（对称/右偏/尖峰/clamp/异常输入）。"""

from __future__ import annotations

import numpy as np
import pytest

from prediction.report import _adaptive_clim


class TestAdaptiveClim:
    def test_all_nan_returns_default(self):
        assert _adaptive_clim(np.array([np.nan, np.nan])) == (0.0, 1.0)

    def test_symmetric_centers_median(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=1000)
        vmin, vmax = _adaptive_clim(values)
        median = float(np.median(values))
        assert vmin < median < vmax
        # 近对称 → 中位数居中（两侧等距）
        assert median - vmin == pytest.approx(vmax - median, rel=0.2)

    def test_right_skew_tukey_with_low_clamp(self):
        rng = np.random.default_rng(2)
        values = np.abs(rng.normal(size=1000))  # 半正态 → 右偏
        vmin, vmax = _adaptive_clim(values, low=0)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        assert vmin >= 0  # clamp 到物理下限
        assert vmax <= q3 + 1.5 * iqr + 1e-9  # Tukey 上界

    def test_spike_keeps_band_narrow(self):
        """尖峰分布：中位数居中，色带不被极少数离群拉开到 100。"""
        values = np.zeros(1000)
        values[:60] = 100.0  # 6% 离群
        vmin, vmax = _adaptive_clim(values)
        assert vmax < 50
        assert vmin == 0.0

    def test_clamp_physical_limits(self):
        values = np.linspace(-10, 200, 100)
        vmin, vmax = _adaptive_clim(values, low=0, high=90)
        assert vmin >= 0
        assert vmax <= 90

    def test_constant_values(self):
        values = np.full(10, 5.0)
        assert _adaptive_clim(values) == (5.0, 5.0)
