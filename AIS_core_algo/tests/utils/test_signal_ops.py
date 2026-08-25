"""signal_ops 单元测试 — 确定性合成轮廓，验证各函数行为正确。"""

from __future__ import annotations

import numpy as np

RNG_SEED = 2026


def _straight_contour() -> np.ndarray:
    """直线轮廓：x=[0,100], y=0, 导数恒为 0。"""
    x = np.linspace(0, 100, 200)
    return np.column_stack([x, np.zeros_like(x)])


def _sine_contour() -> np.ndarray:
    """正弦轮廓：已知导数 cos(x)。"""
    x = np.linspace(0, 2 * np.pi, 200)
    return np.column_stack([x, np.sin(x)])


def _noisy_semicircle() -> np.ndarray:
    """半圆 + 固定种子噪声（与 golden 测试的 _contour 一致）。"""
    rng = np.random.default_rng(RNG_SEED)
    theta = np.linspace(0, np.pi, 200)
    return np.column_stack([100 * np.cos(theta), 100 * np.sin(theta)]) + rng.normal(0, 0.1, size=(200, 2))


class TestSmoothContour:
    def test_output_shape_matches_input(self) -> None:
        from landmarks.signal_ops import smooth_contour

        contour = _noisy_semicircle()
        result = smooth_contour(contour)
        assert result.shape == contour.shape

    def test_smooth_reduces_noise(self) -> None:
        """平滑后相邻点差分标准差应小于原始轮廓。"""
        from landmarks.signal_ops import smooth_contour

        contour = _noisy_semicircle()
        raw_diff = np.diff(contour, axis=0)
        smoothed = smooth_contour(contour, sigma=2.0)
        smooth_diff = np.diff(smoothed, axis=0)
        assert np.std(smooth_diff) < np.std(raw_diff)


class TestNormalizeXY:
    def test_sorted_and_unique_x(self) -> None:
        from landmarks.signal_ops import normalize_xy

        contour = _noisy_semicircle()
        x, y = normalize_xy(contour)
        assert len(x) == len(y)
        assert x.shape[0] <= contour.shape[0]  # 可能有重复 X
        assert np.all(np.diff(x) > 0)  # 严格递增

    def test_linear_contour(self) -> None:
        from landmarks.signal_ops import normalize_xy

        contour = _straight_contour()
        x, y = normalize_xy(contour)
        assert len(x) == 200
        assert np.allclose(x, np.linspace(0, 100, 200))
        assert np.all(y == 0.0)


class TestComputeDerivativesFromXY:
    def test_derivative_shape(self) -> None:
        from landmarks.signal_ops import compute_derivatives_from_xy, normalize_xy

        contour = _sine_contour()
        x, y = normalize_xy(contour)
        deriv = compute_derivatives_from_xy(x, y)
        assert deriv.shape == x.shape

    def test_straight_contour_zero_derivative(self) -> None:
        from landmarks.signal_ops import compute_derivatives_from_xy, normalize_xy

        contour = _straight_contour()
        x, y = normalize_xy(contour)
        deriv = compute_derivatives_from_xy(x, y)
        assert np.allclose(deriv, 0, atol=1e-6)

    def test_sine_derivative_approximates_cosine(self) -> None:
        from landmarks.signal_ops import compute_derivatives_from_xy, normalize_xy

        contour = _sine_contour()
        x, y = normalize_xy(contour)
        deriv = compute_derivatives_from_xy(x, y)
        expected = np.cos(x)
        # 边界处 savgol 偏差大，只比较中间 80%
        mid = slice(len(x) // 10, -len(x) // 10)
        assert np.allclose(deriv[mid], expected[mid], atol=0.1)

    def test_too_few_points_returns_empty(self) -> None:
        from landmarks.signal_ops import compute_derivatives_from_xy

        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.0, 0.0, 0.0])
        result = compute_derivatives_from_xy(x, y)
        assert len(result) == 0


class TestFindFlatRegionX:
    def test_left_side_flat(self) -> None:
        """构造左端平坦、右端上升的轮廓，assert 找到平坦区。"""
        from landmarks.signal_ops import compute_derivatives_from_xy, find_flat_region_x, normalize_xy

        x = np.linspace(0, 100, 200)
        # 前 30% 平坦，后 70% 上升
        y = np.where(x < 30, 0.0, x - 30)
        contour = np.column_stack([x, y])
        sorted_pts = contour[np.argsort(contour[:, 0])]
        x_u, y_u = normalize_xy(contour)
        dydx = compute_derivatives_from_xy(x_u, y_u)
        d2ydx2 = compute_derivatives_from_xy(x_u, y_u, derv_order=2)
        # 左端平坦区应找到
        result = find_flat_region_x(sorted_pts, dydx, d2ydx2, side_name="left", window_mm=20.0)
        assert result is not None
        assert result < 30

    def test_right_side_flat(self) -> None:
        """构造左端上升、右端平坦的轮廓，assert 找到平坦区。"""
        from landmarks.signal_ops import compute_derivatives_from_xy, find_flat_region_x, normalize_xy

        x = np.linspace(0, 100, 200)
        y = np.where(x > 70, 0.0, 70 - x)
        contour = np.column_stack([x, y])
        sorted_pts = contour[np.argsort(contour[:, 0])]
        x_u, y_u = normalize_xy(contour)
        dydx = compute_derivatives_from_xy(x_u, y_u)
        d2ydx2 = compute_derivatives_from_xy(x_u, y_u, derv_order=2)
        result = find_flat_region_x(sorted_pts, dydx, d2ydx2, side_name="right", window_mm=20.0)
        assert result is not None

    def test_empty_returns_none(self) -> None:
        from landmarks.signal_ops import find_flat_region_x

        empty = np.empty((0, 2))
        result = find_flat_region_x(empty, np.array([]), np.array([]))
        assert result is None


class TestSelectPointsByDerivative:
    def test_select_greater_than_threshold(self) -> None:
        from landmarks.signal_ops import compute_derivatives_from_xy, normalize_xy, select_points_by_derivative

        contour = _sine_contour()
        x_u, y_u = normalize_xy(contour)
        deriv = compute_derivatives_from_xy(x_u, y_u)
        # 选择导数 > 0.5 的点
        selected = select_points_by_derivative(contour, x_u, deriv, threshold=0.5, keep_greater=True)
        assert len(selected) > 0
        # 验证所有选中点的导数值确实 > 0.5
        sampled = np.interp(selected[:, 0], x_u, deriv)
        assert np.all(sampled > 0.5)

    def test_select_less_than_threshold(self) -> None:
        from landmarks.signal_ops import compute_derivatives_from_xy, normalize_xy, select_points_by_derivative

        contour = _sine_contour()
        x_u, y_u = normalize_xy(contour)
        deriv = compute_derivatives_from_xy(x_u, y_u)
        selected = select_points_by_derivative(contour, x_u, deriv, threshold=0.5, keep_greater=False)
        assert len(selected) > 0
        sampled = np.interp(selected[:, 0], x_u, deriv)
        assert np.all(sampled < 0.5)

    def test_empty_input(self) -> None:
        from landmarks.signal_ops import select_points_by_derivative

        empty = np.empty((0, 2))
        result = select_points_by_derivative(empty, np.array([1.0]), np.array([0.0]), threshold=0.0)
        assert result.shape == (0, 2)
