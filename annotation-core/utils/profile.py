"""Shared helpers for contour width profiles."""

import numpy as np

_XY_NDIM = 2            # 轮廓必须为 2 维数组
_MIN_COORD_DIM = 2      # 坐标列数下限（≥2 才能取 X/Y）
_MIN_CONTOUR_POINTS = 3  # 至少 3 个点才构成可插值轮廓
_MIN_UNIQUE_SAMPLES = 2  # 去重后至少 2 个采样点才能插值


def _sorted_unique_y_x(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort contour points by Y and remove duplicate Y samples."""
    pts = np.asarray(contour, dtype=np.float64)
    if pts.ndim != _XY_NDIM or pts.shape[1] < _MIN_COORD_DIM or len(pts) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    order = np.argsort(pts[:, 1])
    y_sorted = pts[order, 1]
    x_sorted = pts[order, 0]
    unique_y, unique_idx = np.unique(y_sorted, return_index=True)
    unique_x = x_sorted[unique_idx]
    return unique_y, unique_x


def sample_width_profile(
    left_contour: np.ndarray,
    right_contour: np.ndarray,
    n_bins: int = 150,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate left/right X at shared Y samples and return width profile arrays."""
    if left_contour.size < _MIN_CONTOUR_POINTS or right_contour.size < _MIN_CONTOUR_POINTS:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty

    y_min = max(left_contour[:, 1].min(), right_contour[:, 1].min())
    y_max = min(left_contour[:, 1].max(), right_contour[:, 1].max())
    if y_max <= y_min:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty

    y_centers = np.linspace(y_min, y_max, n_bins)

    left_y, left_x = _sorted_unique_y_x(left_contour)
    right_y, right_x = _sorted_unique_y_x(right_contour)
    if len(left_y) < _MIN_UNIQUE_SAMPLES or len(right_y) < _MIN_UNIQUE_SAMPLES:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty

    left_x_samples = np.interp(y_centers, left_y, left_x)
    right_x_samples = np.interp(y_centers, right_y, right_x)
    widths = right_x_samples - left_x_samples
    return widths, y_centers, left_x_samples, right_x_samples


def build_width_profile_lines(
    left_contour: np.ndarray,
    right_contour: np.ndarray,
    n_bins: int = 150,
) -> np.ndarray:
    """Build histogram-style width profile rows [x_left, x_right, y, width]."""
    widths, y_centers, left_x, right_x = sample_width_profile(
        left_contour, right_contour, n_bins=n_bins
    )
    if len(widths) == 0:
        return np.empty((0, 4), dtype=np.float64)
    return np.column_stack([left_x, right_x, y_centers, widths])
