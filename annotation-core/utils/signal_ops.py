"""一维/二维轮廓信号处理。

轮廓平滑（smooth_contour）、导数计算（compute_derivatives_from_xy）、
导数阈值筛选（select_points_by_derivative / find_flat_region_x）与
候选点过滤（gradient_filter_candidates）。

landmark 各子域（axilla / neck_root / waist 等）共用，是检测管线的公共路径。
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from utils.logger import logger

_MIN_FIT_POINTS = 5       # savgol 拟合最少样本点
_MIN_GRADIENT_CANDIDATES = 3  # 候选数≤该值时不值得做梯度过滤


def smooth_contour(
    contour: np.ndarray, sigma: float = 1.5, mode: str = "nearest"
) -> np.ndarray:
    """Gaussian smooth X and Y of a contour."""
    return np.column_stack([
        gaussian_filter1d(contour[:, 0].astype(float), sigma=sigma, mode=mode),
        gaussian_filter1d(contour[:, 1].astype(float), sigma=sigma, mode=mode),
    ])


def normalize_xy(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort contour points by X and deduplicate X values."""
    x = contour[:, 0].astype(np.float64)
    y = contour[:, 1].astype(np.float64)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    x_unique, unique_idx = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_idx]
    return x_unique, y_unique


def compute_derivatives_from_xy(
    x_unique: np.ndarray, y_unique: np.ndarray, derv_order: int = 1
) -> np.ndarray:
    """Compute Savitzky-Golay derivatives from sorted unique X/Y samples."""
    if len(x_unique) < _MIN_FIT_POINTS:
        return np.array([])

    window = min(11, len(x_unique) if len(x_unique) % 2 == 1 else len(x_unique) - 1)
    if window < _MIN_FIT_POINTS:
        return np.array([])

    polyorder = 3 if window >= _MIN_FIT_POINTS else 2
    delta = float(np.median(np.diff(x_unique)))
    if not np.isfinite(delta) or delta <= 0:
        delta = 1.0
    return savgol_filter(
        y_unique,
        window_length=window,
        polyorder=polyorder,
        deriv=derv_order,
        delta=delta,
        mode="interp",
    )


def select_points_by_derivative(
    points: np.ndarray,
    x_unique: np.ndarray,
    deriv: np.ndarray,
    threshold: float,
    keep_greater: bool = True,
) -> np.ndarray:
    """Select points by interpolating derivative values at each X coordinate."""
    if len(points) == 0 or len(x_unique) == 0 or len(deriv) == 0:
        return np.empty((0, 2), dtype=float)
    sampled = np.interp(points[:, 0], x_unique, deriv, left=np.nan, right=np.nan)
    if keep_greater:
        return points[sampled > threshold]
    return points[sampled < threshold]


def find_flat_region_x(
    sorted_pts: np.ndarray,
    dydx: np.ndarray,
    d2ydx2: np.ndarray,
    side_name: str = "left",
    window_mm: float = 20.0,
    slope_thresh: float = 0.3,
    curv_thresh: float = 0.01,
) -> float | None:
    """Find the first X where a continuous window satisfies flatness thresholds."""
    if len(sorted_pts) == 0:
        return None

    if side_name == "left":
        dydx = dydx[::-1]
        d2ydx2 = d2ydx2[::-1]
        xs = sorted_pts[::-1, 0]
    else:
        xs = sorted_pts[:, 0]

    n = len(xs)
    for i in range(n):
        if side_name == "left":
            j = i
            while j < n and xs[i] - xs[j] < window_mm:
                j += 1
        else:
            j = i
            while j < n and xs[j] - xs[i] < window_mm:
                j += 1

        if j >= n:
            break

        if np.all(np.abs(dydx[i:j]) < slope_thresh) and np.all(
            np.abs(d2ydx2[i:j]) < curv_thresh
        ):
            return float(xs[i])

    return None


def gradient_filter_candidates(
    candidate_indices: list[int], ws_smooth: np.ndarray, ys: np.ndarray
) -> list[int]:
    """Filter candidate row indices by gradient percentile."""
    if len(candidate_indices) <= _MIN_GRADIENT_CANDIDATES:
        return candidate_indices

    ws_grad = np.gradient(ws_smooth, ys)
    cand_grads = np.array([ws_grad[ci] for ci in candidate_indices])
    threshold = float(np.percentile(cand_grads, 10))
    kept = [ci for ci, g in zip(candidate_indices, cand_grads, strict=False) if g >= threshold]
    if kept and len(kept) < len(candidate_indices):
        logger.info(
            f"[GRAD] removed {len(candidate_indices) - len(kept)}/{len(candidate_indices)} "
            f"high-gradient candidates (threshold={threshold:.3f})"
        )
    return kept if kept else candidate_indices
