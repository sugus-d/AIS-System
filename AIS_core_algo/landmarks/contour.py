"""轮廓工具 — 重采样 / 下边界提取 / 段搜索。

弧长均匀重采样（resample_polyline_uniform）、X 分桶下边界
（extract_lower_boundary_per_integer_x）、轮廓段搜索
（search_segment_indices / extract_longest_contiguous_segment_in_box）。

landmark 检测的轮廓预处理公共路径。
"""

import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import box as shapely_box
from shapely.geometry import LineString

# ── 输入数组形状校验阈值 ──
_XY_NDIM = 2              # 点阵必须为 2 维数组
_MIN_COORD_DIM = 2        # 坐标列数下限（≥2 才能取前两列）
_XY_COORD_DIM = 2         # 必须恰好为 XY 两列
_MIN_SEGMENT_POINTS = 2   # 至少 2 个点才构成线段
_SEGMENT_LENGTH_EPSILON = 1e-12  # 线段长度小于该值视为零（去重零长段）
_TOTAL_LENGTH_EPSILON = 1e-8     # 总弧长小于该值视为零（退化为单点）


def resample_polyline_uniform(polyline: np.ndarray, step: float = 0.5) -> np.ndarray:
    """按弧长均匀重采样多段线，返回 (M,2) 点阵。

    Args:
        polyline: 输入多段线，形状 (N,2)。
        step: 目标采样间距（坐标单位），默认为 0.5。

    Returns:
        重采样后的 (M,2) 数组；输入无效时返回空数组。
    """
    if polyline is None:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != _XY_NDIM or pts.shape[1] < _MIN_COORD_DIM or len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(pts) == 1:
        return pts.copy()

    # 用累计弧长 + 线性插值实现均匀重采样，避免逐点调用几何对象接口。
    xy = pts[:, :2]
    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    valid_mask = np.concatenate(([True], segment_lengths > _SEGMENT_LENGTH_EPSILON))
    xy = xy[valid_mask]
    if len(xy) == 1:
        return xy.copy()

    segment_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= _TOTAL_LENGTH_EPSILON:
        return xy[:1].copy()

    sample_count = max(int(np.ceil(total_length / float(step))), 1)
    distances = np.linspace(0.0, total_length, sample_count + 1)
    interpolator = interp1d(cumulative, xy, axis=0)
    return np.asarray(interpolator(distances), dtype=np.float64)


def lower_boundary_per_integer_x(resampled: np.ndarray) -> np.ndarray:
    """按整数 X 分桶，保留每桶中最小的 Y 并按 X 升序返回（向量化实现）。"""
    if resampled is None:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.asarray(resampled, dtype=np.float64)
    if pts.ndim != _XY_NDIM or pts.shape[1] < _MIN_COORD_DIM or len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)

    x_bins = np.rint(pts[:, 0]).astype(int)
    order = np.argsort(x_bins)
    xb = x_bins[order]
    ys = pts[:, 1][order]
    unique_bins, start_idx = np.unique(xb, return_index=True)
    # 使用 ufunc.reduceat 计算每个分组的最小 Y（比 Python 循环快）
    min_y = np.minimum.reduceat(ys, start_idx)
    return np.column_stack([unique_bins.astype(np.float64), min_y.astype(np.float64)])


def extract_lower_boundary_per_integer_x(
    points: np.ndarray, step: float = 0.5
) -> np.ndarray:
    """公共接口：对轮廓按弧长重采样并提取按整数 X 的下边界点。

    参数和返回值与原 `landmarks.axilla` 中的实现兼容。
    """
    if points is None:
        return np.empty((0, 2), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != _XY_NDIM or pts.shape[1] != _XY_COORD_DIM or len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if len(pts) == 1:
        return np.array(
            [[float(np.rint(pts[0, 0])), float(pts[0, 1])]],
            dtype=np.float64,
        )

    resampled = resample_polyline_uniform(pts, step=step)
    return lower_boundary_per_integer_x(resampled)


def search_segment_indices(
    contour: np.ndarray, before_pt: np.ndarray, after_pt: np.ndarray
) -> np.ndarray:
    """在 contour 上找到最靠近 before_pt 与 after_pt 的索引，并返回闭区间索引数组。

    处理 contour 的 wrap-around 情况，返回一个整数索引的 numpy.ndarray（可为空）。
    该函数从业务代码中抽取出来，以供多个模块复用。
    """
    n = len(contour)
    if n == 0:
        return np.array([], dtype=int)

    before_dists = np.linalg.norm(contour[:, :2] - np.asarray(before_pt)[:2], axis=1)
    before_idx = int(np.argmin(before_dists))

    after_dists = np.linalg.norm(contour[:, :2] - np.asarray(after_pt)[:2], axis=1)
    after_idx = int(np.argmin(after_dists))

    if before_idx <= after_idx:
        return np.arange(before_idx, after_idx + 1, dtype=int)
    return np.concatenate([
        np.arange(before_idx, n, dtype=int),
        np.arange(0, after_idx + 1, dtype=int),
    ])


def extract_longest_contiguous_segment_in_box(
    contour: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    closed: bool = True,
) -> np.ndarray:
    """返回轮廓与矩形盒子相交后，最长连续边缘段对应的原始索引。

    Args:
        contour: 轮廓点数组，形状为 (N, >=2)。
        x_min, x_max, y_min, y_max: 轴对齐矩形边界，包含边界点。
        closed: 保留的兼容参数；当轮廓是闭合曲线时，允许结果跨越首尾。

    Returns:
        长度为 M 的整数索引数组，表示输入轮廓中的连续点段；若无相交段，
        返回空数组。
    """
    if contour is None:
        return np.array([], dtype=int)
    pts = np.asarray(contour, dtype=float)
    if pts.ndim != _XY_NDIM or pts.shape[0] == 0 or pts.shape[1] < _MIN_COORD_DIM:
        return np.array([], dtype=int)

    try:
        line = LineString(pts[:, :2].tolist())
        clip_area = shapely_box(float(x_min), float(y_min), float(x_max), float(y_max))
        clipped = line.intersection(clip_area)
    except Exception:
        return np.array([], dtype=int)

    if clipped.is_empty:
        return np.array([], dtype=int)

    segs = (
        [clipped]
        if clipped.geom_type == "LineString"
        else list(getattr(clipped, "geoms", []))
    )
    segs = [
        seg
        for seg in segs
        if getattr(seg, "geom_type", "") == "LineString" and float(seg.length) > 0.0
    ]
    if len(segs) == 0:
        return np.array([], dtype=int)

    best_seg = max(segs, key=lambda seg: float(seg.length))
    coords = np.asarray(best_seg.coords, dtype=float)
    if coords.shape[0] < _MIN_SEGMENT_POINTS:
        return np.array([], dtype=int)

    start_idx = int(np.argmin(np.linalg.norm(pts[:, :2] - coords[0], axis=1)))
    end_idx = int(np.argmin(np.linalg.norm(pts[:, :2] - coords[-1], axis=1)))

    if closed:
        return search_segment_indices(pts, pts[start_idx], pts[end_idx])

    if start_idx <= end_idx:
        return np.arange(start_idx, end_idx + 1, dtype=int)
    return np.arange(end_idx, start_idx + 1, dtype=int)
