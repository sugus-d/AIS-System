"""X 方向切片分析 — 切片提取工具（旧接口保留用于过渡）。"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

_MIN_SLICE_VERTICES = 10   # 每 Y 切片最少顶点数（太少跳过该切片）
_BASELINE_FLOOR = 0.01     # 粗糙度基线下限（防止除零）
_ROUGH_RUN_LIMIT = 3       # 连续粗糙点数达到该值判定为边界


def extract_x_profiles(
    vertices: np.ndarray,
    triangles: np.ndarray,
    roughness: np.ndarray,
    y_step: float = 5.0,
) -> list[dict]:
    """在多个 Y 位置提取 X 方向轮廓曲线。"""
    centers = vertices[triangles].mean(axis=1)
    y_min, y_max = float(centers[:, 1].min()), float(centers[:, 1].max())
    profiles = []

    y_pos = y_max - y_step / 2
    while y_pos > y_min:
        mask = np.abs(centers[:, 1] - y_pos) < y_step / 2
        if mask.sum() < _MIN_SLICE_VERTICES:
            y_pos -= y_step
            continue
        pts = np.column_stack([centers[mask, 0], centers[mask, 2], roughness[mask]])
        pts = pts[np.argsort(pts[:, 0])]
        pts[:, 1] = median_filter(pts[:, 1], size=min(5, len(pts)))
        profiles.append({"y": y_pos, "points": pts})
        y_pos -= y_step

    return profiles


def _profile_roughness_baseline(
    points: np.ndarray,
    center_i: int,
) -> float:
    """计算轮廓中心区域（60% 宽度）的粗糙度基线。"""
    n = len(points)
    half_band = max(1, n // 3)
    lo = max(0, center_i - half_band)
    hi = min(n, center_i + half_band)
    return float(np.median(points[lo:hi, 2]))


def detect_clothing_boundary(
    profile: dict,
    roughness_ratio: float = 2.0,
    center_x: float | None = None,
) -> tuple[float | None, float | None]:
    """在单条轮廓上检测左右衣服分界点。

    从脊柱中心向外扫描，粗糙度超过 2× 基线时视为衣服。
    """
    pts = profile["points"]
    if len(pts) < _MIN_SLICE_VERTICES:
        return None, None

    peak_i = int(np.argmax(pts[:, 1]))
    baseline = _profile_roughness_baseline(pts, peak_i)

    if baseline < _BASELINE_FLOOR:
        baseline = 0.02  # 防止除零

    threshold = baseline * roughness_ratio
    half_width = float(pts[-1, 0] - pts[0, 0]) / 2
    if center_x is None:
        center_x = float(pts[peak_i, 0])

    left_x = _scan_roughness_outward(pts, peak_i, -1, threshold)
    right_x = _scan_roughness_outward(pts, peak_i, 1, threshold)

    # 太靠近中心 → 不是衣服
    for bx, side in [(left_x, "left"), (right_x, "right")]:
        if bx is not None and abs(bx - center_x) < half_width * 0.1:
            if side == "left":
                left_x = None
            else:
                right_x = None

    return left_x, right_x


def _scan_roughness_outward(
    points: np.ndarray,
    start_i: int,
    direction: int,
    threshold: float,
) -> float | None:
    """从 start_i 沿 direction 向外扫描粗糙度。"""
    last_smooth_x: float | None = None
    consecutive_rough = 0

    i = start_i
    while 0 <= i < len(points):
        if points[i, 2] <= threshold:
            last_smooth_x = points[i, 0]
            consecutive_rough = 0
        else:
            consecutive_rough += 1
            if consecutive_rough >= _ROUGH_RUN_LIMIT and last_smooth_x is not None:
                return last_smooth_x
        i += direction

    return None
