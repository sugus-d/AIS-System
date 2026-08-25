"""2D 几何计算。

圆段相交（segment_circle_intersection）、轮廓方向判断（is_contour_ccw）、
维度归一（to_2d）。与 utils/mesh.py（open3d 网格）职责分离，本模块不做网格 I/O。
"""

import numpy as np
from shapely.geometry import LinearRing

from utils.logger import logger

_XY_NDIM = 2              # 数组必须为 2 维
_XY_COORD_DIM = 2         # 必须恰好为 XY 两列
_DOT_EPSILON = 1e-12      # 方向向量点积小于该值视为零（退化为平行）
_MIN_CONTOUR_POINTS = 3   # 至少 3 个点才能判断绕行方向


def segment_circle_intersection(A: np.ndarray, B: np.ndarray, C: np.ndarray, r: float) -> float | None:
    """求解线段 AB 与以 C 为中心、半径 r 的圆的交点参数 t（0<=t<=1）。

    返回第一个满足 0<=t<=1 的 t 值（float），找不到则返回 None。

    Args:
        A, B, C: 长度为 2 的数组或可转换为 ndarray（XY 平面）。
        r: 圆半径（float）。
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)

    D = B - A
    OC = A - C
    a = float(np.dot(D, D))
    if a < _DOT_EPSILON:
        return None
    b = 2.0 * float(np.dot(D, OC))
    c = float(np.dot(OC, OC)) - float(r) * float(r)
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    sqrt_disc = float(np.sqrt(disc))
    t1 = (-b + sqrt_disc) / (2.0 * a)
    t2 = (-b - sqrt_disc) / (2.0 * a)
    for t in (t1, t2):
        if 0.0 <= t <= 1.0:
            return float(t)
    return None


def is_contour_ccw(contour_xy: np.ndarray) -> bool:
    """判断轮廓点列是否为逆时针（CCW）。

    闭合轮廓优先使用 Shapely 的 LinearRing.is_ccw；开口轮廓保留鞋带公式的
    最小代价实现，用于近似判断整体方向。
    """
    xy = np.asarray(contour_xy, dtype=np.float64)[:, :2]
    if len(xy) < _MIN_CONTOUR_POINTS:
        return False

    if np.allclose(xy[0], xy[-1]):
        try:
            return bool(LinearRing(xy).is_ccw)
        except Exception:
            pass

    area = float(np.dot(xy[:-1, 0], xy[1:, 1]) - np.dot(xy[1:, 0], xy[:-1, 1]))
    return area > 0.0


def to_2d(arr: np.ndarray) -> np.ndarray:
    """将 (N,2) 或 (N,3) 数组转为 (N,2) float64。"""
    a = np.asarray(arr)
    if a.ndim != _XY_NDIM:
        logger.error("Input must be 2D array")
        raise ValueError("Input must be 2D array")
    if a.shape[1] == _XY_COORD_DIM:
        return a.astype(np.float64)
    return a[:, :2].astype(np.float64)
