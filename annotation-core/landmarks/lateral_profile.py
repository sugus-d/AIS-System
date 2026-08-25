"""提取网格侧向轮廓并分割为左右两侧。

提供：
- ``extract_split_contours``：从网格顶点提取实际身体边界轮廓并分割为左右两侧。
- ``compute_width_profile``：从左右轮廓计算宽度剖面。
"""

import contextlib
import warnings
from typing import Any

import alphashape
import numpy as np
import open3d as o3d

from utils.logger import logger
from utils.profile import sample_width_profile

_XY_NDIM = 2                # 点阵必须为 2 维数组
_MIN_COORD_DIM = 2          # 坐标列数下限
_MIN_CONTOUR_POINTS = 3     # 轮廓至少 3 个点才做 PCA 切分
_MIN_DOWNSAMPLE_POINTS = 2  # max_points ≥ 2 才做均匀下采样
_MIN_TRANSITIONS = 2        # 至少 2 个左右过渡点才做段分析


def extract_split_contours(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从网格顶点提取实际边界轮廓并分割为左右两侧。

    Args:
        vertices: 网格顶点数组 (N, 3)。

    Returns:
        tuple: (left_contour, right_contour)，均为 (M, 2) 数组，保持原始 CW 顺序。
    """
    contour: np.ndarray = _extract_body_contour(vertices)
    left_contour, right_contour = _split_contours(contour)

    return left_contour, right_contour


def compute_width_profile(
    left_contour: np.ndarray,
    right_contour: np.ndarray,
    n_bins: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    """从左右轮廓插值计算宽度剖面。

    替代原 ``build_lateral_profiles`` 的 widths / y_centers 输出，
    改用实际边界轮廓而非每层极值 X。

    Args:
        left_contour: 左轮廓 (M, 2)，任意排列顺序。
        right_contour: 右轮廓 (M, 2)，任意排列顺序。
        n_bins: 垂直采样数，默认 150。

    Returns:
        tuple: (widths, y_centers)，均为 (n_bins,) 数组。
    """
    widths: np.ndarray
    y_centers: np.ndarray
    widths, y_centers, _, _ = sample_width_profile(left_contour, right_contour, n_bins=n_bins)
    return widths, y_centers


def _extract_body_contour(vertices: np.ndarray, max_points: int = 300) -> np.ndarray:
    """基于 α-shape 提取网格外轮廓。

    先用体素下采样去除内部点，再用 alphashape 提取外轮廓。
    对返回轮廓等间隔下采样到最多 max_points 个点。
    CW 排列且从最高点开始。

    Args:
        vertices: 网格顶点 (N, 3)。
        max_points: 轮廓最大采样点数；≥2 时做下采样。
    """
    if vertices.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    # 体素下采样：减少内部点数量，提高 alphashape 效率
    bbox_xy: np.ndarray = np.ptp(vertices[:, :2], axis=0)
    diag: float = float(np.linalg.norm(bbox_xy))
    voxel_size: float = max(diag / 300.0, 0.5)

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd_ds = pcd.voxel_down_sample(voxel_size)
        sampled: np.ndarray = np.asarray(pcd_ds.points, dtype=np.float64)
    except (ValueError, RuntimeError):
        sampled = vertices

    try:
        if sampled.ndim != _XY_NDIM or sampled.shape[1] < _MIN_COORD_DIM or sampled.shape[0] < _MIN_CONTOUR_POINTS:
            logger.warning("Not enough 2D points for alphashape, falling back to raw vertices")
            raise ValueError("not enough 2D points for alphashape")

        # alphashape 可能抛出第三方警告，统一屏蔽
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shape: Any = alphashape.alphashape(sampled[:, :2], 0.3)

        if shape is None or getattr(shape, "is_empty", False):
            logger.warning("Alphashape returned empty geometry, falling back to raw vertices")
            raise ValueError("alphashape returned empty geometry")

        # GeometryCollection 时选取面积最大子几何体
        geoms = getattr(shape, "geoms", None)
        if geoms:
            # GeometryCollection 退化时取面积最大者；失败则保留原几何
            with contextlib.suppress(ValueError, RuntimeError):
                shape = max(list(geoms), key=lambda g: getattr(g, "area", 0))

        exterior = getattr(shape, "exterior", None)
        coords: np.ndarray | None = None
        if exterior is not None and hasattr(exterior, "coords"):
            coords = np.asarray(list(exterior.coords), dtype=np.float64)

        if coords is None or coords.ndim != _XY_NDIM or coords.shape[1] < _MIN_COORD_DIM:
            logger.warning("Invalid exterior coords from alphashape, falling back to raw vertices")
            raise ValueError("invalid exterior coords from alphashape")

        contour: np.ndarray = coords[:-1, :2]

        contour_size: int = contour.shape[0]
        if contour_size == 0:
            return np.empty((0, 2), dtype=np.float64)
        elif contour_size > max_points and max_points >= _MIN_DOWNSAMPLE_POINTS:
            # 保留首点（top），其余均匀采样
            rest: int = max_points - 1
            idxs: np.ndarray = np.round(np.linspace(1, contour_size - 1, rest)).astype(int)
            selected: np.ndarray = np.concatenate(([0], idxs))
            contour = contour[selected]

    except (ValueError, RuntimeError):
        # 回退到原始顶点的 XY 投影，保证函数鲁棒
        contour = np.asarray(vertices[:, :2], dtype=np.float64)

    # 面积符号判断 CW/CCW，确保返回 CW 方向
    x: np.ndarray = contour[:, 0]
    y: np.ndarray = contour[:, 1]
    area: float = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if area > 0:
        contour = contour[::-1]

    # 滚动使最高点（最大 Y）在 index 0
    top_idx: int = int(np.argmax(contour[:, 1]))
    if top_idx != 0:
        contour = np.roll(contour, -top_idx, axis=0)

    return contour


def _split_contours(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从完整轮廓中分割为左右两侧轮廓线。

    WHY: 先 PCA 对齐主轴再切分，适应歪斜 body 姿态，
    然后用连通段连续性分析代替简单的中位数 X 切分，防止错切。

    Args:
        contour: 完整轮廓 (N, 2)，CW 排列且从最高点开始。

    Returns:
        tuple: (left_contour, right_contour)，均为 (M, 2) 数组。
    """
    if contour is None or contour.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    pts: np.ndarray = np.asarray(contour, dtype=np.float64)
    n: int = pts.shape[0]
    if n < _MIN_CONTOUR_POINTS:
        split_x: float = float(np.median(pts[:, 0]))
        mask: np.ndarray = pts[:, 0] < split_x
        return pts[mask], pts[~mask]

    # PCA 对齐主轴后按旋转坐标系的 X 中位数切分
    centered: np.ndarray = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    principal: np.ndarray = vt[0]
    if principal[1] < 0 or (principal[1] == 0 and principal[0] < 0):
        principal = -principal
    phi: float = float(np.arctan2(principal[1], principal[0]))
    rot: float = np.pi / 2.0 - phi
    c, s = np.cos(rot), np.sin(rot)
    R: np.ndarray = np.array([[c, -s], [s, c]], dtype=np.float64)
    rotated: np.ndarray = centered.dot(R.T)

    split_x = float(np.median(rotated[:, 0]))
    is_left: np.ndarray = rotated[:, 0] < split_x

    # 找左右侧分界点（transitions），按连续段长度决定左右归属
    trans: np.ndarray = np.where(is_left[1:] != is_left[:-1])[0] + 1
    if trans.size < _MIN_TRANSITIONS:
        fx: float = float(np.median(pts[:, 0]))
        mask = pts[:, 0] < fx
        return pts[mask], pts[~mask]

    runs: list[tuple[int, int, bool]] = []
    prev: int = 0
    for t in trans:
        runs.append((prev, t - prev, bool(is_left[prev])))
        prev = t
    runs.append((prev, n - prev, bool(is_left[prev])))

    left_run: tuple[int, int, bool] | None = max((r for r in runs if r[2]), key=lambda r: r[1], default=None)
    right_run: tuple[int, int, bool] | None = max((r for r in runs if not r[2]), key=lambda r: r[1], default=None)

    if left_run is None or right_run is None:
        fx = float(np.median(pts[:, 0]))
        mask = pts[:, 0] < fx
        return pts[mask], pts[~mask]

    # 滚动使左侧段在 index 0，在右侧段起点处切开
    n_pts: int = pts.shape[0]
    pts = np.roll(pts, -left_run[0], axis=0)
    cut: int = int((right_run[0] - left_run[0]) % n_pts)
    return pts[:cut], pts[cut:]
