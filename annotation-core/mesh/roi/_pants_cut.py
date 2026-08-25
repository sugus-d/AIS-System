"""裤子切割 — 粗糙度扫描 + Y 分位数备用。

先用粗糙度扫描找裤腰边界。切太少则补充 Y 分位数切割。
"""

from __future__ import annotations

import numpy as np

from utils.logger import logger

from ._bfs_impl import compute_mesh_roughness, largest_component
from ._mesh_cut import filter_by_polygon

_MIN_SLICE_VERTICES = 10    # 每 X 切片最少顶点数
_ROUGH_RUN_LIMIT = 3        # 连续粗糙点数达到该值判定为边界
_MIN_BOUNDARY_POINTS = 5    # 粗糙度边界点数下限（太少视为未检测到）
_MIN_X_SLICE_VERTICES = 5   # Y 分位数扫描每切片最少顶点数
_MIN_FALLBACK_POINTS = 3    # 分位边界点数下限（太少回退到两端点）
_MIN_REMOVED_FRACTION = 0.08  # 切除三角面比例下限（低于则触发备用切割）


def _scan_by_roughness(
    vertices: np.ndarray,
    triangles: np.ndarray,
    roughness: np.ndarray,
    *,
    roughness_ratio: float = 1.8,
) -> np.ndarray | None:
    """逐 X 切片扫描粗糙度跃迁，返回边界曲线 (X, Y)。"""
    centers = vertices[triangles].mean(axis=1)
    y_min, y_max = float(centers[:, 1].min()), float(centers[:, 1].max())
    y_mid = y_min + (y_max - y_min) * 0.4

    x_step = max(5.0, float(centers[:, 0].ptp()) * 0.02)
    x_positions = np.arange(centers[:, 0].min() + x_step,
                            centers[:, 0].max() - x_step, x_step)

    boundary: list[tuple[float, float]] = []
    for x in x_positions:
        mask = (np.abs(centers[:, 0] - x) < x_step / 2) & (centers[:, 1] < y_mid)
        if mask.sum() < _MIN_SLICE_VERTICES:
            continue

        pts = np.column_stack([centers[mask, 0], centers[mask, 1], roughness[mask]])
        pts = pts[np.argsort(pts[:, 1])]

        baseline = float(np.median(pts[:max(3, len(pts) // 4), 2]))
        threshold = max(baseline * roughness_ratio, 0.03)

        last_smooth_y: float | None = None
        rough_count = 0
        for i in range(len(pts)):
            if pts[i, 2] <= threshold:
                last_smooth_y = pts[i, 1]
                rough_count = 0
            else:
                rough_count += 1
                if rough_count >= _ROUGH_RUN_LIMIT and last_smooth_y is not None:
                    boundary.append((x, last_smooth_y))
                    break

    return np.array(boundary) if len(boundary) >= _MIN_BOUNDARY_POINTS else None


def _yfraction_curve(
    vertices: np.ndarray,
    triangles: np.ndarray,
    fraction: float = 0.23,
) -> np.ndarray:
    """在固定 Y 分位数处构造曲线边界点。"""
    centers = vertices[triangles].mean(axis=1)
    y_min, y_max = float(centers[:, 1].min()), float(centers[:, 1].max())
    cut_y = y_min + (y_max - y_min) * fraction

    x_step = max(5.0, float(centers[:, 0].ptp()) * 0.03)
    x_lo, x_hi = float(centers[:, 0].min()), float(centers[:, 0].max())
    x_positions = np.arange(x_lo + x_step, x_hi - x_step, x_step)

    bp: list[tuple[float, float]] = []
    for x in x_positions:
        mask = np.abs(centers[:, 0] - x) < x_step / 2
        if mask.sum() >= _MIN_X_SLICE_VERTICES:
            bp.append((x, float(np.percentile(centers[mask, 1], fraction * 100))))
        else:
            bp.append((x, cut_y))

    if len(bp) < _MIN_FALLBACK_POINTS:
        bp = [(x_lo + 1, cut_y), (x_hi - 1, cut_y)]

    return np.array(bp)


def _apply_cut(
    vertices: np.ndarray,
    triangles: np.ndarray,
    boundary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """用边界曲线切割网格，返回最大连通分量。"""
    import open3d as o3d

    bps = boundary[np.argsort(boundary[:, 0])]
    x_min_all, x_max_all = float(vertices[:, 0].min()), float(vertices[:, 0].max())
    y_max = float(vertices[triangles].mean(axis=1)[:, 1].max())

    poly = np.vstack([
        np.array([[x_min_all, y_max]]),
        np.array([[x_max_all, y_max]]),
        bps[::-1],
        np.array([[x_min_all, float(bps[:, 1].min())]]),
    ])

    out_v, out_t = filter_by_polygon(vertices, triangles, poly)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(out_v)
    mesh.triangles = o3d.utility.Vector3iVector(out_t)
    mesh = largest_component(mesh)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.triangles)


def remove_pants(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    roughness_ratio: float = 1.8,
    y_fraction: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """切除裤子区域。

    先用粗糙度扫描找边界。
    若切太少（< 8% 三角面），追加 Y 分位数切割。

    Parameters
    ----------
    vertices : (N, 3)
    triangles : (M, 3)
    roughness_ratio : 粗糙度跃迁阈值，默认 1.8。
    y_fraction : 备用 Y 分位数，默认 0.23。

    Returns
    -------
    (out_v, out_t)
    """
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    roughness = compute_mesh_roughness(mesh)

    boundary = _scan_by_roughness(vertices, triangles, roughness,
                                  roughness_ratio=roughness_ratio)

    if boundary is not None:
        # 先用粗糙度边界切
        out_v, out_t = _apply_cut(vertices, triangles, boundary)
        removed_pct = (len(triangles) - len(out_t)) / max(len(triangles), 1)

        if removed_pct >= _MIN_REMOVED_FRACTION:
            logger.info("Pants cut (roughness): %d removed (%.1f%%)",
                        len(triangles) - len(out_t), removed_pct * 100)
            return out_v, out_t

        # 切太少 → 追加 Y 分位数切割
        logger.info("Roughness cut only %.1f%%, applying Y-fraction fallback", removed_pct * 100)

    # 粗糙度失败或切太少 → Y 分位数切割
    y_boundary = _yfraction_curve(vertices, triangles, y_fraction)
    out_v, out_t = _apply_cut(vertices, triangles, y_boundary)

    removed = len(triangles) - len(out_t)
    logger.info("Pants cut (Y=%.0f%%): %d removed (%.1f%%)",
                y_fraction * 100, removed, removed / max(len(triangles), 1) * 100)
    return out_v, out_t
