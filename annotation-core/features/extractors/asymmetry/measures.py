"""8 种逐顶点测量指标计算。

每个函数接收 open3d TriangleMesh 并返回 (N,) 或 (N, 3) ndarray。
纯函数，无 I/O 无副作用。
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from mesh.curvature import calculate_curvature
from mesh.roi.bfs import compute_mesh_roughness as _compute_roughness_face


def compute_mean_curvature(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """逐顶点平均曲率。

    底层调用 Open3D 曲率计算，经 1%-99% 百分位截断去除离群值。

    Args:
        mesh: 三角网格。

    Returns:
        (N,) float64 — 平均曲率，异常时返回全零数组。
    """
    km = calculate_curvature(mesh, "mean")
    if km is None:
        km = np.zeros(len(mesh.vertices))
    km = np.asarray(km, dtype=np.float64)
    _clip_percentile(km)
    return km


def compute_gauss_curvature(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """逐顶点高斯曲率。

    同 mean_curvature，经百分位截断。

    Args:
        mesh: 三角网格。

    Returns:
        (N,) float64 — 高斯曲率，异常时返回全零数组。
    """
    kg = calculate_curvature(mesh, "gaussian")
    if kg is None:
        kg = np.zeros(len(mesh.vertices))
    kg = np.asarray(kg, dtype=np.float64)
    _clip_percentile(kg)
    return kg


def compute_roughness(mesh: o3d.geometry.TriangleMesh, faces: np.ndarray) -> np.ndarray:
    """逐顶点粗糙度（面粗糙度 → 散射累加到顶点）。

    先计算每个三角面的粗糙度，再按面索引散射累加到顶点并归一化。

    Args:
        mesh: 三角网格。
        faces: (M, 3) 三角面顶点索引，与 mesh.triangles 相同。

    Returns:
        (N,) float64 — 逐顶点粗糙度。
    """
    rough_f = _compute_roughness_face(mesh)
    rv = np.zeros(len(mesh.vertices), dtype=np.float64)
    np.add.at(rv, faces.ravel(), np.repeat(rough_f, 3))
    fc = np.zeros(len(mesh.vertices), dtype=np.float64)
    np.add.at(fc, faces.ravel(), 1)
    rv /= np.maximum(fc, 1)
    return rv


def compute_height(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """逐顶点曲面高度（Z 坐标）。

    Args:
        mesh: 三角网格。

    Returns:
        (N,) float64 — Z 坐标值。
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return vertices[:, 2].copy()


def compute_normal_angle(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """逐顶点法向量与垂直方向的夹角（度）。

    计算 ``abs(n_y)`` 的反余弦，有效范围 [0, 90]。

    Args:
        mesh: 三角网格（``compute_vertex_normals`` 会被调用）。

    Returns:
        (N,) float64 — 法向量角度（度）。
    """
    mesh.compute_vertex_normals()
    vn = np.asarray(mesh.vertex_normals)
    na = np.degrees(np.arccos(np.clip(np.abs(vn[:, 1]), 0.0, 1.0)))
    return np.clip(na, 0, 90)


def compute_normal_vectors(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """逐顶点归一化法向量。

    Args:
        mesh: 三角网格（``compute_vertex_normals`` 会被调用）。

    Returns:
        (N, 3) float64 — 归一化法向量。
    """
    mesh.compute_vertex_normals()
    vn = np.asarray(mesh.vertex_normals, dtype=np.float64)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.maximum(norms, 1e-12)


def compute_normal_cos(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """左右法向量余弦相似度（用于 DM 模式）。

    注：DM 模式下左右分别平均法向量后点积得到余弦角，
    此函数仅返回归一化法向量，实际余弦在 ``differences.py`` 中计算。

    Args:
        mesh: 三角网格。

    Returns:
        (N, 3) float64 — 归一化法向量（与 ``compute_normal_vectors`` 相同）。
    """
    return compute_normal_vectors(mesh)


def compute_normal_sin(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """左右法向量正弦值（用于 DM 模式）。

    注：同 cos，实际正弦在 ``differences.py`` 中基于左右平均法向量计算。
    此函数返回归一化法向量。

    Args:
        mesh: 三角网格。

    Returns:
        (N, 3) float64 — 归一化法向量。
    """
    return compute_normal_vectors(mesh)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _clip_percentile(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> None:
    """对数组进行百分位截断（就地操作）。"""
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return
    lo, hi = np.percentile(finite, [low, high])
    np.clip(arr, lo, hi, out=arr)
