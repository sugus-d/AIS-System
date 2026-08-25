"""曲率计算 — 纯算法（渲染已剥离到 visualization/ 编排层）。"""

import numpy as np
import open3d as o3d
import pyvista as pv

from utils.logger import logger


def calculate_curvature(mesh: o3d.geometry.TriangleMesh, curv_type: str = "mean") -> np.ndarray | None:
    """用 PyVista 计算逐顶点曲率。

    Args:
        mesh: 输入网格。
        curv_type: 曲率类型，"mean" 或 "gaussian"。

    Returns:
        逐顶点曲率数组，无三角面时返回 None。
    """
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if len(faces) == 0:
        logger.warning("Mesh has no triangles, cannot compute curvature.")
        return None

    faces_pyvista = np.hstack((np.full((len(faces), 1), 3), faces))
    pv_mesh = pv.PolyData(vertices, faces_pyvista)
    return pv_mesh.curvature(curv_type=curv_type)


def compute_mean_curvature(
    vertices: np.ndarray,
    triangles: np.ndarray,
    clip_range: tuple[float, float] = (-0.03, 0.03),
) -> np.ndarray:
    """Mean curvature with Taubin smoothing, clipped for visualization display."""
    pvm = pv.PolyData(vertices, np.hstack([np.full((len(triangles), 1), 3), triangles]))
    pvm = pvm.smooth_taubin(n_iter=5, pass_band=0.1)
    curv = pvm.curvature(curv_type="mean")
    return np.clip(curv, clip_range[0], clip_range[1])
