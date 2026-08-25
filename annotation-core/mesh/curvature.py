"""Curvature computation and visualization for back surface meshes."""

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import open3d as o3d
import pyvista as pv

from utils.io import save_img
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


def visualize_curvature(
    mesh: o3d.geometry.TriangleMesh,
    curvatures: np.ndarray,
    clip_range: tuple[float, float] | None,
    output_path: str | None = None,
) -> np.ndarray:
    """创建并保存顶点曲率的 2D tripcolor 图。

    Returns:
        原始曲率数组（受 clip_range 限制前的拷贝）。
    """
    faces = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if len(faces) == 0:
        logger.warning("Mesh has no triangles.")
        return np.zeros(len(vertices))

    vertex_curvatures = curvatures.copy()
    original_range = (np.min(vertex_curvatures), np.max(vertex_curvatures))
    logger.info(f"Original vertex curvature range: {original_range}")

    if clip_range is not None:
        vertex_curvatures = np.clip(vertex_curvatures, clip_range[0], clip_range[1])
        logger.info(f"Clipping range: {clip_range} for color mapping.")

    try:
        fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
        x, y = vertices[:, 0], vertices[:, 1]
        triang = mtri.Triangulation(x, y, faces)
        vmin = np.min(vertex_curvatures) if clip_range is None else clip_range[0]
        vmax = np.max(vertex_curvatures) if clip_range is None else clip_range[1]
        ax.tripcolor(
            triang,
            vertex_curvatures,
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
            shading="gouraud",
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("black")
        if output_path is not None:
            save_img(fig, output_path)
    except Exception as e:
        logger.error(f"Failed to create or save the plot: {e}")


def get_curvature_img(
    mesh: o3d.geometry.TriangleMesh,
    output_path: str | None = None,
    clip_range: tuple[float, float] | None = None,
    curv_type: str = "mean",
    smooth_iterations: int = 50,
    outlier_percentile: float = 99.5,
) -> None:
    """Generate and save a curvature image with smoothing + outlier removal.

    Pipeline: smooth (Taubin) → curvature → percentile clip → save.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.triangles, dtype=np.int64)
    pv_mesh = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]))

    if smooth_iterations > 0:
        pv_mesh = pv_mesh.smooth(
            n_iter=smooth_iterations,
            relaxation_factor=0.3,
            boundary_smoothing=False,
            feature_smoothing=False,
        )

    curvatures = pv_mesh.curvature(curv_type=curv_type)
    finite = np.isfinite(curvatures)
    if finite.sum() > 0:
        lo = np.percentile(curvatures[finite], 100 - outlier_percentile)
        hi = np.percentile(curvatures[finite], outlier_percentile)
        curvatures = np.clip(curvatures, lo, hi)

    import open3d as o3d

    viz_mesh = o3d.geometry.TriangleMesh()
    viz_mesh.vertices = o3d.utility.Vector3dVector(pv_mesh.points)
    viz_mesh.triangles = o3d.utility.Vector3iVector(pv_mesh.faces.reshape(-1, 4)[:, 1:])
    visualize_curvature(viz_mesh, curvatures, clip_range=clip_range, output_path=output_path)


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
