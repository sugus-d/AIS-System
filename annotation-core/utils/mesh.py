"""open3d 网格工具 — 克隆 / 重建 / I/O / 2D→3D 提升。

提供 mesh 级操作：clone_mesh、build_mesh、load_mesh（按文件/受检者）、
preprocess_to_vertices、lift_2d_to_vertex（轮廓点映射回网格顶点）。
与 utils/geometry.py（2D 几何）互补，本模块只处理网格数据。
"""

import copy
import os

import numpy as np
import open3d as o3d

from utils.logger import logger

_XY_NDIM = 2  # 二维点阵
_XYZ_DIM = 3  # 已含 Z 的三维点阵
_MIN_SAMPLE_VERTICES = 10  # 下半身采样顶点数下限（太少则退化为全量估计）


def clone_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """克隆 TriangleMesh，兼容 CPU 与 CUDA-backed 实例。

    CUDA 实例（Open3D Tensor 网格）必须用其自带 clone() 方法，
    CPU 实例用 copy.deepcopy 深拷贝。
    """
    if hasattr(mesh, "cuda"):
        return mesh.clone()
    return copy.deepcopy(mesh)


def reindex_mesh(vertices: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """移除未引用的顶点，重新索引三角面。

    裁剪后的网格常含孤立顶点；本函数按三角面实际引用重建
    (vertices, triangles)，使顶点编号连续且从 0 开始。

    Args:
        vertices: (N, 3) 顶点坐标。
        triangles: (M, 3) 三角面顶点索引。

    Returns:
        (reindexed_vertices, reindexed_triangles): 紧凑化后的网格。
    """
    used = set()
    for tri in triangles:
        used.update(tri)
    sorted_v = sorted(used)
    v_map = {old: new for new, old in enumerate(sorted_v)}
    reindexed_t = np.array([[v_map[int(v)] for v in tri] for tri in triangles], dtype=np.int32)
    reindexed_v = vertices[sorted_v].copy()
    return reindexed_v, reindexed_t


def build_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """用泊松曲面重建从点云重建网格。"""
    logger.info("Input is a point cloud. Reconstructing a mesh for curvature analysis...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals

    pcd.orient_normals_consistent_tangent_plane(100)

    logger.info("Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    logger.info("Cropping mesh based on density to remove artifacts...")
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh


def load_mesh_by_project(project_id: str) -> o3d.geometry.TriangleMesh:
    """按 project_id 加载 mesh 文件（自动匹配 STD_fuse_mesh*.ply）。

    Args:
        project_id: 如 "S0069"。

    Returns:
        TriangleMesh 对象。

    Raises:
        FileNotFoundError: 无匹配文件时。
    """
    import glob

    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mesh_dir = os.path.join(proj_dir, "data", "mesh", project_id)
    pattern = os.path.join(mesh_dir, "STD_fuse_mesh*.ply")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No STD_fuse_mesh*.ply found in {mesh_dir}")
    return o3d.io.read_triangle_mesh(matches[0])


def load_mesh(file_path: str, is_build_mesh: bool = True) -> o3d.geometry.TriangleMesh:
    """从 PLY 文件加载网格。

    Args:
        file_path: PLY 文件路径。
        is_build_mesh: 当文件不含三角形时是否用点云重建网格。

    Returns:
        TriangleMesh 对象。

    Raises:
        FileNotFoundError: 文件不存在时。
        ValueError: 网格无顶点时。
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    mesh: o3d.geometry.TriangleMesh = o3d.io.read_triangle_mesh(file_path)

    if mesh.is_empty():
        logger.error("Error: No vertices found in the final mesh. Aborting.")
        raise ValueError("No vertices found in the final mesh. Aborting.")

    if not mesh.has_triangles() and is_build_mesh:
        logger.info("No triangles found in the mesh. Building a mesh from the point cloud...")
        mesh = build_mesh(mesh)
    return mesh


def preprocess_to_vertices(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """ROI 提取 → Envelope 重建 → 顶点数组。

    Args:
        mesh: 原始 TriangleMesh。

    Returns:
        (N, 3) 预处理后的顶点数组。
    """
    from mesh.preprocess import preprocess_back_scan_mesh
    from mesh.roi_extract import extract_back_roi

    roi = extract_back_roi(mesh)
    processed, _ = preprocess_back_scan_mesh(roi)
    return np.asarray(processed.vertices, dtype=np.float64)


def load_cached_mesh(subject: str, cache_dir: str = "results/cache") -> o3d.geometry.TriangleMesh | None:
    """Load cached mesh from pipeline output (align or extract_roi step)."""
    for subdir in ("align", "extract_roi"):
        path = os.path.join(cache_dir, subject, subdir, "output.ply")
        if os.path.exists(path):
            return o3d.io.read_triangle_mesh(path)
    return None


def lift_2d_to_vertex(vertices: np.ndarray, pts2d: np.ndarray | None) -> np.ndarray | None:
    """将 (N,2) 的 xy 点映射到最近的顶点，返回 (N,3)。

    若传入 pts2d 为 None，返回 None；若 pts2d 已为 (N,3) 则原样返回拷贝。
    仅按 XY 最近邻匹配，不做空间插值。
    """
    if pts2d is None:
        return None
    pts = np.asarray(pts2d)
    if pts.ndim != _XY_NDIM:
        logger.error(f"pts2d has ndim={pts.ndim}, expected 2.")
        raise ValueError("pts2d must be (N,2) or (N,3)")
    if pts.shape[1] == _XYZ_DIM:
        return pts.copy()
    verts_xy = np.asarray(vertices)[:, :2]
    out = np.zeros((pts.shape[0], 3), dtype=np.float64)
    for i, p in enumerate(pts):
        dists = np.sum((verts_xy - p[:2]) ** 2, axis=1)
        idx = int(np.argmin(dists))
        out[i] = np.asarray(vertices)[idx]
    return out


def lift_or_raise(vertices: np.ndarray, pts2d: np.ndarray, name: str = "points") -> np.ndarray:
    """lift_2d_to_vertex 包装，失败时自动 raise ValueError。"""
    result = lift_2d_to_vertex(vertices, pts2d)
    if result is None:
        logger.error(f"Failed to lift {name} points to vertices.")
        raise ValueError(f"Failed to lift {name} points to vertices.")
    return result


def compute_mid_x(vertices: np.ndarray) -> float:
    """从下半身躯干顶点估算体中线 X 坐标。"""
    y = vertices[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    y_range = y_max - y_min
    lower_half = vertices[y < y_min + 0.50 * y_range]
    if len(lower_half) > _MIN_SAMPLE_VERTICES:
        return float((lower_half[:, 0].min() + lower_half[:, 0].max()) / 2.0)
    return float((vertices[:, 0].min() + vertices[:, 0].max()) / 2.0)


def estimate_vertex_radius(vertices: np.ndarray, nb_neighbors: int) -> float:
    """估计顶点邻域半径（用于 radius outlier），返回 median(kth neighbor distance) * 2.

    Args:
        vertices: (N,3) 顶点数组
        nb_neighbors: 邻居数（int）
    """
    from scipy.spatial import KDTree

    tree = KDTree(np.asarray(vertices))
    k = min(nb_neighbors + 1, len(vertices))
    dists, _ = tree.query(vertices, k=k)
    return float(np.median(dists[:, -1]) * 2.0)
