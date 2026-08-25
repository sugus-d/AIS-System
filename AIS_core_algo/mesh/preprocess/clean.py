"""Mesh cleaning utilities: denoise, fill holes, smooth."""

import copy

import numpy as np
import open3d as o3d
import pyvista as pv

from utils.mesh import estimate_vertex_radius


def denoise_mesh(
    mesh: o3d.geometry.TriangleMesh,
    method: str = "statistical",
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    iterations: int = 1,
) -> o3d.geometry.TriangleMesh:
    """Remove noise / outlier vertices from a mesh or point cloud."""
    if not mesh.has_triangles():
        return _denoise_pointcloud(mesh, method, nb_neighbors, std_ratio, iterations)

    result = copy.deepcopy(mesh)
    for _ in range(iterations):
        vertices = np.asarray(result.vertices)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)

        if method == "statistical":
            _, keep_idx = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio
            )
        elif method == "radius":
            radius = estimate_vertex_radius(vertices, nb_neighbors)
            _, keep_idx = pcd.remove_radius_outlier(
                nb_points=nb_neighbors, radius=radius
            )
        else:
            raise ValueError(
                f"Unknown method {method!r}. Use 'statistical' or 'radius'."
            )

        result = result.select_by_index(keep_idx)

    result.remove_unreferenced_vertices()
    return result


def _denoise_pointcloud(
    mesh: o3d.geometry.TriangleMesh,
    method: str,
    nb_neighbors: int,
    std_ratio: float,
    iterations: int,
) -> o3d.geometry.TriangleMesh:
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    if mesh.has_vertex_normals():
        pcd.normals = mesh.vertex_normals
    if mesh.has_vertex_colors():
        pcd.colors = mesh.vertex_colors

    for _ in range(iterations):
        if method == "statistical":
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio
            )
        elif method == "radius":
            radius = estimate_vertex_radius(np.asarray(pcd.points), nb_neighbors)
            pcd, _ = pcd.remove_radius_outlier(nb_points=nb_neighbors, radius=radius)

    result = o3d.geometry.TriangleMesh()
    result.vertices = pcd.points
    if pcd.has_normals():
        result.vertex_normals = pcd.normals
    if pcd.has_colors():
        result.vertex_colors = pcd.colors
    return result


# _estimate_radius 已移动到 utils.mesh.estimate_vertex_radius


def fill_mesh_holes(
    mesh: o3d.geometry.TriangleMesh,
    hole_size: int = 500,
    max_edge_multiplier: float = 5.0,
) -> o3d.geometry.TriangleMesh:
    """Fill open boundary holes in a triangle mesh."""
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    pv_faces = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces])
    pv_mesh = pv.PolyData(vertices.astype(np.float64), pv_faces)

    pts = pv_mesh.points
    f = pv_mesh.faces.reshape(-1, 4)[:, 1:]
    edge_lengths = np.concatenate([
        np.linalg.norm(pts[f[:, 1]] - pts[f[:, 0]], axis=1),
        np.linalg.norm(pts[f[:, 2]] - pts[f[:, 1]], axis=1),
        np.linalg.norm(pts[f[:, 0]] - pts[f[:, 2]], axis=1),
    ])
    median_edge = float(np.median(edge_lengths))
    edge_limit = median_edge * max_edge_multiplier

    if pv_mesh.n_open_edges > 0:
        pv_mesh = pv_mesh.fill_holes(hole_size=hole_size)

        f2 = pv_mesh.faces.reshape(-1, 4)[:, 1:]
        pts2 = pv_mesh.points
        max_e = np.maximum(
            np.maximum(
                np.linalg.norm(pts2[f2[:, 1]] - pts2[f2[:, 0]], axis=1),
                np.linalg.norm(pts2[f2[:, 2]] - pts2[f2[:, 1]], axis=1),
            ),
            np.linalg.norm(pts2[f2[:, 0]] - pts2[f2[:, 2]], axis=1),
        )
        good = f2[max_e < edge_limit]
        pv_mesh = pv.PolyData(pts2, np.hstack([np.full((len(good), 1), 3), good]))

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(np.array(pv_mesh.points))
    out.triangles = o3d.utility.Vector3iVector(
        pv_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
    )
    out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    return out


def smooth_mesh(
    mesh: o3d.geometry.TriangleMesh,
    method: str = "laplacian",
    iterations: int = 10,
    lambda_filter: float = 0.5,
    mu: float = -0.53,
) -> o3d.geometry.TriangleMesh:
    """Smooth a triangle mesh."""
    if not mesh.has_triangles():
        raise ValueError("smooth_mesh requires a mesh with triangles.")

    result = copy.deepcopy(mesh)

    if method == "laplacian":
        result = result.filter_smooth_laplacian(
            number_of_iterations=iterations,
            lambda_filter=lambda_filter,
        )
    elif method == "taubin":
        result = result.filter_smooth_taubin(
            number_of_iterations=iterations,
            lambda_filter=lambda_filter,
            mu=mu,
        )
    elif method == "simple":
        result = result.filter_smooth_simple(number_of_iterations=iterations)
    else:
        raise ValueError(
            f"Unknown method {method!r}. Use 'laplacian', 'taubin', or 'simple'."
        )

    result.compute_vertex_normals()
    return result
