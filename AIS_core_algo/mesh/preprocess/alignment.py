"""
Rigid body alignment / initial registration (paper §3.1.2).

Align a point cloud or mesh to a standard coordinate frame:
  - Torso upright (Y-axis aligned with spine)
  - Centred at back centroid
  - Consistent left-right orientation

Uses ICP (Iterative Closest Point) or PCA-based alignment.

TODO:
  - Implement PCA-based canonical orientation
  - Implement ICP refinement against a reference pose
  - Handle left-right ambiguity detection and correction
"""

import copy

import numpy as np
import open3d as o3d

from utils.logger import logger


def align_mesh(
    mesh: o3d.geometry.TriangleMesh,
    reference: o3d.geometry.TriangleMesh = None,
    icp_max_dist: float = 10.0,
) -> tuple:
    """Align a back mesh to a canonical coordinate frame.

    Pipeline:
      1. PCA — rotate so that the longest axis is Y (spine / height),
         medium axis is X (width), shortest axis is Z (depth into scan).
      2. Centroid translate to origin.
      3. Optional ICP refinement against *reference* if provided.

    Args:
        mesh: Open3D TriangleMesh or PointCloud.
        reference: Optional reference mesh for ICP refinement.
        icp_max_dist: Maximum correspondence distance for ICP.

    Returns:
        aligned_mesh: Aligned Open3D mesh.
        transform: 4×4 rigid transformation matrix (float64 numpy array)
                   that was applied to the mesh.
    """
    vertices = np.asarray(mesh.vertices)
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    # PCA on centred vertices
    cov = (centered.T @ centered) / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order

    # eigenvectors[:,i] = i-th principal axis
    # Smallest eigenvalue → depth (Z), largest → height (Y), mid → width (X)
    new_z = eigenvectors[:, 0]  # smallest variance = normal to back surface
    new_y = eigenvectors[:, 2]  # largest variance  = spine / height direction
    new_x = np.cross(new_y, new_z)
    new_x /= np.linalg.norm(new_x)
    new_y = np.cross(new_z, new_x)
    new_y /= np.linalg.norm(new_y)

    # Ensure consistent orientation:
    #   Y should point upward  (same direction as global Y when possible)
    if np.dot(new_y, np.array([0.0, 1.0, 0.0])) < 0:
        new_y = -new_y
        new_x = -new_x
    #   Z should point away from body (positive Z toward camera)
    if np.dot(new_z, np.array([0.0, 0.0, 1.0])) < 0:
        new_z = -new_z
        new_x = -new_x

    # Rotation: R transforms old frame → new frame; R = [new_x | new_y | new_z]^T
    R = np.stack([new_x, new_y, new_z], axis=0)  # 3×3

    # Full 4×4 transform: translate centroid to origin then rotate
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = -(R @ centroid)

    aligned = copy.deepcopy(mesh)
    aligned.transform(T)

    if reference is not None:
        result = o3d.pipelines.registration.registration_icp(
            aligned,
            reference,
            max_correspondence_distance=icp_max_dist,
            estimation_method=(
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            ),
        )
        aligned.transform(result.transformation)
        T = result.transformation @ T

    return aligned, T


def apply_rotation(
    mesh: o3d.geometry.TriangleMesh,
    x: float = 0,
    y: float = 0,
    z: float = 0,
    in_degrees: bool = True,
) -> o3d.geometry.TriangleMesh:
    """对 mesh 应用旋转矩阵。

    Args:
        mesh: 待旋转的 mesh。
        x, y, z: 旋转角度（度或弧度）。
        in_degrees: 角度是否以度为单位。

    Returns:
        旋转后的 mesh。
    """
    if in_degrees:
        x = np.radians(x)
        y = np.radians(y)
        z = np.radians(z)
    rotation_matrix = mesh.get_rotation_matrix_from_xyz([x, y, z])
    mesh.rotate(rotation_matrix)
    return mesh


def calculate_distance_from_plane(
    mesh: o3d.geometry.TriangleMesh,
    plane_a: float = 0,
    plane_b: float = 0,
    plane_c: float = 1,
    plane_d: float = 0,
) -> np.ndarray:
    """计算顶点到平面 (ax+by+cz+d=0) 的距离。"""
    vertices = np.asarray(mesh.vertices)
    normal_vector = np.array([plane_a, plane_b, plane_c])
    denominator = np.linalg.norm(normal_vector)
    if denominator == 0:
        logger.warning("Plane normal vector is zero. Using default values.")
        normal_vector = np.array([0, 0, 1])
        denominator = 1
    unit_normal = normal_vector / denominator
    return np.abs(np.sum(vertices * unit_normal, axis=1) + plane_d / denominator)
