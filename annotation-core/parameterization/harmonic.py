r"""Harmonic (Laplacian) parameterization of a triangle mesh onto [0,1] UV space.

Solves Laplace's equation :math:`\nabla^2 u = 0` on the mesh surface with
Dirichlet boundary conditions at landmark vertices.

.. math::
    \text{argmin}_{\mathbf{u}} \frac{1}{2} \mathbf{u}^\top L \mathbf{u}
    \quad\text{s.t.}\quad \mathbf{u}[k] = y_k

where :math:`L` is the cotangent Laplacian.  The same solve is performed
independently for the U and V coordinates; the Z coordinate of every vertex
is preserved unchanged.

When the mesh has disconnected components, the largest component
containing any landmark is used for the solve.  Vertices on other
components receive UV = (0, 0).
"""

import gpytoolbox as gpy
import numpy as np
import open3d as o3d
from scipy.sparse import csgraph, csr_matrix

_MIN_LANDMARK_CONSTRAINTS = 4  # Dirichlet 边界约束最少地标数


def _extract_largest_lm_component(
    vertices: np.ndarray,
    faces: np.ndarray,
    landmark_vertex_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取包含地标的最大连通分量子网格。

    当网格有多个断开的连通分量时，只保留包含最多地标的分量。

    Args:
        vertices:          (N, 3) 原始网格顶点。
        faces:             (M, 3) 三角面。
        landmark_vertex_ids: (L,) 地标在原始网格中的顶点索引。

    Returns:
        (comp_verts, comp_faces, old_to_new):
            comp_verts: (K, 3) 子网格顶点 (K ≤ N)。
            comp_faces: (P, 3) 子网格三角面（索引已重映射）。
            old_to_new: (N,) 原始顶点到子网格顶点的映射（不在子网格中的为 -1）。
    """
    # Build adjacency from edges.
    edges: set[tuple[int, int]] = set()
    for a, b, c in faces:
        edges.add((int(a), int(b)))
        edges.add((int(b), int(c)))
        edges.add((int(a), int(c)))
    rows, cols = zip(*edges, strict=True)
    adj = csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)),
        shape=(len(vertices), len(vertices)),
    )
    adj = (adj + adj.T) > 0
    n_comp, labels = csgraph.connected_components(adj, directed=False)

    if n_comp == 1:
        return vertices, faces, np.arange(len(vertices), dtype=np.int64)

    # Find component with most landmarks.
    lm_labels = labels[landmark_vertex_ids]
    comp_count = np.bincount(lm_labels, minlength=n_comp)
    target = int(np.argmax(comp_count))

    mask = labels == target
    old_idx = np.where(mask)[0]
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    old_to_new[old_idx] = np.arange(len(old_idx))
    old_to_new_set = set(old_idx)

    # Filter faces: keep only those where all 3 vertices are in target comp.
    keep = np.array(
        [f[0] in old_to_new_set and f[1] in old_to_new_set and f[2] in old_to_new_set for f in faces], dtype=bool
    )
    comp_faces = np.array(
        [
            [old_to_new[int(a)], old_to_new[int(b)], old_to_new[int(c)]]
            for (a, b, c), keep_flag in zip(faces, keep, strict=True)
            if keep_flag
        ],
        dtype=np.int64,
    )
    comp_verts = vertices[old_idx]

    return comp_verts, comp_faces, old_to_new


PARAM_METHODS = ("harmonic", "biharmonic")


def harmonic_parameterize(
    mesh: o3d.geometry.TriangleMesh,
    landmark_vertex_ids: np.ndarray,
    landmark_uv: np.ndarray,
    method: str = "biharmonic",
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """通过调和映射将三角网格参数化到 [0,1] UV 空间。

    从网格几何构建余切拉普拉斯算子，然后在固定地标约束下，
    分别对 U 和 V 坐标独立求解二次规划。

    自动处理扫描产生的断连分量：仅对包含地标的最大分量求解，
    孤立分量上的顶点 UV = (0, 0)。

    Args:
        mesh:               输入三角网格，必须至少有一个三角面。
        landmark_vertex_ids: (M,) 作为 Dirichlet 边界条件的顶点索引，至少需要 4 个。
        landmark_uv:         (M, 2) 各地标顶点的目标 (u, v) 坐标。

    Returns:
        (registered_mesh, uv_coords):
            registered_mesh: 新 TriangleMesh，连通性和 Z 坐标与输入相同，
                             XY 坐标被替换为计算出的 UV 坐标。
            uv_coords:    (N, 2) 每个顶点的 UV 坐标。

    Raises:
        ValueError: 网格无三角面或地标少于 4 个。
        IndexError: 地标顶点索引越界。
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)

    if faces.shape[0] == 0:
        raise ValueError("Mesh has no triangles. Harmonic parameterization requires a triangulated mesh.")

    num_landmarks = len(landmark_vertex_ids)
    if num_landmarks < _MIN_LANDMARK_CONSTRAINTS:
        raise ValueError(f"At least 4 landmark constraints are required, got {num_landmarks}.")

    num_vertices = vertices.shape[0]
    if np.any(landmark_vertex_ids < 0) or np.any(landmark_vertex_ids >= num_vertices):
        raise IndexError(f"Landmark vertex IDs must be in [0, {num_vertices}).")

    # Handle disconnected components by extracting the main mesh.
    comp_verts, comp_faces, old_to_new = _extract_largest_lm_component(vertices, faces, landmark_vertex_ids)
    on_main = old_to_new >= 0

    # Remap landmark vertex IDs to the submesh.
    comp_lm_ids = old_to_new[landmark_vertex_ids]
    # Snap any landmark on a different component to nearest vertex on main component
    if np.any(comp_lm_ids < 0):
        bad = np.where(comp_lm_ids < 0)[0]
        for bi in bad:
            pt = vertices[int(landmark_vertex_ids[bi])]
            dists = np.linalg.norm(comp_verts - pt, axis=1)
            comp_lm_ids[bi] = int(np.argmin(dists))
        from utils.logger import logger

        logger.warning(f"Snapped {len(bad)} landmark(s) to main mesh component")
    # Deduplicate (snapping may cause collisions – keep first occurrence)
    uniq_idx, uniq_pos = np.unique(comp_lm_ids, return_index=True)
    comp_lm_ids = uniq_idx
    landmark_uv = landmark_uv[uniq_pos]

    # Assemble cotangent Laplacian on the main component.
    l_sq = gpy.halfedge_lengths_squared(comp_verts, comp_faces)
    laplacian = gpy.cotangent_laplacian_intrinsic(l_sq, comp_faces, n=comp_verts.shape[0])

    # Use L² (biharmonic) or L (harmonic) as the energy matrix.
    quad_matrix = laplacian @ laplacian if method == "biharmonic" else laplacian

    # Solve for U and V independently.
    u_coord = gpy.min_quad_with_fixed(
        quad_matrix,
        k=comp_lm_ids,
        y=landmark_uv[:, 0],
    )
    v_coord = gpy.min_quad_with_fixed(
        quad_matrix,
        k=comp_lm_ids,
        y=landmark_uv[:, 1],
    )

    # Map results back to full vertex array (orphan components get 0,0).
    uv_coords = np.zeros((num_vertices, 2), dtype=np.float64)
    uv_coords[on_main, 0] = u_coord
    uv_coords[on_main, 1] = v_coord

    # Build output mesh: UV in XY, original Z.
    registered_mesh = o3d.geometry.TriangleMesh()
    output_vertices = np.column_stack([uv_coords, vertices[:, 2]])
    registered_mesh.vertices = o3d.utility.Vector3dVector(output_vertices)
    registered_mesh.triangles = mesh.triangles
    registered_mesh.compute_vertex_normals()

    return registered_mesh, uv_coords
