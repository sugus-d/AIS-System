"""BFS seed-growing + adjacency helpers + largest-component filter.

Module structure
----------------
- _bfs_seed.py:      _compute_triangle_normals, _select_seed
- _bfs_holes.py:     _fill_holes_by_boundary
- _bfs_roughness.py: _compute_face_normals, compute_mesh_roughness
- _bfs_impl.py:      mesh_bfs, largest_component (this file)
"""

from collections import deque

import numpy as np
import open3d as o3d

from utils.logger import logger

from ._bfs_holes import _fill_holes_by_boundary
from ._bfs_roughness import compute_mesh_roughness
from ._bfs_seed import _compute_triangle_normals, _select_seed
from ._mesh_graph import (
    build_edge_to_triangles,
    build_triangle_adjacency,
    compute_triangle_areas,
    find_connected_components,
)

# Module-level roughness cache keyed by mesh id()
_roughness_cache: dict[int, np.ndarray] = {}

_MIN_COMPONENT_TRIANGLES = 2  # 至少 2 个三角面才谈连通分量


# ── BFS ───────────────────────────────────────────────────────────────────────


def mesh_bfs(
    mesh: o3d.geometry.TriangleMesh,
    angle_threshold_deg: float = 45.0,
    roughness_threshold: float = 0.25,
    roughness_radius: float = 20.0,
    fill_holes: bool = True,
    max_hole_boundary: float = 100.0,
    max_hole_area: float = 3000.0,
) -> o3d.geometry.TriangleMesh:
    """BFS region growing on mesh triangles — stops at rough / non-smooth areas.

    Two stopping constraints:

    1. **Local edge** — angle between normals of adjacent triangles exceeds
       *angle_threshold_deg* (prevents crossing sharp creases).
    2. **Local roughness** — a triangle's neighborhood normal dispersion
       exceeds *roughness_threshold* (stops at the boundary between smooth
       back and rough clothing/fabric).

    Seed is chosen from the smoothest region (roughness < threshold × 0.6).
    After BFS, optional **hole-filling** detects small internal holes by
    boundary length (``≤max_hole_boundary``) and area (``≤max_hole_area``)
    and fills them.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles)
    n_tri = len(triangles)
    if n_tri == 0:
        return o3d.geometry.TriangleMesh()

    # Build edge→tri map ONCE — reused by adj, blocking hole‑filling & smoothing
    edge_to_tri = build_edge_to_triangles(triangles)
    adj = build_triangle_adjacency(triangles, edge_to_tri)
    tri_normals = _compute_triangle_normals(mesh)
    tri_centers = vertices[triangles].mean(axis=1)

    # Use cached roughness if available, else compute once
    # Key includes vertex/triangle count to guard against reused id()
    cache_key = (id(mesh), n_tri)
    roughness = _roughness_cache.get(cache_key)
    if roughness is None:
        roughness = compute_mesh_roughness(mesh)
        _roughness_cache[cache_key] = roughness

    # Pre‑compute triangle areas once — reused by hole‑filling
    tri_areas = compute_triangle_areas(vertices, triangles)

    seed_tri = _select_seed(roughness, tri_centers, adj, roughness_threshold)
    logger.info(
        "Seed tri=%d roughness=%.4f at (%.0f, %.0f)",
        seed_tri,
        roughness[seed_tri],
        tri_centers[seed_tri, 0],
        tri_centers[seed_tri, 1],
    )
    cos_threshold = float(np.cos(np.radians(angle_threshold_deg)))

    visited = set()
    queue = deque([seed_tri])
    visited.add(seed_tri)
    component = set([seed_tri])

    while queue:
        ti = queue.popleft()
        ni = tri_normals[ti]
        for nj in adj[ti]:
            if nj in visited:
                continue
            visited.add(nj)
            # 局部尖锐边检查：相邻面法向量夹角过大
            dot = float(np.dot(ni, tri_normals[nj]))
            if dot < cos_threshold:
                continue
            # 局部粗糙度：邻域法向量散布超过阈值 → 衣服/织物区域
            if roughness[nj] > roughness_threshold:
                continue
            component.add(nj)
            queue.append(nj)

    # ── 形态学补洞：补全完全被 kept 包围的 removed 三角形 ──
    if fill_holes:
        component = _fill_holes_by_boundary(
            component,
            vertices,
            triangles,
            adj,
            max_hole_boundary=max_hole_boundary,
            max_hole_area=max_hole_area,
            edge_to_tri=edge_to_tri,
            tri_areas=tri_areas,
        )

    keep = np.array(sorted(component), dtype=int)
    result = o3d.geometry.TriangleMesh()
    result.vertices = mesh.vertices
    result.triangles = o3d.utility.Vector3iVector(triangles[keep])
    result.remove_unreferenced_vertices()
    result.compute_vertex_normals()
    return result


# ── Largest component ─────────────────────────────────────────────────────────


def largest_component(
    mesh: o3d.geometry.TriangleMesh,
    adj: list[set[int]] | None = None,
) -> o3d.geometry.TriangleMesh:
    """Keep only the largest connected component by triangle adjacency.

    Parameters
    ----------
    adj:
        Optional pre-built triangle adjacency (list[set[int]]).
        If not provided, it is built internally.
        Pass from a caller that already computed it to avoid O(N) rebuild.
    """
    triangles = np.asarray(mesh.triangles)
    if len(triangles) < _MIN_COMPONENT_TRIANGLES:
        return mesh

    if adj is None:
        adj = build_triangle_adjacency(triangles)
    components = find_connected_components(adj)

    if len(components) <= 1:
        return mesh

    keep = np.array(sorted(components[0]), dtype=int)
    result = o3d.geometry.TriangleMesh()
    result.vertices = mesh.vertices
    result.triangles = o3d.utility.Vector3iVector(triangles[keep])
    result.remove_unreferenced_vertices()
    result.compute_vertex_normals()
    logger.info(f"Largest component: {len(components[0])}/{len(triangles)} tris")
    return result
