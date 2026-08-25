"""BFS seed-growing + adjacency helpers + largest-component filter."""

from collections import deque

import numpy as np
import open3d as o3d

from utils.logger import logger

from ._mesh_graph import (
    build_edge_to_triangles,
    build_triangle_adjacency,
    compute_triangle_areas,
    find_connected_components,
)

# Module-level roughness cache keyed by mesh id()
_roughness_cache: dict[int, np.ndarray] = {}

_NORM_EPSILON = 1e-10      # 平均法线长度小于该值视为退化（不归一化）
_MIN_FACE_COUNT = 3        # 顶点至少关联 3 个三角面（否则粗糙度置零）
_MIN_COMPONENT_TRIANGLES = 2  # 至少 2 个三角面才谈连通分量


# ── Triangle adjacency helpers ────────────────────────────────────────────────


def _compute_triangle_normals(
    mesh: o3d.geometry.TriangleMesh,
) -> np.ndarray:
    """Compute triangle normals without mutating the input mesh."""
    cloned = o3d.geometry.TriangleMesh()
    cloned.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    cloned.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
    cloned.compute_triangle_normals()
    return np.asarray(cloned.triangle_normals)


# ── Seed selection (roughness-based) ────────────────────────────────────────────


def _select_seed(
    roughness: np.ndarray,
    tri_centers: np.ndarray,
    adj: list[set[int]],
    roughness_threshold: float = 0.25,
    min_component_tris: int = 500,
    center_weight: float = 0.4,
) -> int:
    """Choose the best seed triangle for BFS region growing.

    Uses a stricter roughness threshold (60% of the BFS growth threshold)
    to find the smoothest region as the seed — ensures the BFS starts
    on a reliably smooth patch of the back, not near rough clothing.

    Strategy:
    1. Find every connected component of low-roughness triangles (roughness < threshold × 0.6).
    2. Discard tiny components (< *min_component_tris* tris) — isolated smooth patches.
    3. Score each viable component by size × center proximity.
    4. Within the winning component, pick the triangle nearest its centre.

    Parameters
    ----------
    roughness_threshold:
        BFS growth threshold; 60% of this is used as the seed smoothness cutoff.
    center_weight:
        Blend between size score (0 = pure size) and center proximity
        (1 = pure proximity).  Default 0.4 balances picking a large region
        near the mesh centre rather than an extreme but large flat patch.

    Returns seed triangle index.
    """
    n_tri = len(roughness)
    seed_cutoff = roughness_threshold * 0.6
    good = roughness < seed_cutoff

    # Find connected components within the good mask.
    visited = np.zeros(n_tri, dtype=bool)
    components: list[list[int]] = []
    for ti in range(n_tri):
        if not good[ti] or visited[ti]:
            continue
        stack = [ti]
        visited[ti] = True
        comp: list[int] = []
        while stack:
            t = stack.pop()
            comp.append(t)
            for nj in adj[t]:
                if good[nj] and not visited[nj]:
                    visited[nj] = True
                    stack.append(nj)
        components.append(comp)

    if not components:
        # Fallback: lowest roughness in the whole mesh.
        return int(np.argmin(roughness))

    # Discard tiny components (isolated smooth patches / noise fragments).
    viable = [c for c in components if len(c) >= min_component_tris]
    if not viable:
        viable = [max(components, key=len)]  # use the least-bad one

    if len(viable) == 1:
        best_comp = viable[0]
    else:
        # Score by relative size × centre proximity.
        mesh_centre = tri_centers.mean(axis=0)
        max_size = max(len(c) for c in viable)
        xy_ranges = tri_centers[:, :2].ptp(axis=0)
        max_dist = float(np.linalg.norm(xy_ranges))

        def _score(comp: list[int]) -> float:
            size_s = len(comp) / max_size
            comp_centre = tri_centers[comp].mean(axis=0)
            dist = float(np.linalg.norm(comp_centre[:2] - mesh_centre[:2]))
            dist_s = 1.0 - dist / max(max_dist, 1e-8)
            return size_s * (1.0 - center_weight) + dist_s * center_weight

        viable.sort(key=_score, reverse=True)
        best_comp = viable[0]

    # Within the winning component, pick triangle nearest its centre.
    comp_centre = tri_centers[best_comp].mean(axis=0)
    best_tri = min(best_comp, key=lambda ti: float(np.linalg.norm(tri_centers[ti] - comp_centre)))
    return best_tri


# ── Hole filling (boundary-based) ────────────────────────────────────────────


def _fill_holes_by_boundary(
    component: set[int],
    vertices: np.ndarray,
    triangles: np.ndarray,
    adj: list[set[int]],
    max_hole_boundary: float = 200.0,
    max_hole_area: float = 3000.0,
    edge_to_tri: dict | None = None,
    tri_areas: np.ndarray | None = None,
) -> set[int]:
    """按外边界长度/面积补洞：找到 removed 连通分量的外边界，小的才补。

    真正的孔洞检测：
    1. 找出所有 removed 三角形的连通分量。
    2. 对每个分量，计算其外边界的总长度（与 kept 共享的边）和总面积。
    3. 如果两者都小于阈值，则视为内部空洞，全部补上。

    Parameters
    ----------
    component: 当前 kept 三角形索引集合。
    vertices: (n_vert, 3) 顶点数组。
    triangles: (n_tri, 3) 三角面数组。
    adj: 三角形邻接表。
    max_hole_boundary: 孔洞外边界最大长度（mm），默认 200mm。
    max_hole_area: 孔洞最大面积（mm²），默认 3000mm²。
    edge_to_tri: 预计算的边→三角面字典，为 None 则内部构建。
    tri_areas: 预计算的三角面面积数组，为 None 则内部计算。

    Returns 补洞后的 component set。
    """
    result = set(component)
    n_tri = len(triangles)
    all_tri_set = set(range(n_tri))

    # 找出所有 removed 三角形的连通分量
    removed_set = all_tri_set - result
    if not removed_set:
        return result

    if edge_to_tri is None:
        edge_to_tri = build_edge_to_triangles(triangles)
    if tri_areas is None:
        tri_areas = compute_triangle_areas(vertices, triangles)

    # 找出所有 removed 三角形的连通分量（完整 BFS，不早退，保证正确性）
    visited: set[int] = set()
    hole_components: list[list[int]] = []
    for seed in removed_set:
        if seed in visited:
            continue
        comp: list[int] = []
        stack = [seed]
        visited.add(seed)
        while stack:
            ti = stack.pop()
            comp.append(ti)
            for nj in adj[ti]:
                if nj in removed_set and nj not in visited:
                    visited.add(nj)
                    stack.append(nj)
        hole_components.append(comp)

    n_filled = 0
    for hole in hole_components:
        # 先算面积（O(n) C 级操作），面积超阈值则跳过边界长度计算（Python 循环较慢）
        hole_area = float(np.sum(tri_areas[hole]))
        if hole_area > max_hole_area:
            continue

        # 计算该孔洞的外边界长度（与 kept 共享的边的总长度）
        boundary_len = 0.0
        for ti in hole:
            a, b, c = triangles[ti]
            for edge in [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))]:
                for nj in edge_to_tri.get(edge, []):
                    if nj in result:  # 相邻的是 kept → 这条边是外边界
                        boundary_len += float(np.linalg.norm(vertices[edge[0]] - vertices[edge[1]]))
                        break

        if boundary_len <= max_hole_boundary:
            for ti in hole:
                result.add(ti)
            n_filled += 1

    if n_filled > 0:
        logger.info(
            "Hole fill by boundary: filled %d holes (max_len=%.0fmm max_area=%.0fmm²)",
            n_filled,
            max_hole_boundary,
            max_hole_area,
        )
    return result


# ── High-performance roughness pre-computation ────────────────────────────────

# ── High-performance roughness pre-computation ────────────────────────────────


def _compute_face_normals(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Fully vectorized face normals — no open3d clone."""
    v0, v1, v2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    return fn


def compute_mesh_roughness(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Pre‑compute per‑triangle roughness, fully vectorised.

    Strategy (vertex roughness → max‑pool to triangles):
      1. Compute face normals via numpy cross‑product (vectorised).
      2. Per **vertex**: scatter‑add face normals → mean normal →
         scatter‑max angular deviation of incident faces.
         (Fully vectorised — no Python loop over 850k vertices.)
      3. Per **triangle**: max of its 3 vertex roughnesses (numpy, 0‑copy).

    Returns
    -------
    roughness : ndarray, shape (n_tri,) — radians.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    t = np.asarray(mesh.triangles)
    n_vert = len(v)

    # ── 1. Face normals (fully vectorised) ──
    fn = _compute_face_normals(v, t)

    # ── 2. Per‑vertex mean normal (scatter‑add, fully vectorised) ──
    fn_sum = np.zeros((n_vert, 3), dtype=np.float64)
    for c in range(3):
        np.add.at(fn_sum[:, c], t[:, 0], fn[:, c])
        np.add.at(fn_sum[:, c], t[:, 1], fn[:, c])
        np.add.at(fn_sum[:, c], t[:, 2], fn[:, c])

    face_count = np.bincount(t.ravel(), minlength=n_vert)
    mean_norm = fn_sum / np.maximum(face_count[:, None], 1)
    norm = np.linalg.norm(mean_norm, axis=1)
    good = norm > _NORM_EPSILON
    if good.any():
        mean_norm[good] /= norm[good, None]

    # ── 3. Per‑vertex roughness (scatter‑max, fully vectorised) ──
    dot0 = np.sum(fn * mean_norm[t[:, 0]], axis=1)
    dot1 = np.sum(fn * mean_norm[t[:, 1]], axis=1)
    dot2 = np.sum(fn * mean_norm[t[:, 2]], axis=1)

    np.clip(dot0, -1.0, 1.0, out=dot0)
    np.clip(dot1, -1.0, 1.0, out=dot1)
    np.clip(dot2, -1.0, 1.0, out=dot2)

    vert_rough = np.zeros(n_vert, dtype=np.float64)
    np.maximum.at(vert_rough, t[:, 0], np.arccos(dot0))
    np.maximum.at(vert_rough, t[:, 1], np.arccos(dot1))
    np.maximum.at(vert_rough, t[:, 2], np.arccos(dot2))
    vert_rough[face_count < _MIN_FACE_COUNT] = 0.0

    # ── 4. Triangle‑level → max of its 3 vertices (0‑copy) ──
    roughness = np.maximum(vert_rough[t[:, 0]], vert_rough[t[:, 1]])
    roughness = np.maximum(roughness, vert_rough[t[:, 2]])

    logger.info(
        "Roughness(vert‑max): range=[%.4f, %.4f] rad, median=%.4f",
        float(roughness.min()),
        float(roughness.max()),
        float(np.median(roughness)),
    )
    return roughness


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
