"""BFS seed selection — roughness-based seed triangle selection."""

import numpy as np
import open3d as o3d


def _compute_triangle_normals(
    mesh: o3d.geometry.TriangleMesh,
) -> np.ndarray:
    """Compute triangle normals without mutating the input mesh."""
    cloned = o3d.geometry.TriangleMesh()
    cloned.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    cloned.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
    cloned.compute_triangle_normals()
    return np.asarray(cloned.triangle_normals)


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
