"""Removed-triangle region detection and connected-component grouping."""

import numpy as np
from scipy.spatial import KDTree

from .._mesh_graph import find_connected_components

_MIN_SHARED_TRIS = 2  # 至少 2 个三角面共享边才构成邻接关系


def _mark_removed_adjacent_triangles(
    cut_edges: set[tuple[int, int]],
    edge_to_tris: dict[tuple[int, int], list[int]],
    kept_vertices: np.ndarray,
    original_vertices: np.ndarray,
    original_triangles: np.ndarray,
    n_total: int,
) -> np.ndarray:
    """Mark triangles adjacent to cut edges that are NOT in the kept mesh."""
    tree = KDTree(original_vertices)
    _, nearest = tree.query(kept_vertices)
    kept_v_set = set(nearest.tolist())

    removed_adjacent: set[int] = set()
    for e in cut_edges:
        for ti in edge_to_tris.get(e, []):
            tri = original_triangles[ti]
            is_removed = int(tri[0]) not in kept_v_set or int(tri[1]) not in kept_v_set or int(tri[2]) not in kept_v_set
            if is_removed:
                removed_adjacent.add(ti)
    mask = np.zeros(n_total, dtype=bool)
    for ti in removed_adjacent:
        mask[ti] = True
    return mask


def _build_removed_components(
    removed_indices: list[int],
    edge_to_tris: dict[tuple[int, int], list[int]],
) -> list[list[int]]:
    """Build connected components from a set of removed triangle indices."""
    removed_lookup = {ti: i for i, ti in enumerate(removed_indices)}
    adj: list[set[int]] = [set() for _ in range(len(removed_indices))]
    for _, tris in edge_to_tris.items():
        if len(tris) < _MIN_SHARED_TRIS:
            continue
        relevant = [ti for ti in tris if ti in removed_lookup]
        if len(relevant) < _MIN_SHARED_TRIS:
            continue
        for ti in relevant:
            for tj in relevant:
                if tj != ti:
                    ri = removed_lookup[ti]
                    rj = removed_lookup[tj]
                    adj[ri].add(rj)
    raw_components = find_connected_components(adj)
    components: list[list[int]] = []
    for comp in raw_components:
        orig_indices = [removed_indices[i] for i in comp]
        components.append(orig_indices)
    return components


def _build_comp_vertex_sets(
    removed_components: list[list[int]],
    original_triangles: np.ndarray,
) -> list[set[int]]:
    """Build vertex set for each removed component."""
    comp_vertex_sets: list[set[int]] = []
    for comp in removed_components:
        vset: set[int] = set()
        for ti in comp:
            vset.update(int(v) for v in original_triangles[ti])
        comp_vertex_sets.append(vset)
    return comp_vertex_sets


def _build_comp_to_segments(
    segments: list[list[int]],
    comp_vertex_sets: list[set[int]],
) -> list[set[int]]:
    """Build mapping: for each removed component, which segments touch it."""
    comp_to_segs: list[set[int]] = [set() for _ in range(len(comp_vertex_sets))]
    for si, seg_vids in enumerate(segments):
        seg_set = set(seg_vids)
        for ci, comp_vset in enumerate(comp_vertex_sets):
            if seg_set & comp_vset:
                comp_to_segs[ci].add(si)
    return comp_to_segs
