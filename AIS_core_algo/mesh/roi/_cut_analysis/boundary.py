"""Cut boundary edge detection and segmentation for ROI meshes."""

import numpy as np
from scipy.spatial import KDTree

from utils.logger import logger

from .._mesh_graph import (
    build_edge_to_triangles,
    compute_boundary_edges,
)


def _find_cut_boundary_edges(
    original_vertices: np.ndarray,
    original_triangles: np.ndarray,
    kept_vertices: np.ndarray,
    kept_triangles: np.ndarray,
) -> set[tuple[int, int]]:
    """Find cut boundary edges: kept-mesh boundary edges that are not also
    boundaries of the original mesh.

    Kept-triangle vertices use kept-mesh indexing (0..K-1 after
    ``remove_unreferenced_vertices``), so we must map back to original
    vertex indices via KDTree before comparing with orig_boundary.

    Returns a set of (a, b) with a < b, using original-mesh vertex IDs.
    """
    # Boundary edges of the kept mesh (kept vertex indices)
    kept_edges = build_edge_to_triangles(kept_triangles)
    kept_boundary = {e for e, tris in kept_edges.items() if len(tris) == 1}

    # Map kept→original vertex indices
    tree = KDTree(original_vertices)
    _, kept_to_orig = tree.query(kept_vertices)
    kept_v_orig = [int(kept_to_orig[i]) for i in range(len(kept_vertices))]

    # Boundary edges of the original mesh (original vertex indices)
    orig_boundary = compute_boundary_edges(original_triangles)

    # Edge→tri index for the original mesh — verify edge exists
    orig_edge_to_tris = build_edge_to_triangles(original_triangles)

    cut_edges: set[tuple[int, int]] = set()
    for a, b in kept_boundary:
        oa, ob = kept_v_orig[a], kept_v_orig[b]
        if oa == ob:
            continue
        edge = (oa, ob) if oa < ob else (ob, oa)
        if edge not in orig_boundary and edge in orig_edge_to_tris:
            cut_edges.add(edge)

    logger.info("Cut boundary edges: %d", len(cut_edges))
    return cut_edges


def _group_edges_into_segments(
    cut_edges: set[tuple[int, int]],
) -> list[list[int]]:
    """Group cut boundary edges into connected vertex segments.

    Returns a list of segments, each being a list of vertex IDs (original
    indices) comprising one connected component of the cut boundary.
    """
    adj: dict[int, set[int]] = {}
    for a, b in cut_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    all_verts = list(adj.keys())
    visited: set[int] = set()
    segments: list[list[int]] = []
    for v in all_verts:
        if v in visited:
            continue
        stack = [v]
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    stack.append(nb)
        segments.append(comp)

    logger.info("Cut boundary segments: %d", len(segments))
    return segments


def _filter_segments_by_length(
    segments: list[list[int]],
    cut_edges: set[tuple[int, int]],
    original_vertices: np.ndarray,
    min_seg_length: float,
) -> tuple[list[list[int]], set[tuple[int, int]]]:
    """Filter segments shorter than min_seg_length.

    Returns (filtered_segments, filtered_cut_edges).
    """
    seen: set[int] = set()
    filtered_segments: list[list[int]] = []
    for seg_vids in segments:
        seg_set = set(seg_vids)
        edges = [e for e in cut_edges if e[0] in seg_set and e[1] in seg_set]
        seg_len = sum(float(np.linalg.norm(original_vertices[a] - original_vertices[b])) for a, b in edges)
        if seg_len >= min_seg_length:
            seen.update(seg_vids)
            filtered_segments.append(seg_vids)
    filtered_cut_edges = {e for e in cut_edges if e[0] in seen and e[1] in seen}
    return filtered_segments, filtered_cut_edges
