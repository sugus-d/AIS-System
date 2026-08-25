"""Cut boundary analysis and invalid-cut restoration for ROI meshes.

All functions operate on np.ndarray (not open3d objects) and accept
all thresholds as parameters — no hardcoded values.

Dependencies
------------
mesh_graph
    Builds edge indexes, adjacency, connected components, triangle areas.
scipy.spatial.KDTree
    Maps kept-vertex arrays back to original-vertex indices.
"""

import numpy as np
from scipy.spatial import KDTree

from utils.logger import logger

from .._mesh_graph import (
    build_edge_to_triangles,
    compute_triangle_areas,
)
from .boundary import (
    _filter_segments_by_length,
    _find_cut_boundary_edges,
    _group_edges_into_segments,
)
from .classification import _classify_segments_by_validity
from .regions import _build_removed_components


def compute_removed_triangles(
    original_vertices: np.ndarray,
    original_triangles: np.ndarray,
    kept_vertices: np.ndarray,
) -> list[int]:
    """Return indices of triangles removed from the original mesh.

    A triangle is considered removed if *any* of its three vertices is
    not present in the kept mesh (within a coordinate tolerance).

    Parameters
    ----------
    original_vertices : np.ndarray, shape (M, 3)
    original_triangles : np.ndarray, shape (N, 3), int
    kept_vertices : np.ndarray, shape (K, 3)

    Returns
    -------
    list[int]
        Triangle indices (into original_triangles) that are removed.
    """
    tree = KDTree(original_vertices)
    _, nearest = tree.query(kept_vertices)
    kept_set = set(nearest.tolist())

    removed: list[int] = []
    for ti in range(len(original_triangles)):
        tri = original_triangles[ti]
        if int(tri[0]) not in kept_set or int(tri[1]) not in kept_set or int(tri[2]) not in kept_set:
            removed.append(ti)

    logger.info("Removed triangles: %d / %d", len(removed), len(original_triangles))
    return removed


def _empty_cut_analysis() -> dict:
    """Return an empty cut analysis result."""
    return {
        "segments": [],
        "removals": [],
        "total_cut_length_mm": 0.0,
        "total_removed_area_mm2": 0.0,
        "cut_edges": [],
    }


def _finalize_cut_analysis(
    seg_results: list[dict],
    rem_results: list[dict],
    cut_edges: set[tuple[int, int]] | None = None,
) -> dict:
    """Compute totals and format final analysis result."""
    total_cut = sum(s["length_mm"] for s in seg_results)
    total_rem = sum(r["area_mm2"] for r in rem_results)
    logger.info(
        "Cut analysis: %d segments, %d removed components, total cut %.2f mm, total removed %.2f mm^2",
        len(seg_results),
        len(rem_results),
        total_cut,
        total_rem,
    )
    return {
        "segments": seg_results,
        "removals": rem_results,
        "total_cut_length_mm": round(float(total_cut), 2),
        "total_removed_area_mm2": round(float(total_rem), 2),
        "cut_edges": [list(e) for e in cut_edges] if cut_edges else [],
    }


def analyze_cut_boundary(
    original_vertices: np.ndarray,
    original_triangles: np.ndarray,
    kept_vertices: np.ndarray,
    kept_triangles: np.ndarray,
    min_seg_length: float = 3.0,
    min_area: float = 150.0,
    min_al_ratio: float = 5.0,
) -> dict:
    """Analyze cut boundary: find segments, removed components, classify validity."""
    cut_edges = _find_cut_boundary_edges(original_vertices, original_triangles, kept_vertices, kept_triangles)
    if not cut_edges:
        return _empty_cut_analysis()

    segments = _group_edges_into_segments(cut_edges)
    filtered_segments, filtered_cut_edges = _filter_segments_by_length(
        segments, cut_edges, original_vertices, min_seg_length
    )
    if not filtered_segments:
        return _empty_cut_analysis()

    # ── Find ALL removed triangles and build their full components ──
    tree = KDTree(original_vertices)
    _, nearest = tree.query(kept_vertices)
    kept_v_set = set(nearest.tolist())

    all_removed_tris: list[int] = []
    for ti in range(len(original_triangles)):
        if any(int(original_triangles[ti, j]) not in kept_v_set for j in range(3)):
            all_removed_tris.append(ti)

    if not all_removed_tris:
        return _empty_cut_analysis()

    # Full connected components of ALL removed tris
    per_triangle_areas = compute_triangle_areas(original_vertices, original_triangles)
    edge_to_tris = build_edge_to_triangles(original_triangles)
    full_removed_components = _build_removed_components(all_removed_tris, edge_to_tris)

    # Filter to only components that touch cut edges
    filtered_cut_edges_set = set(filtered_cut_edges)
    removed_components: list[list[int]] = []
    for comp in full_removed_components:
        comp_verts: set[int] = set()
        for ti in comp:
            for j in range(3):
                comp_verts.add(int(original_triangles[ti, j]))
        # Does this component touch any filtered cut edge?
        touches = any(a in comp_verts or b in comp_verts for a, b in filtered_cut_edges_set)
        if touches:
            removed_components.append(comp)

    if not removed_components:
        return _empty_cut_analysis()

    seg_results, rem_results = _classify_segments_by_validity(
        filtered_segments,
        removed_components,
        original_vertices,
        original_triangles,
        per_triangle_areas,
        filtered_cut_edges,
        min_area=min_area,
        min_al_ratio=min_al_ratio,
    )
    return _finalize_cut_analysis(seg_results, rem_results, filtered_cut_edges)


def _find_invalid_removal_tris(
    original_vertices: np.ndarray,
    original_triangles: np.ndarray,
    kept_vertices: np.ndarray,
    kept_triangles: np.ndarray,
    min_area: float,
    min_al_ratio: float,
) -> tuple[set[int], set[int]]:
    """Find triangle indices of invalid removals and kept-vertex original indices.

    An invalid removal is a component whose area is too small or whose
    area / border-length ratio is too low (thin strip).  These triangles
    will be restored back into the kept mesh.

    Returns (invalid_tris, kept_v_orig).
    """
    an = analyze_cut_boundary(
        original_vertices,
        original_triangles,
        kept_vertices,
        kept_triangles,
        min_seg_length=0.0,
        min_area=min_area,
        min_al_ratio=min_al_ratio,
    )

    invalid_tris: set[int] = set()
    for r in an["removals"]:
        if not r["valid"]:
            invalid_tris.update(r["triangle_indices"])

    tree = KDTree(original_vertices)
    _, nearest = tree.query(kept_vertices)
    kept_v_orig = set(nearest.tolist())
    return invalid_tris, kept_v_orig


def restore_invalid_cuts(
    original_vertices: np.ndarray,
    original_triangles: np.ndarray,
    kept_vertices: np.ndarray,
    kept_triangles: np.ndarray,
    min_area: float = 150.0,
    min_al_ratio: float = 5.0,
    dilate_back: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Restore triangles cut by invalid (low area/length) segments.

    Directly restores the exact triangles of invalid removed components
    (those whose area is too small or area/border ratio is too low).
    No multi-layer expansion — that would leak into valid components.
    """
    invalid_tris, kept_v_orig = _find_invalid_removal_tris(
        original_vertices,
        original_triangles,
        kept_vertices,
        kept_triangles,
        min_area,
        min_al_ratio,
    )
    if not invalid_tris:
        logger.info("No invalid removals to restore")
        return kept_vertices.copy(), kept_triangles.copy()

    current_kept: set[int] = set(kept_v_orig)
    total_restored = len(invalid_tris)

    # Add all triangles of invalid removals to the kept set
    for ti in invalid_tris:
        for j in range(3):
            current_kept.add(int(original_triangles[ti, j]))

    if total_restored == 0:
        logger.info("No triangles restored from invalid segments")
        return kept_vertices.copy(), kept_triangles.copy()

    # Build result mesh from expanded kept set
    final_tris: list[int] = []
    for ti in range(len(original_triangles)):
        tri_v = [int(original_triangles[ti, j]) for j in range(3)]
        if all(v in current_kept for v in tri_v):
            final_tris.append(ti)

    sorted_v = sorted(current_kept)
    v_map = {ov: nv for nv, ov in enumerate(sorted_v)}
    new_tris_list = [[v_map[int(original_triangles[ti, j])] for j in range(3)] for ti in final_tris]

    result = (
        original_vertices[sorted_v].copy(),
        np.array(new_tris_list, dtype=np.int32),
    )
    logger.info(
        "Restored %d triangles — %dv -> %dv",
        total_restored,
        len(kept_v_orig),
        len(sorted_v),
    )
    return result
