"""Validity classification of cut segments and removed regions."""

import numpy as np

from .regions import _build_comp_to_segments, _build_comp_vertex_sets


def _compute_segment_results(
    segments: list[list[int]],
    cut_edges: set[tuple[int, int]],
    original_vertices: np.ndarray,
) -> list[dict]:
    """Build per-segment result dicts (length, edges, topology only).

    ``valid`` is NOT computed here — it is set later from the removals
    each segment borders.  Area is informational (directly adjacent
    removed triangles); the true restore decision lives on removals.
    """
    seg_results: list[dict] = []
    for seg_vids in segments:
        seg_set = set(seg_vids)
        seg_edges = [e for e in cut_edges if e[0] in seg_set and e[1] in seg_set]
        seg_len = 0.0
        for a, b in seg_edges:
            seg_len += float(np.linalg.norm(original_vertices[a] - original_vertices[b]))

        seg_results.append(
            {
                "edge_count": int(len(seg_edges)),
                "length_mm": round(float(seg_len), 2),
                "vertex_ids": seg_vids,
                "valid": True,  # placeholder, overridden below
            }
        )
    return seg_results


def _compute_removed_results(
    removed_components: list[list[int]],
    comp_to_segs: list[set[int]],
    original_vertices: np.ndarray,
    original_triangles: np.ndarray,
    cut_edges: set[tuple[int, int]],
    per_triangle_areas: np.ndarray,
    min_area: float,
    min_al_ratio: float,
) -> list[dict]:
    """Build per-removed-component result dicts (region-centric).

    Each removal is a connected component of triangles not in the kept
    mesh.  The restore decision is based on the component's *own*
    properties:

    * ``area_mm2 < min_area`` → too small, restore
    * ``area_mm2 / border_length < min_al_ratio`` → thin strip, restore
    * Otherwise → valid cut, keep removed.
    """
    rem_results: list[dict] = []
    for ci, comp in enumerate(removed_components):
        comp_area = sum(per_triangle_areas[ti] for ti in comp)
        involved_segs = sorted(comp_to_segs[ci])

        # Build vertex set of this component
        comp_verts: set[int] = set()
        for ti in comp:
            for j in range(3):
                comp_verts.add(int(original_triangles[ti, j]))

        # Bordering cut edges: cut_edges that have at least one endpoint
        # in this component.
        border_edges = [(a, b) for (a, b) in cut_edges if a in comp_verts or b in comp_verts]
        border_len = sum(float(np.linalg.norm(original_vertices[a] - original_vertices[b])) for a, b in border_edges)

        # Restore decision (region-centric)
        too_small = comp_area < min_area
        too_thin = comp_area < min_al_ratio * max(border_len, 1.0)
        valid = not (too_small or too_thin)

        rem_results.append(
            {
                "area_mm2": round(float(comp_area), 2),
                "tri_count": int(len(comp)),
                "triangle_indices": list(comp),
                "segment_indices": involved_segs,
                "border_length_mm": round(float(border_len), 2),
                "border_edge_count": int(len(border_edges)),
                "valid": bool(valid),
            }
        )
    return rem_results


def _classify_segments_by_validity(
    segments: list[list[int]],
    removed_components: list[list[int]],
    original_vertices: np.ndarray,
    original_triangles: np.ndarray,
    per_triangle_areas: np.ndarray,
    cut_edges: set[tuple[int, int]],
    min_area: float = 10.0,
    min_al_ratio: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """Classify removed components (region-centric) and derive segment validity.

    *Removals* are the primary decision unit: each removal's validity is
    based on its own area and border-edge length.  *Segments* derive
    their ``valid`` from the removals they border.

    Returns
    -------
    seg_results : list[dict]
    rem_results : list[dict]
    """
    comp_vertex_sets = _build_comp_vertex_sets(removed_components, original_triangles)
    comp_to_segs = _build_comp_to_segments(segments, comp_vertex_sets)

    # 1. Compute removals first (region-centric logic)
    rem_results = _compute_removed_results(
        removed_components,
        comp_to_segs,
        original_vertices,
        original_triangles,
        cut_edges,
        per_triangle_areas,
        min_area,
        min_al_ratio,
    )

    # 2. Compute segments (topology only — no valid yet)
    seg_results = _compute_segment_results(
        segments,
        cut_edges,
        original_vertices,
    )

    # 3. Derive segment validity from removals
    #    A segment is valid if any bordering removal is valid.
    for si, seg in enumerate(seg_results):
        seg["valid"] = any(rem_results[ci]["valid"] for ci, segs in enumerate(comp_to_segs) if si in segs)

    return seg_results, rem_results
