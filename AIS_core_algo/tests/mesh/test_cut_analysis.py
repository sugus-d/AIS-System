"""Tests for mesh/roi/_cut_analysis.py.

Covers cut boundary detection, segment classification, and invalid-cut
restoration using synthetic mesh geometries.
"""

from __future__ import annotations

import numpy as np
import pytest

from mesh.roi._cut_analysis import (
    analyze_cut_boundary,
    compute_removed_triangles,
    restore_invalid_cuts,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic meshes
# ---------------------------------------------------------------------------


@pytest.fixture
def quad_mesh() -> tuple[np.ndarray, np.ndarray]:
    """2-triangle quad (4 verts, 2 tris) on the z=0 plane."""
    verts = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], dtype=float)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return verts, tris


@pytest.fixture
def strip_mesh() -> tuple[np.ndarray, np.ndarray]:
    r"""Triangle strip with 4 triangles, 6 vertices.

    Vertices:
      0---2---4
      |\  |\  |
      | \ | \ |
      1---3---5
    """
    verts = np.array(
        [
            [0, 0, 0],
            [0, -1, 0],
            [2, 0, 0],
            [2, -1, 0],
            [4, 0, 0],
            [4, -1, 0],
        ],
        dtype=float,
    )
    tris = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [2, 3, 4],
            [3, 5, 4],
        ],
        dtype=np.int32,
    )
    return verts, tris


@pytest.fixture
def cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    """A simple cube (8 verts, 12 tris) for more realistic tests."""
    v = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],  # bottom face
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],  # top face
        ],
        dtype=float,
    )
    t = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 5, 6],
            [4, 6, 7],  # top
            [0, 1, 5],
            [0, 5, 4],  # front
            [2, 3, 7],
            [2, 7, 6],  # back
            [1, 2, 6],
            [1, 6, 5],  # right
            [3, 0, 4],
            [3, 4, 7],  # left
        ],
        dtype=np.int32,
    )
    return v, t


# ---------------------------------------------------------------------------
# compute_removed_triangles
# ---------------------------------------------------------------------------


class TestComputeRemovedTriangles:
    def test_no_removal(self, quad_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        verts, tris = quad_mesh
        removed = compute_removed_triangles(verts, tris, verts)
        assert removed == []

    def test_one_triangle_removed(self, quad_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        verts, tris = quad_mesh
        # Keep only first triangle's vertices
        kept_v = verts[:3]
        removed = compute_removed_triangles(verts, tris, kept_v)
        assert removed == [1]

    def test_all_removed(self, quad_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        verts, tris = quad_mesh
        # Keep only one vertex
        kept_v = verts[:1]
        removed = compute_removed_triangles(verts, tris, kept_v)
        assert removed == [0, 1]

    def test_partial_kept_vertex(self, strip_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        verts, tris = strip_mesh
        # Keep vertices for first two triangles only
        kept_v = verts[:4]
        removed = compute_removed_triangles(verts, tris, kept_v)
        assert removed == [2, 3]


# ---------------------------------------------------------------------------
# analyze_cut_boundary
# ---------------------------------------------------------------------------


class TestAnalyzeCutBoundary:
    def test_no_cut_returns_empty(self, quad_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Keep full mesh -> no cut boundary -> empty result."""
        verts, tris = quad_mesh
        result = analyze_cut_boundary(
            verts,
            tris,
            verts,
            tris,
            min_seg_length=0.0,
            min_area=0.0,
            min_al_ratio=0.0,
        )
        assert result["segments"] == []
        assert result["removals"] == []
        assert result["total_cut_length_mm"] == 0.0

    def test_cut_detects_segment(self, quad_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Remove one triangle -> single cut segment detected."""
        verts, tris = quad_mesh
        kept_v = verts[:3]
        kept_t = tris[:1]

        result = analyze_cut_boundary(
            verts,
            tris,
            kept_v,
            kept_t,
            min_seg_length=0.0,
            min_area=0.0,
            min_al_ratio=0.0,
        )

        assert len(result["segments"]) == 1
        seg = result["segments"][0]
        assert seg["edge_count"] >= 1
        assert seg["length_mm"] > 0
        assert result["total_removed_area_mm2"] > 0

    def test_cut_filters_by_min_seg_length(self, quad_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Short cut below min_seg_length should be filtered."""
        verts, tris = quad_mesh
        kept_v = verts[:3]
        kept_t = tris[:1]

        result = analyze_cut_boundary(
            verts,
            tris,
            kept_v,
            kept_t,
            min_seg_length=100.0,
            min_area=0.0,
            min_al_ratio=0.0,
        )

        assert result["segments"] == []

    def test_invalid_segment_min_area(self, quad_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Segment with area below min_area should be not valid."""
        verts, tris = quad_mesh
        kept_v = verts[:3]
        kept_t = tris[:1]

        result = analyze_cut_boundary(
            verts,
            tris,
            kept_v,
            kept_t,
            min_seg_length=0.0,
            min_area=100.0,
            min_al_ratio=0.0,
        )

        assert len(result["segments"]) == 1
        assert not result["segments"][0]["valid"]

    def test_invalid_segment_min_al_ratio(self, quad_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Segment with al_ratio below min_al_ratio should be not valid."""
        verts, tris = quad_mesh
        kept_v = verts[:3]
        kept_t = tris[:1]

        result = analyze_cut_boundary(
            verts,
            tris,
            kept_v,
            kept_t,
            min_seg_length=0.0,
            min_area=0.0,
            min_al_ratio=100.0,
        )

        assert len(result["segments"]) == 1
        assert not result["segments"][0]["valid"]

    def test_multiple_cut_segments(self, strip_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Removing a component with unique (unshared) vertices creates cut segment."""
        verts, tris = strip_mesh
        # Keep only the first triangle of the strip; T1,T2,T3 are "removed"
        # T1 uses v3 which is NOT in kept set {0,1,2} → detected as removed
        kept_v = verts[:3]
        kept_t = tris[:1]  # T0(0,1,2) only

        result = analyze_cut_boundary(
            verts,
            tris,
            kept_v,
            kept_t,
            min_seg_length=0.0,
            min_area=0.0,
            min_al_ratio=0.0,
        )

        # The cut produces at least 1 segment and 1 removed component
        assert len(result["segments"]) >= 1
        assert len(result["removals"]) >= 1

    def test_cube_cut_no_removal(self, cube_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Full cube cut should have no cut boundaries."""
        v, t = cube_mesh
        result = analyze_cut_boundary(
            v,
            t,
            v,
            t,
            min_seg_length=0.0,
            min_area=0.0,
            min_al_ratio=0.0,
        )
        assert result["segments"] == []

    def test_cube_remove_one_tri(self, cube_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Remove a single triangle, verify cut detection isn't empty."""
        v, t = cube_mesh
        kept_v = v[:3]
        kept_t = t[:1]

        result = analyze_cut_boundary(
            v,
            t,
            kept_v,
            kept_t,
            min_seg_length=0.0,
            min_area=0.0,
            min_al_ratio=0.0,
        )
        # Should have at least one cut segment
        assert len(result["segments"]) >= 1


# ---------------------------------------------------------------------------
# restore_invalid_cuts
# ---------------------------------------------------------------------------


class TestRestoreInvalidCuts:
    def test_no_restore_needed(self, quad_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """No invalid cuts -> same mesh returned."""
        verts, tris = quad_mesh
        restored_v, restored_t = restore_invalid_cuts(
            verts,
            tris,
            verts,
            tris,
            min_area=0.0,
            min_al_ratio=0.0,
        )
        np.testing.assert_array_equal(restored_v, verts)
        np.testing.assert_array_equal(restored_t, tris)

    def test_restores_invalid_cut(self, quad_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Invalid (low-area) cut should restore the removed triangle."""
        verts, tris = quad_mesh
        kept_v = verts[:3]
        kept_t = tris[:1]

        # With min_area=0.6, the cut in quad (area=0.5) is below threshold
        # but the segment is still valid because the *original* mesh's
        # removed component is smaller than min_area — restoration is triggered
        # by invalid segments from analyze_cut_boundary.
        # quad mesh triangle has area 1.0 (verts 0,1,2 -> actually calculation
        # yields area = 0.5 * |cross| for each triangle)
        # The quad area is 1.0 per triangle? Let me check:
        # v0=(0,0,0), v1=(2,0,0), v2=(2,1,0) -> cross = (0,0,2) -> area=1.0
        # So min_area=1.1 would make it invalid
        restored_v, restored_t = restore_invalid_cuts(
            verts,
            tris,
            kept_v,
            kept_t,
            min_area=1.1,
            min_al_ratio=0.0,
        )

        # Should have restored the original quad (4 verts, 2 tris)
        assert len(restored_v) == 4
        assert len(restored_t) == 2

    def test_valid_cut_not_restored(self, quad_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Valid (large-area) cut should keep the cut mesh unchanged."""
        verts, tris = quad_mesh
        kept_v = verts[:3]
        kept_t = tris[:1]

        restored_v, restored_t = restore_invalid_cuts(
            verts,
            tris,
            kept_v,
            kept_t,
            min_area=0.0,
            min_al_ratio=0.0,
        )
        # With min_area=0.0, the segment is valid -> no restoration
        assert len(restored_v) == 3
        assert len(restored_t) == 1

    def test_restore_missing_no_invalid(self, strip_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """When no invalid vids found, kept mesh is returned unchanged."""
        verts, tris = strip_mesh
        kept_v = verts[:4]
        kept_t = tris[:2]

        restored_v, restored_t = restore_invalid_cuts(
            verts,
            tris,
            kept_v,
            kept_t,
            min_area=0.0,
            min_al_ratio=0.0,
        )
        np.testing.assert_array_equal(restored_v, kept_v)
        np.testing.assert_array_equal(restored_t, kept_t)

    def test_cube_restore(self, cube_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Restore on a cube mesh doesn't crash."""
        v, t = cube_mesh
        kept_v = v[:3]
        kept_t = t[:1]

        restored_v, restored_t = restore_invalid_cuts(
            v,
            t,
            kept_v,
            kept_t,
            min_area=0.0,
            min_al_ratio=0.0,
        )
        # Should at least produce valid mesh (no invalid check needed)
        assert len(restored_v) >= 3
        assert len(restored_t) >= 1


# ---------------------------------------------------------------------------
# Integration: empty / degenerate inputs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_identical_kept_and_original(self, quad_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Edge case: kept == original yields no cut."""
        verts, tris = quad_mesh
        result = analyze_cut_boundary(verts, tris, verts, tris)
        assert result["segments"] == []

    def test_single_triangle_original(self) -> None:
        """Single triangle original mesh: no cut boundary possible."""
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        t = np.array([[0, 1, 2]], dtype=np.int32)
        result = analyze_cut_boundary(v, t, v, t)
        assert result["segments"] == []

    def test_all_vertices_removed(self, quad_mesh: tuple[np.ndarray, np.ndarray]) -> None:
        """Edge case: kept is empty should not crash (no kept triangles)."""
        verts, tris = quad_mesh
        kept_v = np.empty((0, 3), dtype=float)
        kept_t = np.empty((0, 3), dtype=np.int32)
        # Should not raise
        compute_removed_triangles(verts, tris, kept_v)
        # analyze removes all triangles -> no cut boundary in empty kept mesh
        result = analyze_cut_boundary(verts, tris, kept_v, kept_t)
        assert result["segments"] == []

    def test_restore_with_all_invalid(self) -> None:
        """Multiple adjacent removed triangles trigger proper restoration.

        Uses a 3-quad strip (8 verts, 6 tris). Kept is the first 2 tris
        (1 quad) with high min_al_ratio making the cut boundary invalid,
        triggering restoration of the adjacent removed triangles.
        """
        v = np.array(
            [
                [0, 0, 0],
                [0, -1, 0],
                [2, 0, 0],
                [2, -1, 0],
                [4, 0, 0],
                [4, -1, 0],
                [6, 0, 0],
                [6, -1, 0],
            ],
            dtype=float,
        )
        t = np.array(
            [
                [0, 1, 2],
                [1, 3, 2],
                [2, 3, 4],
                [3, 5, 4],
                [4, 5, 6],
                [5, 7, 6],
            ],
            dtype=np.int32,
        )
        # Keep first 2 triangles (1 quad), 4 vertices => prefix slice
        kept_v = v[:4]
        kept_t = t[:2]

        restored_v, restored_t = restore_invalid_cuts(
            v,
            t,
            kept_v,
            kept_t,
            min_area=0.0,
            min_al_ratio=100.0,
        )
        # Invalid due to al_ratio -> should restore some triangles
        assert len(restored_t) > len(kept_t)
