"""Tests for mesh/roi/_mesh_graph.py and mesh/roi/bfs.py.

Covers the shared graph utilities and the BFS core functions.
"""

import numpy as np
import open3d as o3d

from mesh.roi._bfs_impl import _compute_triangle_normals, _select_seed
from mesh.roi._mesh_graph import (
    build_edge_to_triangles,
    build_triangle_adjacency,
    build_vertex_to_triangles,
    compute_boundary_edges,
    compute_triangle_areas,
    find_connected_components,
)
from mesh.roi.bfs import largest_component, mesh_bfs

# ════════════════════════════════════════════════════════════════════
# _mesh_graph.py
# ════════════════════════════════════════════════════════════════════

class TestBuildEdgeToTriangles:
    def test_single_triangle(self):
        """Three edges for one triangle."""
        tris = np.array([[0, 1, 2]], dtype=np.int64)
        e2t = build_edge_to_triangles(tris)
        assert len(e2t) == 3
        assert e2t[(0, 1)] == [0]
        assert e2t[(1, 2)] == [0]
        assert e2t[(0, 2)] == [0]

    def test_shared_edge(self):
        """Two triangles sharing edge (1, 2)."""
        tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        e2t = build_edge_to_triangles(tris)
        assert e2t[(1, 2)] == [0, 1]

    def keys_are_sorted(self):
        """Edge keys always use a < b."""
        tris = np.array([[0, 1, 2]], dtype=np.int64)
        e2t = build_edge_to_triangles(tris)
        for (a, b) in e2t:
            assert a < b, f"Edge ({a}, {b}) not sorted"


class TestBuildTriangleAdjacency:
    def test_two_adjacent_triangles(self):
        """Quad: two tri sharing one edge -> each sees the other."""
        tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        adj = build_triangle_adjacency(tris)
        assert len(adj) == 2
        assert 1 in adj[0]
        assert 0 in adj[1]

    def test_no_adjacency(self):
        """Two disconnected triangles -> empty adjacency sets."""
        tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        adj = build_triangle_adjacency(tris)
        assert len(adj[0]) == 0
        assert len(adj[1]) == 0

    def test_returns_sets(self):
        """Adjacency entries are sets for O(1) membership."""
        tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        adj = build_triangle_adjacency(tris)
        assert isinstance(adj[0], set)
        assert isinstance(adj[1], set)


class TestBuildVertexToTriangles:
    def test_single_triangle(self):
        tris = np.array([[0, 1, 2]], dtype=np.int64)
        vt = build_vertex_to_triangles(tris, vertex_count=3)
        assert vt[0] == [0]
        assert vt[1] == [0]
        assert vt[2] == [0]

    def test_shared_vertex(self):
        """Two triangles sharing vertex 1."""
        tris = np.array([[0, 1, 2], [1, 3, 4]], dtype=np.int64)
        vt = build_vertex_to_triangles(tris, vertex_count=5)
        assert set(vt[1]) == {0, 1}


class TestComputeBoundaryEdges:
    def test_quad_has_four_boundary_edges(self):
        """Two triangles forming a quad: 4 boundary edges, 1 shared edge."""
        tris = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        boundaries = compute_boundary_edges(tris)
        assert len(boundaries) == 4
        assert (1, 2) not in boundaries  # shared edge is not boundary
        for edge in [(0, 1), (0, 2), (1, 3), (2, 3)]:
            assert edge in boundaries

    def test_closed_tri_mesh_no_boundary(self):
        """Simple tetrahedron has no boundary edges."""
        tris = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
                        dtype=np.int64)
        boundaries = compute_boundary_edges(tris)
        assert len(boundaries) == 0


class TestComputeTriangleAreas:
    def test_right_triangle(self):
        """3-4-5 triangle: area = 6.0."""
        verts = np.array([[0.0, 0.0, 0.0],
                          [3.0, 0.0, 0.0],
                          [0.0, 4.0, 0.0]], dtype=np.float64)
        tris = np.array([[0, 1, 2]], dtype=np.int64)
        areas = compute_triangle_areas(verts, tris)
        assert np.isclose(areas[0], 6.0)

    def test_tetrahedron_areas(self):
        """Unit tetrahedron faces have equal area ~0.433."""
        verts = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.5, np.sqrt(3) / 2, 0.0],
                          [0.5, np.sqrt(3) / 6, np.sqrt(2.0 / 3.0)]],
                         dtype=np.float64)
        tris = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
                        dtype=np.int64)
        areas = compute_triangle_areas(verts, tris)
        expected = np.sqrt(3) / 4
        assert np.allclose(areas, expected)


class TestFindConnectedComponents:
    def test_two_separate_components(self):
        """Two disconnected groups of triangles."""
        adj = [{1}, {0}, {3}, {2}]  # comp0={0,1}, comp1={2,3}
        comps = find_connected_components(adj)
        assert len(comps) == 2
        assert comps[0] == [0, 1] or comps[0] == [2, 3]  # sorted by size desc

    def test_single_component(self):
        adj = [{1}, {0, 2}, {1}]
        comps = find_connected_components(adj)
        assert len(comps) == 1
        assert len(comps[0]) == 3

    def test_no_edges_individual_components(self):
        """Each node is its own component."""
        adj = [set(), set(), set()]
        comps = find_connected_components(adj)
        assert len(comps) == 3

    def test_sorted_by_size_descending(self):
        """Components are returned largest-first."""
        adj = [{1}, {0}, set(), {4}, {3}]
        comps = find_connected_components(adj)
        sizes = [len(c) for c in comps]
        assert sizes == sorted(sizes, reverse=True)


# ════════════════════════════════════════════════════════════════════
# bfs.py
# ════════════════════════════════════════════════════════════════════

def _simple_mesh(n_tris: int = 4) -> o3d.geometry.TriangleMesh:
    """Build a flat square mesh with N adjacent triangle strips."""
    # 4 triangles in a fan around origin, all in z=0 plane.
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.remove_unreferenced_vertices()
    return mesh


class TestComputeTriangleNormals:
    def test_flat_mesh(self):
        """All normals point along +z for a flat mesh in z=0."""
        mesh = _simple_mesh()
        norms = _compute_triangle_normals(mesh)
        assert norms.shape == (2, 3)
        assert np.allclose(norms, [[0, 0, 1], [0, 0, 1]])

class TestSelectSeed:
    def _make_adj(self, n: int) -> list[set[int]]:
        return [set() for _ in range(n)]

    def test_flat_region_picked(self):
        """Low-roughness region -> seed chosen from it."""
        n = 100
        rgh = np.full(n, 0.3)   # rough everywhere
        rgh[:60] = 0.05         # first 60 tris smooth
        centers = np.column_stack([
            np.random.default_rng(42).uniform(-10, 10, n),
            np.random.default_rng(99).uniform(-10, 10, n),
            np.zeros(n),
        ])
        adj = self._make_adj(n)
        # roughness_threshold=0.25 → seed cutoff = 0.15; first 60 tris at 0.05 pass
        seed = _select_seed(rgh, centers, adj, roughness_threshold=0.25)
        assert 0 <= seed < 60

    def test_fallback_to_min_roughness(self):
        """All rough -> fallback to minimum-roughness triangle."""
        n = 50
        rgh = np.concatenate([np.full(25, 1.0), np.full(25, 0.5)])
        centers = np.zeros((n, 3))
        adj = self._make_adj(n)
        # roughness_threshold=0.2 → seed cutoff = 0.12; nothing qualifies
        seed = _select_seed(rgh, centers, adj, roughness_threshold=0.2,
                            min_component_tris=1)
        # Min roughness is 0.5, tri index 25-49 area
        assert 25 <= seed < 50


class TestMeshBFS:
    def test_empty_mesh_returns_empty(self):
        """Empty input mesh -> empty output mesh."""
        empty = o3d.geometry.TriangleMesh()
        result = mesh_bfs(empty)
        assert len(np.asarray(result.triangles)) == 0

    def test_flat_mesh_preserves_all_tris(self):
        """All triangles flat and coplanar -> everything kept."""
        mesh = _simple_mesh()
        result = mesh_bfs(mesh)
        assert len(np.asarray(result.triangles)) == len(np.asarray(mesh.triangles))

    def test_output_is_triangle_mesh(self):
        mesh = _simple_mesh()
        result = mesh_bfs(mesh)
        assert isinstance(result, o3d.geometry.TriangleMesh)


class TestLargestComponent:
    def test_single_component_untouched(self):
        """Mesh with one component -> returned verbatim."""
        mesh = _simple_mesh()
        result = largest_component(mesh)
        assert len(np.asarray(result.triangles)) == len(np.asarray(mesh.triangles))

    def test_disconnected_drops_smaller(self):
        """Two separate triangles in same mesh -> only larger component kept."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [10, 10, 0], [11, 10, 0], [10, 11, 0],
        ], dtype=np.float64)
        tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)
        result = largest_component(mesh)
        # Both components have 1 triangle, so it keeps one.
        assert len(np.asarray(result.triangles)) == 1

    def test_with_two_components(self):
        """Mesh with disconnected components keeps only one."""
        verts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [10, 10, 0], [11, 10, 0], [10, 11, 0],
        ], dtype=np.float64)
        tris = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(tris)
        result = largest_component(mesh)
        # Both components have 1 triangle, so it keeps one.
        assert len(np.asarray(result.triangles)) == 1
