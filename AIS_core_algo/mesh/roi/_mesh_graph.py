"""Mesh graph utilities: edge indexes, adjacency, connected components.

All functions accept np.ndarray inputs (not open3d objects) and are
designed to be the single canonical implementation across the project.
"""

import numpy as np

from utils.logger import logger

_MIN_SHARED_TRIS = 2  # 至少 2 个三角面共享边才构成邻接关系


def build_edge_to_triangles(triangles: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Build undirected edge -> triangle index map.

    Edge (a, b) is stored with a < b for deterministic keys.
    Complexity O(N) where N = number of triangles.

    Notes
    -----
    Hand‑unrolled loop (×3) beats ``for ei in range(3)`` by avoiding
    iterator overhead on ~5M edge iterations.  ``int()`` on array values
    is skipped because numpy int64 works directly as dict keys.
    """
    edge_to_tris: dict[tuple[int, int], list[int]] = {}
    for ti in range(len(triangles)):
        a, b, c = triangles[ti]
        # edge 0→1
        if a < b:
            edge_to_tris.setdefault((a, b), []).append(ti)
        else:
            edge_to_tris.setdefault((b, a), []).append(ti)
        # edge 1→2
        if b < c:
            edge_to_tris.setdefault((b, c), []).append(ti)
        else:
            edge_to_tris.setdefault((c, b), []).append(ti)
        # edge 2→0
        if c < a:
            edge_to_tris.setdefault((c, a), []).append(ti)
        else:
            edge_to_tris.setdefault((a, c), []).append(ti)
    logger.info(
        "Built edge-to-tri index: %d edges from %d tris",
        len(edge_to_tris),
        len(triangles),
    )
    return edge_to_tris


def build_triangle_adjacency(
    triangles: np.ndarray,
    edge_to_tris: dict[tuple[int, int], list[int]] | None = None,
) -> list[set[int]]:
    """Build triangle adjacency list: adj[ti] = {neighbor triangle indices}.

    Two triangles are adjacent if they share an edge.
    Uses sets for O(1) membership checks.

    Parameters
    ----------
    edge_to_tris:
        Pre‑built edge→triangle map (from ``build_edge_to_triangles``).
        When provided, avoids rebuilding this expensive index.
    """
    if edge_to_tris is None:
        edge_to_tris = build_edge_to_triangles(triangles)
    adj: list[set[int]] = [set() for _ in range(len(triangles))]
    for tris in edge_to_tris.values():
        if len(tris) < _MIN_SHARED_TRIS:
            continue
        for ti in tris:
            for tj in tris:
                if tj != ti:
                    adj[ti].add(tj)
    logger.info("Built triangle adjacency: %d triangles", len(triangles))
    return adj


def build_vertex_to_triangles(
    triangles: np.ndarray,
    vertex_count: int,
) -> list[list[int]]:
    """Build vertex -> triangle index: result[vid] = [triangle indices].

    Pre-built lookup replacing O(N^2) brute-force scanning.
    """
    vt: list[list[int]] = [[] for _ in range(vertex_count)]
    for ti in range(len(triangles)):
        for vid in triangles[ti]:
            vt[int(vid)].append(ti)
    return vt


def compute_boundary_edges(triangles: np.ndarray) -> set[tuple[int, int]]:
    """Compute boundary edges: edges shared by exactly one triangle.

    Returns a set of (a, b) tuples with a < b.
    """
    edge_to_tris = build_edge_to_triangles(triangles)
    boundary: set[tuple[int, int]] = {edge for edge, tris in edge_to_tris.items() if len(tris) == 1}
    logger.info("Found %d boundary edges from %d tris", len(boundary), len(triangles))
    return boundary


def compute_triangle_areas(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Compute per-triangle area (mm^2), shape = (N,).

    Based on cross-product: area = 0.5 * |cross(v1 - v0, v2 - v0)|.
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.sqrt(np.sum(cross**2, axis=1))
    return areas


def find_connected_components(
    adjacency: list[set[int]],
) -> list[list[int]]:
    """Find connected components via DFS, sorted by size descending.

    Outperforms networkx for graphs with < 100k nodes.
    adjacency: adj[node_index] = set of neighbor node indices.
    """
    n = len(adjacency)
    visited = [False] * n
    components: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp: list[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nb in adjacency[node]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        components.append(comp)

    components.sort(key=len, reverse=True)
    logger.info("Found %d connected components (largest=%d)", len(components), len(components[0]) if components else 0)
    return components
