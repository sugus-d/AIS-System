"""BFS region growing for back ROI extraction — public API.

Usage:

    import open3d as o3d
    from mesh.roi.bfs import largest_component, mesh_bfs

    mesh = o3d.io.read_triangle_mesh("scan.ply")
    trunk = mesh_bfs(mesh)               # BFS region growing
    trunk = largest_component(trunk)     # keep only the largest component
"""

from __future__ import annotations

from ._bfs_impl import compute_mesh_roughness, largest_component, mesh_bfs

__all__ = ["compute_mesh_roughness", "largest_component", "mesh_bfs"]
