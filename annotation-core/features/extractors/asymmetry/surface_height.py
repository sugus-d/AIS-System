import numpy as np
import open3d as o3d


def compute_surface_height(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Return per-vertex height as the raw Z coordinate on the original mesh.

    The height of each vertex *v* is simply its Z coordinate:

        z(v) = v_z

    Left-right asymmetry is computed later via UV-parameterisation matching
    (see :func:`features.extractors.asymmetry.landmark_regions.compute_region_features`), so no
    plane-fitting is needed here.

    Returns:
        (N,) — Z coordinate per vertex.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise ValueError("Mesh has no vertices.")

    heights = vertices[:, 2].copy()
    return heights
