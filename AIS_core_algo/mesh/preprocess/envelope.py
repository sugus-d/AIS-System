"""2.5D envelope reconstruction and smoothing from a back scan mesh."""

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_fill_holes, gaussian_filter

from utils.logger import logger
from utils.mesh import clone_mesh


def _build_envelope_mesh(
    z_values: np.ndarray,
    valid_mask: np.ndarray,
    xy_min: np.ndarray,
    grid_resolution_mm: float,
    max_triangle_z_span_mm: float,
) -> o3d.geometry.TriangleMesh:
    """Build a triangulated mesh from a 2D Z-valued grid."""
    ny, nx = valid_mask.shape
    grid_indices = -np.ones((ny, nx), dtype=int)
    valid_y, valid_x = np.where(valid_mask)
    grid_indices[valid_y, valid_x] = np.arange(len(valid_x))

    out_vertices = np.column_stack((
        xy_min[0] + valid_x.astype(float) * grid_resolution_mm,
        xy_min[1] + valid_y.astype(float) * grid_resolution_mm,
        z_values[valid_y, valid_x],
    ))

    out_triangles: list[list[int]] = []
    for row in range(ny - 1):
        for col in range(nx - 1):
            a = grid_indices[row, col]
            b = grid_indices[row, col + 1]
            c = grid_indices[row + 1, col]
            d = grid_indices[row + 1, col + 1]
            if (
                a >= 0
                and b >= 0
                and c >= 0
                and float(np.max(out_vertices[[a, b, c], 2]) - np.min(out_vertices[[a, b, c], 2])) <= max_triangle_z_span_mm
            ):
                out_triangles.append([int(a), int(b), int(c)])
            if (
                b >= 0
                and c >= 0
                and d >= 0
                and float(np.max(out_vertices[[b, d, c], 2]) - np.min(out_vertices[[b, d, c], 2])) <= max_triangle_z_span_mm
            ):
                out_triangles.append([int(b), int(d), int(c)])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(out_vertices.astype(float))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(out_triangles, dtype=np.int32))
    mesh.compute_vertex_normals()
    return mesh


def rebuild_single_layer_envelope(
    mesh: o3d.geometry.TriangleMesh,
    grid_resolution_mm: float = 1.5,
    smooth_sigma_px: float = 1.25,
    valid_weight_threshold: float = 0.16,
    max_triangle_z_span_mm: float = 35.0,
    max_smooth_delta_mm: float = 6.0,
) -> tuple[o3d.geometry.TriangleMesh, dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the scan as a single-layer XY envelope and smooth it on a height field."""
    vertices = np.asarray(mesh.vertices)
    smooth_stats: dict[str, object] = {
        "grid_resolution_mm": float(grid_resolution_mm),
        "smooth_sigma_px": float(smooth_sigma_px),
        "valid_weight_threshold": float(valid_weight_threshold),
        "max_triangle_z_span_mm": float(max_triangle_z_span_mm),
        "max_smooth_delta_mm": float(max_smooth_delta_mm),
        "output_vertices": 0, "output_triangles": 0,
        "mean_abs_delta_mm": 0.0, "p90_abs_delta_mm": 0.0, "max_abs_delta_mm": 0.0,
    }
    if len(vertices) == 0:
        empty = clone_mesh(mesh)
        return empty, smooth_stats, np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=bool), np.array([0.0, 0.0])

    xy_min = vertices[:, :2].min(axis=0) - 1e-6
    xy_max = vertices[:, :2].max(axis=0) + 1e-6
    nx = int(np.floor((xy_max[0] - xy_min[0]) / grid_resolution_mm)) + 1
    ny = int(np.floor((xy_max[1] - xy_min[1]) / grid_resolution_mm)) + 1

    ix = np.clip(((vertices[:, 0] - xy_min[0]) / grid_resolution_mm).astype(int), 0, nx - 1)
    iy = np.clip(((vertices[:, 1] - xy_min[1]) / grid_resolution_mm).astype(int), 0, ny - 1)
    flat_idx = iy * nx + ix

    z_sum = np.zeros(ny * nx, dtype=float)
    z_count = np.zeros(ny * nx, dtype=float)
    np.add.at(z_sum, flat_idx, vertices[:, 2])
    np.add.at(z_count, flat_idx, 1.0)
    z_grid = np.zeros(ny * nx, dtype=float)
    np.divide(z_sum, z_count, out=z_grid, where=z_count > 0)
    z_grid = np.where(z_count > 0, z_grid, -np.inf)
    z_grid = z_grid.reshape(ny, nx)
    observed_mask = (z_count.reshape(ny, nx) > 0)

    filled_grid = np.where(observed_mask, z_grid, 0.0)
    support_weight = gaussian_filter(observed_mask.astype(float), sigma=smooth_sigma_px)
    smoothed_raw = gaussian_filter(filled_grid, sigma=smooth_sigma_px)
    smoothed_raw = np.divide(smoothed_raw, np.maximum(support_weight, 1e-6))
    valid_mask = support_weight > valid_weight_threshold
    valid_mask = binary_fill_holes(valid_mask)
    raw_grid = np.where(observed_mask, z_grid, smoothed_raw)
    smoothed_grid = raw_grid + np.clip(
        smoothed_raw - raw_grid, -max_smooth_delta_mm, max_smooth_delta_mm
    )

    smoothed_mesh = _build_envelope_mesh(
        smoothed_grid, valid_mask, xy_min, grid_resolution_mm, max_triangle_z_span_mm
    )

    delta_abs = (
        np.abs(smoothed_grid[observed_mask] - z_grid[observed_mask])
        if observed_mask.any() else np.empty(0)
    )
    smooth_stats.update({
        "output_vertices": int(len(np.asarray(smoothed_mesh.vertices))),
        "output_triangles": int(len(np.asarray(smoothed_mesh.triangles))),
        "mean_abs_delta_mm": float(np.mean(delta_abs)) if len(delta_abs) > 0 else 0.0,
        "p90_abs_delta_mm": float(np.percentile(delta_abs, 90)) if len(delta_abs) > 0 else 0.0,
        "max_abs_delta_mm": float(np.max(delta_abs)) if len(delta_abs) > 0 else 0.0,
    })

    logger.info(
        "Preprocess single-layer envelope: grid={:.2f}mm, valid_cells={}, output={}/{} verts/tris".format(
            float(grid_resolution_mm), int(np.count_nonzero(valid_mask)),
            int(smooth_stats["output_vertices"]), int(smooth_stats["output_triangles"]),
        )
    )

    return smoothed_mesh, smooth_stats, z_grid, valid_mask, xy_min
