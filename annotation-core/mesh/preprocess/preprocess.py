"""Public API for back scan mesh preprocessing (2.5D envelope)."""

import open3d as o3d

from mesh.preprocess.envelope import rebuild_single_layer_envelope
from utils.mesh import clone_mesh


def preprocess_back_scan_mesh(
    mesh: o3d.geometry.TriangleMesh,
    envelope_grid_resolution_mm: float = 1.5,
    envelope_smooth_sigma_px: float = 1.25,
    envelope_valid_weight_threshold: float = 0.16,
    envelope_max_triangle_z_span_mm: float = 35.0,
    envelope_max_smooth_delta_mm: float = 6.0,
) -> tuple[o3d.geometry.TriangleMesh, list[dict[str, object]]]:
    """Run the preprocess-only pipeline on a back scan mesh."""
    if mesh.is_empty() or not mesh.has_triangles():
        return clone_mesh(mesh), []

    smoothed_envelope_mesh, smooth_stats, z_grid, valid_mask, xy_min = (
        rebuild_single_layer_envelope(
            mesh,
            grid_resolution_mm=envelope_grid_resolution_mm,
            smooth_sigma_px=envelope_smooth_sigma_px,
            valid_weight_threshold=envelope_valid_weight_threshold,
            max_triangle_z_span_mm=envelope_max_triangle_z_span_mm,
            max_smooth_delta_mm=envelope_max_smooth_delta_mm,
        )
    )

    grid_origin = {
        "xy_min": [float(xy_min[0]), float(xy_min[1])],
        "grid_resolution_mm": float(envelope_grid_resolution_mm),
        "nx": int(z_grid.shape[1]),
        "ny": int(z_grid.shape[0]),
    }

    debug_steps: list[dict[str, object]] = [
        {
            "name": "02_envelope_smoothing",
            "description": (
                "高度场平滑：在规则 envelope 上做受限 Z 平滑，"
                f"mean|dz|={float(smooth_stats.get('mean_abs_delta_mm', 0.0)):.3f} mm，"
                f"p90|dz|={float(smooth_stats.get('p90_abs_delta_mm', 0.0)):.3f} mm，"
                f"clip=+/-{float(smooth_stats.get('max_smooth_delta_mm', 0.0)):.1f} mm"
            ),
            "stats": {**smooth_stats, "grid_origin": grid_origin},
            "mesh": clone_mesh(smoothed_envelope_mesh),
        },
    ]
    return smoothed_envelope_mesh, debug_steps
