"""Debug payload builder for shoulder transition detection."""

import numpy as np


def _empty_debug() -> dict:
    """Build empty debug dict for shoulder transition when no valid detection."""
    return {
        "contour": np.empty((0, 2)),
        "long_axis_angles": np.empty(0),
        "candidate_mask": np.empty(0, dtype=bool),
        "box_mask": np.empty(0, dtype=bool),
        "peak_index": 0,
        "peak_point": np.zeros(2),
        "peak_angle_deg": 180.0,
        "has_arm": True,
        "fallback": True,
        "outer_x": 0.0,
        "inner_x": 0.0,
        "axY": 0.0,
        "n_candidates": 0,
    }
