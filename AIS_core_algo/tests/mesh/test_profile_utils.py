import numpy as np

from landmarks.lateral_profile import (
    compute_width_profile as compute_lateral_width_profile,
)
from landmarks.profile import build_width_profile_lines


def test_shared_width_profile_helpers_agree_on_simple_contours() -> None:
    y = np.linspace(0.0, 9.0, 10)
    left = np.column_stack([np.zeros_like(y), y])
    right = np.column_stack([np.full_like(y, 10.0), y])

    widths, y_centers = compute_lateral_width_profile(left, right, n_bins=6)
    shared_lines = build_width_profile_lines(left, right, n_bins=6)

    assert np.allclose(widths, 10.0)
    assert np.allclose(shared_lines[:, 3], 10.0)
    assert np.allclose(shared_lines[:, 2], y_centers)
