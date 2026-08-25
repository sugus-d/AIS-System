import numpy as np

from landmarks.contour import (
    extract_longest_contiguous_segment_in_box,
    resample_polyline_uniform,
)
from landmarks.signal_ops import compute_derivatives_from_xy


def test_extract_longest_contiguous_segment_in_box_open_polyline() -> None:
    contour = np.array([[float(x), 0.0] for x in range(10)], dtype=float)

    indices = extract_longest_contiguous_segment_in_box(
        contour,
        x_min=3.0,
        x_max=6.0,
        y_min=-1.0,
        y_max=1.0,
        closed=False,
    )

    assert np.array_equal(indices, np.array([3, 4, 5, 6], dtype=int))


def test_extract_longest_contiguous_segment_in_box_closed_contour() -> None:
    contour = np.array(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    indices = extract_longest_contiguous_segment_in_box(
        contour,
        x_min=3.5,
        x_max=4.5,
        y_min=-0.5,
        y_max=4.5,
        closed=True,
    )

    assert np.array_equal(indices, np.array([1, 2], dtype=int))


def test_extract_longest_contiguous_segment_in_box_no_intersection() -> None:
    contour = np.array([[float(x), 0.0] for x in range(5)], dtype=float)

    indices = extract_longest_contiguous_segment_in_box(
        contour,
        x_min=10.0,
        x_max=12.0,
        y_min=10.0,
        y_max=12.0,
        closed=False,
    )

    assert indices.size == 0


def test_resample_polyline_uniform_linear_spacing() -> None:
    polyline = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=float)

    resampled = resample_polyline_uniform(polyline, step=1.0)

    assert np.allclose(resampled[0], [0.0, 0.0])
    assert np.allclose(resampled[-1], [3.0, 4.0])
    assert len(resampled) == 6


def test_compute_derivatives_from_xy_linear_function() -> None:
    x_unique = np.arange(5, dtype=float)
    y_unique = 2.0 * x_unique + 1.0

    dydx = compute_derivatives_from_xy(x_unique, y_unique, derv_order=1)

    assert dydx.shape == x_unique.shape
    assert np.allclose(dydx, 2.0, atol=1e-6)
