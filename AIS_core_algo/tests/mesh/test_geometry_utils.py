import numpy as np

from landmarks.geometry import is_contour_ccw


def test_is_contour_ccw_closed_ring() -> None:
    ccw = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    cw = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    assert is_contour_ccw(ccw) is True
    assert is_contour_ccw(cw) is False


def test_is_contour_ccw_open_curve_fallback() -> None:
    open_curve = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )

    assert is_contour_ccw(open_curve) is True
