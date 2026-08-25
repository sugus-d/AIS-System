"""Waist GT coordinate validation."""

import numpy as np

from .._validate_utils import check, contour_distance, long_axis_angle

_MAX_CONTOUR_DIST_MM = 5.0   # W1: 腰点距轮廓最大允许距离（mm）
_MAX_LONG_AXIS_ANGLE = 10.0  # W2: 长轴转角上限（°）


def validate(
    gt: dict,
    _features: dict,
    _vertices: np.ndarray,
    left_c: np.ndarray,
    right_c: np.ndarray,
    _relaxed: bool = False,
) -> list[dict]:
    """Validate waist GT coordinates."""
    issues: list[dict] = []
    wa = gt["waist"]

    # W1: contour distance
    for side, c in [("L", left_c), ("R", right_c)]:
        x, y, _z = wa[side]
        d = contour_distance(c, x, y)
        if issue := check(d < _MAX_CONTOUR_DIST_MM, "W1", f"waist_{side} off contour {d:.0f}mm"):
            issues.append(issue)

    # W2: long axis angle < 10 deg (no relaxed threshold per spec)
    for side, c in [("L", left_c), ("R", right_c)]:
        ang = long_axis_angle(c, wa[side][0], wa[side][1])
        if issue := check(ang < _MAX_LONG_AXIS_ANGLE, "W2", f"waist_{side} long_axis={ang:.0f}° >= 10°"):
            issues.append(issue)

    return issues
