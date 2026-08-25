"""Shoulder transition GT coordinate validation."""

import numpy as np

from .._validate_utils import check, contour_distance

_MAX_CONTOUR_DIST_MM = 5.0  # S1: 肩转点距轮廓最大允许距离（mm）


def validate(
    gt: dict,
    _features: dict,
    _vertices: np.ndarray,
    left_c: np.ndarray,
    right_c: np.ndarray,
    relaxed: bool,
) -> list[dict]:
    """Validate shoulder_transition GT coordinates."""
    issues: list[dict] = []
    st: dict = gt["shoulder_transition"]
    nr: dict = gt["neck_root"]
    ax: dict = gt["axilla"]

    neck_width: float = nr["R"][0] - nr["L"][0]

    # S1: contour distance
    for side, c in [("L", left_c), ("R", right_c)]:
        x, y, _z = st[side]
        d: float = contour_distance(c, x, y)
        if issue := check(d < _MAX_CONTOUR_DIST_MM, "S1", f"shoulder_transition_{side} off contour {d:.0f}mm"):
            issues.append(issue)

    # S2: Y relative position (order check)
    for side in ["L", "R"]:
        st_y: float = st[side][1]
        nr_y: float = nr[side][1]
        ax_y: float = ax[side][1]
        lo_y, hi_y = sorted([nr_y, ax_y])
        if issue := check(lo_y <= st_y <= hi_y, "S2", f"st_{side} Y={st_y:.0f} not in [{lo_y:.0f},{hi_y:.0f}]"):
            issues.append(issue)

    # S3: X spread ratio
    for side in ["L", "R"]:
        dx: float = abs(st[side][0] - nr[side][0])
        dx_ratio: float = dx / neck_width * 100 if neck_width > 0 else 0
        lo_dx, hi_dx = (35, 80) if relaxed else (40, 70)
        if issue := check(
            lo_dx <= dx_ratio <= hi_dx, "S3", f"st_{side} dX/neck={dx_ratio:.0f}% not in [{lo_dx},{hi_dx}]"
        ):
            issues.append(issue)

    return issues
