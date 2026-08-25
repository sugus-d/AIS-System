"""Axilla GT coordinate validation."""

import numpy as np

from .._validate_utils import check, contour_distance

_MAX_CONTOUR_DIST_MM = 5.0  # A1: 腋窝点距轮廓最大允许距离（mm）


def validate(
    gt: dict,
    _features: dict,
    _vertices: np.ndarray,
    left_c: np.ndarray,
    right_c: np.ndarray,
    _relaxed: bool = False,
) -> list[dict]:
    """Validate axilla GT coordinates."""
    issues: list[dict] = []
    ax = gt["axilla"]
    st = gt["shoulder_transition"]
    wa = gt["waist"]

    # A1: 腋窝点必须在轮廓线上 5mm 以内——超出此距离说明检测点不在身体轮廓上
    for side, c in [("L", left_c), ("R", right_c)]:
        x, y, _z = ax[side]
        d = contour_distance(c, x, y)
        if issue := check(d < _MAX_CONTOUR_DIST_MM, "A1", f"axilla_{side} off contour {d:.0f}mm"):
            issues.append(issue)

    # A2: Y 方向解剖层级验证——腰部在最下方，腋窝居中，肩部过渡在最上方
    # 违反此顺序说明检测点落在了不可能的身体区域
    for side in ["L", "R"]:
        if issue := check(wa[side][1] < ax[side][1] < st[side][1], "A2", f"axilla_{side} Y out of order"):
            issues.append(issue)

    return issues
