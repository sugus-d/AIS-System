"""Scapular peak GT coordinate validation."""

import numpy as np

from .._validate_utils import check, load_curvature_at_point, z_top_percent


def validate(
    gt: dict,
    _features: dict,
    vertices: np.ndarray,
    _left_c: np.ndarray,
    _right_c: np.ndarray,
    relaxed: bool,
) -> list[dict]:
    """Validate scapular_peaks GT coordinates.

    Requires subject key in features for curvature loading.
    """
    issues: list[dict] = []
    sp = gt["scapular_peaks"]
    nr = gt["neck_root"]
    ax = gt["axilla"]

    # P1: X proximity to neck_root
    for side in ["L", "R"]:
        dx = abs(sp[side][0] - nr[side][0])
        th_p1 = 30 if relaxed else 20
        if issue := check(dx < th_p1, "P1", f"scapular_{side} ΔX={dx:.0f} > {th_p1}"):
            issues.append(issue)

    # P2: Y between axilla and neck_root
    for side in ["L", "R"]:
        lo_y, hi_y = sorted([nr[side][1], ax[side][1]])
        if issue := check(lo_y <= sp[side][1] <= hi_y, "P2", f"scapular_{side} Y out of [{lo_y:.0f},{hi_y:.0f}]"):
            issues.append(issue)

    # P3: left-right Y height match
    dy_sp = abs(sp["L"][1] - sp["R"][1])
    th_sp_y = 20 if relaxed else 10
    if issue := check(dy_sp < th_sp_y, "P3", f"scapular ΔY_LR={dy_sp:.0f} > {th_sp_y}"):
        issues.append(issue)

    # P4: Z top percent
    for side in ["L", "R"]:
        ztop = z_top_percent(vertices, nr[side][0], ax[side][1], nr[side][1], sp[side][2])
        th_p4 = 15 if relaxed else 10
        if issue := check(ztop < th_p4, "P4", f"scapular_{side} Z-top={ztop:.1f}% > {th_p4}%"):
            issues.append(issue)

    # P5: curvature positive
    subject = _features.get("_subject", "")
    for side in ["L", "R"]:
        cv = load_curvature_at_point(subject, sp[side][0], sp[side][1], vertices)
        if issue := check(cv > 0, "P5", f"scapular_{side} curv={cv:.4f} <= 0"):
            issues.append(issue)

    # P6: bilateral Z proximity
    dz = abs(sp["L"][2] - sp["R"][2])
    th_p6 = 80 if relaxed else 50
    if issue := check(dz < th_p6, "P6", f"scapular ΔZ={dz:.0f} > {th_p6}"):
        issues.append(issue)

    return issues
