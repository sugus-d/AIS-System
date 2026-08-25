"""颈根（neck_root）GT 坐标校验。"""

import numpy as np

from .._validate_utils import check, contour_distance, long_axis_angle

_MAX_CONTOUR_DIST_MM = 5.0  # N1: 颈根点距轮廓最大允许距离（mm）


def validate(
    gt: dict,
    _features: dict,
    _vertices: np.ndarray,
    left_c: np.ndarray,
    right_c: np.ndarray,
    relaxed: bool,
) -> list[dict]:
    """校验 neck_root GT 坐标是否符合约束标准。

    Args:
        gt: Ground Truth 字典，需包含 "neck_root" 键。
        _features: 特征字典（未使用）。
        _vertices: 网格顶点（未使用）。
        left_c: 左侧轮廓 (N, 2)。
        right_c: 右侧轮廓 (N, 2)。
        relaxed: 是否使用放宽阈值（body_asymmetry 或 arms=none 时）。

    Returns:
        问题列表，每个元素为 {"tag": str, "detail": str}。
    """
    issues: list[dict] = []
    nr: dict = gt["neck_root"]

    # N1: 轮廓距离 — 颈根点应在轮廓 5mm 内
    for side, c in [("L", left_c), ("R", right_c)]:
        x, y, _z = nr[side]
        d: float = contour_distance(c, x, y)
        if issue := check(d < _MAX_CONTOUR_DIST_MM, "N1", f"neck_root_{side} off contour {d:.0f}mm"):
            issues.append(issue)

    # N3: 左右 Y 对称 — 两侧颈根高度差应在阈值内
    dy_nr: float = abs(float(nr["L"][1]) - float(nr["R"][1]))
    th_nr_y: int = 40 if relaxed else 20
    if issue := check(dy_nr < th_nr_y, "N3", f"neck_root ΔY={dy_nr:.0f} > {th_nr_y}"):
        issues.append(issue)

    # N2: 颈宽 / 腋宽比 — 正常 30%~55%，放宽 25%~60%
    ax: dict = gt.get("axilla", {})
    neck_width: float = float(nr["R"][0]) - float(nr["L"][0])
    axilla_width: float = float(ax["R"][0]) - float(ax["L"][0]) if ax else 0.0
    neck_ax_ratio: float = neck_width / axilla_width * 100.0 if axilla_width > 0.0 else 0.0
    lo, hi = (25, 60) if relaxed else (30, 55)
    if issue := check(lo <= neck_ax_ratio <= hi, "N2", f"neck/ax={neck_ax_ratio:.0f}% not in [{lo},{hi}]"):
        issues.append(issue)

    # N4: 长轴转角 — 颈根应在过渡区 20°~70°（放宽 15°~80°）
    for side, c in [("L", left_c), ("R", right_c)]:
        ang: float = long_axis_angle(c, float(nr[side][0]), float(nr[side][1]))
        lo_a, hi_a = (15, 80) if relaxed else (20, 70)
        if issue := check(lo_a <= ang <= hi_a, "N4", f"neck_root_{side} long_axis={ang:.0f}° not in [{lo_a},{hi_a}]"):
            issues.append(issue)

    return issues
