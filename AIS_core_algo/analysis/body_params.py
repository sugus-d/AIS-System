"""体征参数计算（论文表 2）— 9 个体征参数的纯几何计算。

输入为 CSV 行风格 dict（`<landmark>_<side>(x,y,z)` 字符串键），输出 9 个参数：
Sh.IB/Sh.A/Sca.IB/Sca.A/ASIS.A/Trunk.L/Sh.W/Sh.AI/Pe.AI。
由 commands/export/raw_tables 批量制表与 prediction/measures 单 subject 预测共用。
"""

from __future__ import annotations

import math

from landmarks.constants import FLAT_KEYS, FLAT_SPINE_KEYS

COSMETIC_PARAMS = [
    "Sh.IB",
    "Sh.A",
    "Sca.IB",
    "Sca.A",
    "ASIS.A",
    "Trunk.L",
    "Sh.W",
    "Sh.AI",
    "Pe.AI",
]

# 三维坐标分量数 / 线性插值最少点数
_POINT_DIM = 3
_MIN_SPINE_POINTS = 2
# Cobb 角严重度分级阈值
COBB_MILD = 10
COBB_MODERATE = 20
COBB_SEVERE = 40


def _parse_landmark(coord_str: str) -> tuple[float, float, float]:
    s = coord_str.strip("()").replace(",", " ")
    parts = [float(x) for x in s.split() if x]
    return tuple(parts) if len(parts) == _POINT_DIM else (0.0, 0.0, 0.0)


def compute_cosmetic(row: dict) -> dict:
    st_L = _parse_landmark(row.get("shoulder_transition_L(x,y,z)", "(0,0,0)"))
    st_R = _parse_landmark(row.get("shoulder_transition_R(x,y,z)", "(0,0,0)"))
    sp_L = _parse_landmark(row.get("scapular_peaks_L(x,y,z)", "(0,0,0)"))
    sp_R = _parse_landmark(row.get("scapular_peaks_R(x,y,z)", "(0,0,0)"))
    wl_L = _parse_landmark(row.get("waist_lower_L(x,y,z)", "(0,0,0)"))
    wl_R = _parse_landmark(row.get("waist_lower_R(x,y,z)", "(0,0,0)"))
    sp = [_parse_landmark(row.get(f"{key}(x,y,z)", "(0,0,0)")) for key in FLAT_SPINE_KEYS]

    spine_pts = [(p[1], p[0]) for p in sp if p[0] != 0 or p[1] != 0]

    def spine_x_at(y: float) -> float:
        if len(spine_pts) < _MIN_SPINE_POINTS:
            return 0.0
        ys = [p[0] for p in spine_pts]
        xs = [p[1] for p in spine_pts]
        if y <= ys[0]:
            return xs[0]
        if y >= ys[-1]:
            return xs[-1]
        for i in range(len(ys) - 1):
            if ys[i] <= y <= ys[i + 1]:
                t = (y - ys[i]) / (ys[i + 1] - ys[i])
                return xs[i] + t * (xs[i + 1] - xs[i])
        return xs[0]

    def _y_coord(pt: tuple[float, float]) -> float:
        return pt[1]

    def _x_coord(pt: tuple[float, float]) -> float:
        return pt[0]

    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _angle(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx, dy = b[0] - a[0], b[1] - a[1]
        return math.degrees(math.atan2(dy, dx))

    mid_y = (_y_coord(st_L) + _y_coord(st_R)) / 2
    mid_y_w = (_y_coord(wl_L) + _y_coord(wl_R)) / 2
    sh_mid = ((st_L[0] + st_R[0]) / 2, mid_y)
    wl_mid = ((wl_L[0] + wl_R[0]) / 2, mid_y_w)
    dL_sh = abs(_x_coord(st_L) - spine_x_at(mid_y))
    dR_sh = abs(_x_coord(st_R) - spine_x_at(mid_y))
    dL_w = abs(_x_coord(wl_L) - spine_x_at(mid_y_w))
    dR_w = abs(_x_coord(wl_R) - spine_x_at(mid_y_w))

    return {
        "Sh.IB": _y_coord(st_R) - _y_coord(st_L),
        "Sh.A": _angle(st_L, st_R),
        "Sca.IB": _y_coord(sp_R) - _y_coord(sp_L),
        "Sca.A": _angle(sp_L, sp_R),
        "ASIS.A": _angle(wl_L, wl_R),
        "Trunk.L": _dist(sh_mid, wl_mid),
        "Sh.W": _dist(st_L, st_R),
        "Sh.AI": dL_sh / max(dR_sh, 1e-8),
        "Pe.AI": dL_w / max(dR_w, 1e-8),
    }


def gt_to_csv_row(gt: dict, subject_id: str) -> dict:
    """把扁平 18 键 ground_truth.json 转为 compute_cosmetic 可读的 CSV 行风格 dict。"""
    row: dict = {"subject_id": subject_id}
    for key in FLAT_KEYS:
        pt = gt.get(key)
        if pt is not None:
            x, y, z = (float(v) for v in pt)
            row[f"{key}(x,y,z)"] = f"({x},{y},{z})"
    return row
