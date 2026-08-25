"""Debug payload builder for neck root detection."""

from __future__ import annotations

import numpy as np

from ..constants import AngleCandidate, NeckRoot


def _candidate_debug(candidate: AngleCandidate, side_name: str | None = None) -> dict:
    """把候选点转换为前端可序列化的调试字典。

    WHY：AngleCandidate 是 NamedTuple，包含 numpy 数组，不能直接 JSON 序列化。
    这里显式转为 Python 原生 float/list 类型，供前端可视化渲染使用。
    """
    # WHY：angle_deg > 90° 的候选点在躯干右侧的肩部区域，角度超出了颈根合理范围，
    # 标记为无效。
    _VALID_ANGLE_THRESHOLD = 90
    debug: dict = {
        "left": [float(candidate.left_pt[0]), float(candidate.left_pt[1])],
        "right": [float(candidate.right_pt[0]), float(candidate.right_pt[1])],
        "curr": [float(candidate.point[0]), float(candidate.point[1])],
        "left_dist": float(candidate.left_dist),
        "right_dist": float(candidate.right_dist),
        "angle_deg": float(candidate.angle_deg),
        "valid": bool(candidate.angle_deg > _VALID_ANGLE_THRESHOLD),
    }
    if side_name is not None:
        debug["side"] = side_name
    return debug


def _downsample_poly(poly: np.ndarray, max_n: int) -> list[list[float]]:
    """对轮廓点进行均匀下采样，限制输出点数不超过 max_n。

    WHY：调试数据量太大时前端渲染卡顿，用均匀采样降低点数。
    对于点数不超过上限的情况直接全部返回，避免不必要的采样。
    """
    arr = np.asarray(poly, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    n = len(arr)
    if n <= max_n:
        return arr.tolist()
    idx = np.linspace(0, n - 1, max_n).astype(int)
    return arr[idx].tolist()


def _interp_candidate_derivs(
    candidates: list[AngleCandidate],
    x_arr: np.ndarray | None,
    d_arr: np.ndarray | None,
) -> list[float]:
    """对候选点列表插值其导数数值。

    WHY：候选点来自轮廓段上的一阶导数筛选，但筛选后每个候选点的精确导数值
    已被丢弃（select_points_by_derivative 只返回点坐标）。这里用 np.interp
    重新插值出每个候选点处的导数，用于前端可视化显示导数分布。
    """
    if len(candidates) == 0:
        return []
    if x_arr is None or d_arr is None or len(x_arr) == 0 or len(d_arr) == 0:
        return [float("nan") for _ in candidates]
    vals: list[float] = []
    for candidate in candidates:
        try:
            v = float(np.interp(float(candidate.point[0]), x_arr, d_arr, left=np.nan, right=np.nan))
        except Exception:
            v = float("nan")
        vals.append(v)
    return vals


def build_neck_root_debug(
    *,
    is_debug: bool = True,
    waist_points: np.ndarray,
    left_c: np.ndarray,
    right_c: np.ndarray,
    bin_info: list[dict],
    mode_idx: int,
    mode_width: float,
    waist_w: float,
    left_candidates: list[AngleCandidate],
    right_candidates: list[AngleCandidate],
    left_x: np.ndarray,
    left_d: np.ndarray,
    right_x: np.ndarray,
    right_d: np.ndarray,
    left_best: AngleCandidate,
    right_best: AngleCandidate,
    include_deriv: bool = True,
    max_contour_points: int = NeckRoot.MAX_CONTOUR_POINTS,
) -> dict:
    """构造 neck root 的调试 payload，全部转为可 JSON 序列化的 Python 原生类型。

    WHY：调试信息用于渲染前端可视化面板，所有 numpy 数组和 NamedTuple 必须
    转为 dict / list / float 等可 JSON 序列化的格式。
    """
    if not is_debug:
        return {}

    hist_bins = [float(item["bin_lo"]) for item in bin_info]
    hist_bins.append(float(bin_info[-1]["bin_hi"]))
    hist_counts = [int(item["count"]) for item in bin_info]
    bin_debug = [
        {
            "x_left": float(item["x_left"]),
            "x_right": float(item["x_right"]),
            "y": float(item["y"]),
            "width": float(item["width"]),
            "is_mode": bool(item["is_mode"]),
            "count": int(item["count"]),
            "bin_lo": float(item["bin_lo"]),
            "bin_hi": float(item["bin_hi"]),
        }
        for item in bin_info
        if item["x_left"] is not None
        and item["x_right"] is not None
        and item["y"] is not None
        and item["width"] is not None
    ]
    left_candidates_vis = [[float(c.point[0]), float(c.point[1]), float(c.angle_deg)] for c in left_candidates]
    right_candidates_vis = [[float(c.point[0]), float(c.point[1]), float(c.angle_deg)] for c in right_candidates]

    neck_root_points = np.vstack([left_best.point[:2], right_best.point[:2]])
    neck_width = float(np.linalg.norm(neck_root_points[1] - neck_root_points[0]))
    neck_width_ratio = neck_width / (mode_width if mode_width > 0 else 1.0)
    angle_debug = {
        "left": _candidate_debug(left_best),
        "right": _candidate_debug(right_best),
        "neck_width": neck_width,
        "neck_width_ratio": round(neck_width_ratio, 2),
        "neck_width_ok": bool(neck_width_ratio < NeckRoot.NECK_WIDTH_OK_RATIO),
    }

    if include_deriv:
        left_candidates_deriv = _interp_candidate_derivs(left_candidates, left_x, left_d)
        right_candidates_deriv = _interp_candidate_derivs(right_candidates, right_x, right_d)
    else:
        left_candidates_deriv = []
        right_candidates_deriv = []
    return {
        "waist_points": np.asarray(waist_points, dtype=float).tolist(),
        "left_xy_sm": _downsample_poly(left_c, max_contour_points),
        "right_xy_sm": _downsample_poly(right_c, max_contour_points),
        "bin_debug": bin_debug,
        "hist_bins": hist_bins,
        "hist_counts": hist_counts,
        "hist_mode_bin": int(mode_idx),
        "neck_max": float(waist_w * NeckRoot.WAIST_W_UPPER_RATIO),
        "w_min": float(mode_width),
        "waist_w": float(waist_w),
        "left_candidates": left_candidates_vis,
        "right_candidates": right_candidates_vis,
        "left_candidates_deriv": left_candidates_deriv,
        "right_candidates_deriv": right_candidates_deriv,
        "angle_debug": angle_debug,
    }
