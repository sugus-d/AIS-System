"""Region 面积分布 — UV 坐标下各 region 多边形在 5 个解剖区域的面积占比。

从 ci_decompose 拆出（等价重构）：region 归属判定（CI 反解按区域分摊、
特征展示名区域前缀）统一走本模块。面积分布用多边形 v 带裁剪 + 蒙特卡洛
采样估算，缓存避免重复计算（225 个 region 全量构建一次）。

供:
  - ci_decompose.aggregate_by_region  按区域分摊贡献
  - ci_display.feature_display_name   区域前缀展示
"""

from __future__ import annotations

import numpy as np

_MIN_POLY_VERTICES = 3
"""多边形有效所需的最少顶点数。"""

_BAND_MAJORITY = 0.70
"""区域归属判定中主导 band 的占比阈值。"""

# 5 个解剖区域 + UV v 带边界（自上而下）
_BANDS = [
    ("Shoulder", 1.75, 2.0),
    ("Scapula", 0, 1.75),
    ("Axilla", -1.5, 0),
    ("Waist", -3.0, -1.5),
    ("Pelvis", -4.0, -3.0),
]


def _sutherland_hodgman(pts: np.ndarray, v_lo: float, v_hi: float) -> np.ndarray | None:
    """按 v 带 [v_lo, v_hi) 裁剪凸多边形，返回裁剪后多边形或 None。

    Sutherland–Hodgman 逐边裁剪：两条水平边界各裁剪一次，
    保留 v 在带内的顶点并插入边界交点。
    """
    for bound, keep_above in [(v_lo, True), (v_hi, False)]:
        if len(pts) < _MIN_POLY_VERTICES:
            return None
        out = []
        for i in range(len(pts)):
            cur = pts[i]
            prev = pts[i - 1]
            cur_in = cur[1] >= bound if keep_above else cur[1] < bound
            prev_in = prev[1] >= bound if keep_above else prev[1] < bound
            # 边进入带内：插入交点（除非交点恰为顶点本身）
            if cur_in and not prev_in and cur[1] != bound:
                t = (bound - prev[1]) / (cur[1] - prev[1])
                out.append(prev + t * (cur - prev))
            if cur_in:
                out.append(cur)
            elif prev_in:
                t = (bound - prev[1]) / (cur[1] - prev[1])
                out.append(prev + t * (cur - prev))
        pts = np.array(out) if out else np.empty((0, 2))
    return pts if len(pts) >= _MIN_POLY_VERTICES else None


def _poly_area(pts: np.ndarray | None) -> float:
    """鞋带公式计算多边形面积；无效多边形返回 0。"""
    if pts is None or len(pts) < _MIN_POLY_VERTICES:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _build_region_area_cache() -> dict[str, dict[str, float]]:
    """预计算全部 225 个 region 的 UV 面积分布（{region 名: {区域: 占比}}）。

    蒙特卡洛采样估算：region 多边形包围盒内均匀撒点，射线法判定点是否落在
    左/右多边形内，落在带内比例即面积占比。种子固定（42）保证结果可复现。
    """
    from features.extractors.asymmetry.regions import _get_pairs
    rng = np.random.default_rng(42)
    cache = {}
    for name, left, right in _get_pairs():
        total = _poly_area(left) + _poly_area(right)
        if total == 0:
            continue
        # 包围盒内采样（左+右多边形整体）
        all_pts = np.vstack([left, right])
        min_x, max_x = all_pts[:, 0].min(), all_pts[:, 0].max()
        min_y, max_y = all_pts[:, 1].min(), all_pts[:, 1].max()
        bb_area = (max_x - min_x) * (max_y - min_y)
        if bb_area == 0:
            continue
        n_samples = 5000
        pts = rng.uniform(0, 1, (n_samples, 2))
        pts[:, 0] = min_x + pts[:, 0] * (max_x - min_x)
        pts[:, 1] = min_y + pts[:, 1] * (max_y - min_y)
        # 射线法：左侧多边形内判定
        in_left = np.zeros(n_samples, dtype=bool)
        n = len(left)
        x, y = pts[:, 0], pts[:, 1]
        for i in range(n):
            x1, y1 = left[i]
            x2, y2 = left[(i + 1) % n]
            if y1 == y2:
                continue
            cond = (y > min(y1, y2)) & (y <= max(y1, y2))
            if not cond.any():
                continue
            x_int = (x2 - x1) * (y[cond] - y1) / (y2 - y1) + x1
            in_left[cond] ^= x[cond] <= x_int
        # 射线法：右侧多边形内判定
        in_right = np.zeros(n_samples, dtype=bool)
        n = len(right)
        for i in range(n):
            x1, y1 = right[i]
            x2, y2 = right[(i + 1) % n]
            if y1 == y2:
                continue
            cond = (y > min(y1, y2)) & (y <= max(y1, y2))
            if not cond.any():
                continue
            x_int = (x2 - x1) * (y[cond] - y1) / (y2 - y1) + x1
            in_right[cond] ^= x[cond] <= x_int
        inside = in_left | in_right
        n_inside = inside.sum()
        if n_inside == 0:
            continue
        in_pts = pts[inside]
        dist = {}
        for bname, v_lo, v_hi in _BANDS:
            n_band = ((in_pts[:, 1] >= v_lo) & (in_pts[:, 1] < v_hi)).sum()
            if n_band > 0:
                dist[bname] = n_band / n_inside
        cache[name] = dist
    return cache


_REGION_AREA_CACHE: dict[str, dict[str, float]] | None = None


def _get_region_distribution(name: str) -> dict[str, float]:
    """按特征名取该 region 在 5 个解剖区域的面积分布（如 {"Shoulder": 0.44, "Scapula": 0.56}）。

    特征名形如 ``nr_p0_p1``（去掉测量后缀后的 region 键）；不匹配任何 region 返回空 dict。
    """
    global _REGION_AREA_CACHE
    if _REGION_AREA_CACHE is None:
        _REGION_AREA_CACHE = _build_region_area_cache()

    import re
    key = re.sub(r"_(height|mean_curv|gauss_curv|roughness|normal_angle|normal_vector_cos|normal_vector|normal_vector_sin)(__pw|_pw|_dm)?$", "", name)
    return _REGION_AREA_CACHE.get(key, {})


def _get_region_display(name: str) -> str:
    """按面积占比判定特征归属区域展示名。

    主导 band 占比 >70% 或仅有单一 band → 取该区域名；
    否则取占比前两位叠加（如 "Shoulder+Scapula"）。不匹配返回空串。
    """
    global _REGION_AREA_CACHE
    if _REGION_AREA_CACHE is None:
        _REGION_AREA_CACHE = _build_region_area_cache()

    # 去掉测量后缀得到 region 键（如 "wa_wl_p0_p4_normal_angle__pw" → "wa_wl_p0_p4"）
    import re
    key = re.sub(r"_(height|mean_curv|gauss_curv|roughness|normal_angle|normal_vector_cos|normal_vector|normal_vector_sin)(__pw|_pw|_dm)?$", "", name)
    if key not in _REGION_AREA_CACHE:
        return ""

    dist = _REGION_AREA_CACHE[key]
    sorted_bands = sorted(dist.items(), key=lambda x: -x[1])
    if not sorted_bands:
        return ""

    top = sorted_bands[0]
    if top[1] > _BAND_MAJORITY or len(sorted_bands) == 1:
        return top[0]
    return f"{top[0]}+{sorted_bands[1][0]}"
