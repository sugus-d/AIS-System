"""UV-space anatomical region polygon definitions (landmark_regions 定义区).

Region 多边形（left/right 双侧）、点包含测试、版本化候选区域入口。
候选区域 / 双边形对生成见 :mod:`._regions_gen`，特征计算见 :mod:`._features`。

Public: SEG_SHOULDER/THORACIC/LUMBAR/PELVIC, CANDIDATE_VERSIONS,
build_candidate_polygons（classify_by_region 在 _features）。
"""

from __future__ import annotations

import numpy as np

from parameterization.template import TEMPLATE_LANDMARKS

from ._regions_gen import _get_bilateral, _get_candidates

_LM = TEMPLATE_LANDMARKS

SEG_SHOULDER = 2
SEG_THORACIC = 0
SEG_LUMBAR = 1
SEG_PELVIC = 3

_N_REGIONS = 4

# ---------------------------------------------------------------------------
# UV-space polygon definitions
# ---------------------------------------------------------------------------
# Each bilaterally-symmetric region is represented by a left polygon (U ≤ 0)
# and a right polygon (U ≥ 0).  Spine landmarks (neck_root…waist) sit on the
# midline at U = 0 and serve as the shared boundary between left and right.

_LM = TEMPLATE_LANDMARKS


def _uv(*keys: str) -> np.ndarray:
    """Convert landmark key(s) to an (N, 2) UV array."""
    return np.array([_LM[k] for k in keys], dtype=np.float64)


# Left polygons — counter-clockwise, from the outer edge to the spine.
_LEFT_POLYGONS: dict[int, np.ndarray] = {
    SEG_SHOULDER: _uv(
        "shoulder_transition_L",
        "neck_root_L",
        "neck_root_spine_point",
        "scapular_spine_point",
        "scapular_peaks_L",
    ),
    SEG_THORACIC: _uv(
        "scapular_peaks_L",
        "scapular_spine_point",
        "axilla_spine_point",
        "axilla_L",
    ),
    SEG_LUMBAR: _uv(
        "axilla_L",
        "axilla_spine_point",
        "waist_spine_point",
        "waist_L",
    ),
    # Pelvic — overwritten below with synthetic-depth vertices.
}

_RIGHT_POLYGONS: dict[int, np.ndarray] = {
    SEG_SHOULDER: _uv(
        "shoulder_transition_R",
        "neck_root_R",
        "neck_root_spine_point",
        "scapular_spine_point",
        "scapular_peaks_R",
    ),
    SEG_THORACIC: _uv(
        "scapular_peaks_R",
        "scapular_spine_point",
        "axilla_spine_point",
        "axilla_R",
    ),
    SEG_LUMBAR: _uv(
        "axilla_R",
        "axilla_spine_point",
        "waist_spine_point",
        "waist_R",
    ),
    # Pelvic — overwritten below with synthetic-depth vertices.
}

# Waist Y-coordinate — everything below is pelvic.
_WAIST_V: float = _LM["waist_L"][1]  # -3.0

# Pelvic polygons are unbounded at the bottom; we extend them with synthetic
# vertices far below the actual mesh to guarantee containment.
_PELVIC_DEPTH: float = -10.0
_LEFT_POLYGONS[SEG_PELVIC] = np.array(
    [
        [_LM["waist_L"][0], _LM["waist_L"][1]],
        [_LM["waist_spine_point"][0], _LM["waist_spine_point"][1]],
        [0.0, _PELVIC_DEPTH],
        [-2.5, _PELVIC_DEPTH],
    ],
    dtype=np.float64,
)

_RIGHT_POLYGONS[SEG_PELVIC] = np.array(
    [
        [_LM["waist_R"][0], _LM["waist_R"][1]],
        [_LM["waist_spine_point"][0], _LM["waist_spine_point"][1]],
        [0.0, _PELVIC_DEPTH],
        [2.5, _PELVIC_DEPTH],
    ],
    dtype=np.float64,
)

# All region IDs processed by polygon classification (pelvic included).
_POLYGON_REGIONS: list[int] = [
    SEG_SHOULDER,
    SEG_THORACIC,
    SEG_LUMBAR,
    SEG_PELVIC,
]

# ---------------------------------------------------------------------------
# Point-in-polygon (winding number)
# ---------------------------------------------------------------------------


def _points_in_polygon(pts: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Test whether points lie inside a polygon (winding-number / ray-casting).

    Implements the classic even-odd crossing-number algorithm with careful
    handling of horizontal edges and boundary points.

    Args:
        pts: (N, 2) array of query points.
        polygon: (M, 2) array — polygon vertices in CW or CCW order.

    Returns:
        (N,) bool array — *True* for points inside the polygon (boundary
        inclusive).
    """
    if len(polygon) < 3:  # noqa: PLR2004
        return np.zeros(len(pts), dtype=bool)

    pts = np.asarray(pts, dtype=np.float64)
    poly = np.asarray(polygon, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    n = len(poly)
    inside = np.zeros(len(pts), dtype=bool)

    # 射线法（even-odd rule）：统计从点出发向右水平射线与多边形的交点数
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        # 水平边不做计算——与射线平行，不产生有效交点
        if y1 == y2:
            continue

        # 使用半开区间：只计射线穿过边内部的情况，排除上端点。
        # 这样共享顶点不会被重复计数。
        cond = (y > min(y1, y2)) & (y <= max(y1, y2))
        if not cond.any():
            continue

        # 计算射线与边的交点 X 坐标
        x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        # 每次跨越射线时翻转 inside 状态
        inside[cond] ^= x[cond] <= x_intersect[cond]

    return inside
def _seed_centroids() -> dict[tuple[int, int], np.ndarray]:
    """Return approximate region centroids from template landmark positions.

    Used as a fallback when no vertices have been classified yet.
    """
    return {
        (SEG_SHOULDER, 0): _uv("shoulder_transition_L", "neck_root_L", "scapular_spine_point", "scapular_peaks_L").mean(axis=0),
        (SEG_SHOULDER, 1): _uv("shoulder_transition_R", "neck_root_R", "scapular_spine_point", "scapular_peaks_R").mean(axis=0),
        (SEG_THORACIC, 0): _uv("scapular_peaks_L", "scapular_spine_point", "axilla_spine_point", "axilla_L").mean(axis=0),
        (SEG_THORACIC, 1): _uv("scapular_peaks_R", "scapular_spine_point", "axilla_spine_point", "axilla_R").mean(axis=0),
        (SEG_LUMBAR, 0): _uv("axilla_L", "axilla_spine_point", "waist_spine_point", "waist_L").mean(axis=0),
        (SEG_LUMBAR, 1): _uv("axilla_R", "axilla_spine_point", "waist_spine_point", "waist_R").mean(axis=0),
        (SEG_PELVIC, 0): _uv("waist_L", "waist_spine_point").mean(axis=0) + [0.0, -1.0],
        (SEG_PELVIC, 1): _uv("waist_R", "waist_spine_point").mean(axis=0) + [0.0, -1.0],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Versioned public API
# ═══════════════════════════════════════════════════════════════════════════

CANDIDATE_VERSIONS = ("cross_midline", "bilateral")
"""Valid values for the *version* parameter in region-building functions.

- ``"cross_midline"`` (V1): 110 single-polygon regions spanning both sides.
- ``"bilateral"`` (V2): 225 strictly unilateral polygon pairs (default).
"""


def build_candidate_polygons(version: str = "bilateral") -> list[dict]:
    """Build candidate region polygon definitions.

    Args:
        version: ``"cross_midline"`` (V1, 110) or ``"bilateral"`` (V2, 225).

    Returns:
        List of dicts.  V1 dicts have keys ``id``, ``name``, ``polygon``.
        V2 dicts have keys ``id``, ``name``, ``left_polygon``, ``right_polygon``.
    """
    if version == "cross_midline":
        polys: list[dict] = []
        for i, (name, keys) in enumerate(_get_candidates()):
            poly = _uv(*keys)
            if len(poly) < 3:  # noqa: PLR2004
                continue
            polys.append({"id": i, "name": name, "polygon": poly})
        return polys

    return [
        {"id": i, "name": name, "left_polygon": left_poly, "right_polygon": right_poly}
        for i, (name, left_poly, right_poly) in enumerate(_get_bilateral())
    ]
