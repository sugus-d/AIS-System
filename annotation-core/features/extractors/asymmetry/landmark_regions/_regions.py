"""UV-space anatomical region polygon definitions (landmark_regions 定义区).

Region 多边形（left/right 双侧）、点包含测试、V1/V2 候选区域生成。
特征计算见同包 :mod:`._features`。

Public: SEG_SHOULDER/THORACIC/LUMBAR/PELVIC, CANDIDATE_VERSIONS,
build_candidate_polygons（classify_by_region 在 _features）。
"""

from __future__ import annotations

import numpy as np

from parameterization.template import TEMPLATE_LANDMARKS

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
# and a right polygon (U ≥ 0).  Spine landmarks (spine_P0…P3) sit on the
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
        "spine_P0",
        "spine_P1",
        "scapular_peaks_L",
    ),
    SEG_THORACIC: _uv(
        "scapular_peaks_L",
        "spine_P1",
        "spine_P2",
        "axilla_L",
    ),
    SEG_LUMBAR: _uv(
        "axilla_L",
        "spine_P2",
        "spine_P3",
        "waist_L",
    ),
    # Pelvic — overwritten below with synthetic-depth vertices.
}

_RIGHT_POLYGONS: dict[int, np.ndarray] = {
    SEG_SHOULDER: _uv(
        "shoulder_transition_R",
        "neck_root_R",
        "spine_P0",
        "spine_P1",
        "scapular_peaks_R",
    ),
    SEG_THORACIC: _uv(
        "scapular_peaks_R",
        "spine_P1",
        "spine_P2",
        "axilla_R",
    ),
    SEG_LUMBAR: _uv(
        "axilla_R",
        "spine_P2",
        "spine_P3",
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
        [_LM["spine_P3"][0], _LM["spine_P3"][1]],
        [0.0, _PELVIC_DEPTH],
        [-2.5, _PELVIC_DEPTH],
    ],
    dtype=np.float64,
)

_RIGHT_POLYGONS[SEG_PELVIC] = np.array(
    [
        [_LM["waist_R"][0], _LM["waist_R"][1]],
        [_LM["spine_P3"][0], _LM["spine_P3"][1]],
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
        (SEG_SHOULDER, 0): _uv("shoulder_transition_L", "neck_root_L", "spine_P1", "scapular_peaks_L").mean(axis=0),
        (SEG_SHOULDER, 1): _uv("shoulder_transition_R", "neck_root_R", "spine_P1", "scapular_peaks_R").mean(axis=0),
        (SEG_THORACIC, 0): _uv("scapular_peaks_L", "spine_P1", "spine_P2", "axilla_L").mean(axis=0),
        (SEG_THORACIC, 1): _uv("scapular_peaks_R", "spine_P1", "spine_P2", "axilla_R").mean(axis=0),
        (SEG_LUMBAR, 0): _uv("axilla_L", "spine_P2", "spine_P3", "waist_L").mean(axis=0),
        (SEG_LUMBAR, 1): _uv("axilla_R", "spine_P2", "spine_P3", "waist_R").mean(axis=0),
        (SEG_PELVIC, 0): _uv("waist_L", "spine_P3").mean(axis=0) + [0.0, -1.0],
        (SEG_PELVIC, 1): _uv("waist_R", "spine_P3").mean(axis=0) + [0.0, -1.0],
    }
# ═══════════════════════════════════════════════════════════════════════════
# V1 — Cross-midline candidate regions (legacy, 110 regions)
# ═══════════════════════════════════════════════════════════════════════════
# Each polygon spans both left and right sides; asymmetry computed by
# splitting polygon vertices at U=0.  Includes asymmetric crosses that mix
# left and right edge landmarks.
#
# Kept for backward compatibility — previous results (features_550d.csv etc.)
# were generated with this version.
#
# Asymmetry = |mean(U≤0) - mean(U>0)| for points inside polygon.

# Short→full name mapping for landmark keys (used in _gen_candidate_regions)
_S = {
    "NR_L": "neck_root_L",
    "NR_R": "neck_root_R",
    "ST_L": "shoulder_transition_L",
    "ST_R": "shoulder_transition_R",
    "SP_L": "scapular_peaks_L",
    "SP_R": "scapular_peaks_R",
    "AX_L": "axilla_L",
    "AX_R": "axilla_R",
    "WA_L": "waist_L",
    "WA_R": "waist_R",
    "WL_L": "waist_lower_L",
    "WL_R": "waist_lower_R",
    "P0": "spine_P0",
    "P1": "spine_P1",
    "P2": "spine_P2",
    "P3": "spine_P3",
    "P4": "spine_P4",
    "P5": "spine_P5",
}


def _def(name: str, *short_keys: str) -> tuple[str, tuple[str, ...]]:
    """Define a candidate region using short landmark names."""
    return (name, tuple(_S[k] for k in short_keys))


def _gen_candidate_regions() -> list[tuple[str, tuple[str, ...]]]:
    """Systematically generate 80+ candidate UV polygon regions from 18 landmarks."""
    regions: list[tuple[str, tuple[str, ...]]] = []

    # ── A. Band regions ──
    regions.append(_def("top_band", "ST_L", "NR_L", "P0", "NR_R", "ST_R"))
    regions.append(_def("scapular_band", "ST_L", "SP_L", "P1", "SP_R", "ST_R"))
    regions.append(_def("axilla_band", "SP_L", "AX_L", "P2", "AX_R", "SP_R"))
    regions.append(_def("waist_band", "AX_L", "WA_L", "P3", "WA_R", "AX_R"))
    regions.append(_def("pelvic_band", "WA_L", "WL_L", "P4", "WL_R", "WA_R"))
    regions.append(_def("inner_top", "NR_L", "SP_L", "P1", "SP_R", "NR_R"))
    regions.append(_def("inner_axilla", "SP_L", "AX_L", "P2", "AX_R", "SP_R"))

    # ── B. Triangle wedges ──
    left_keys = ["NR_L", "ST_L", "SP_L", "AX_L", "WA_L", "WL_L"]
    right_keys = ["NR_R", "ST_R", "SP_R", "AX_R", "WA_R", "WL_R"]
    B_SP = [
        ("P0", "P1"),
        ("P1", "P5"),
        ("P5", "P2"),
        ("P2", "P3"),
        ("P3", "P4"),
        ("P0", "P2"),
        ("P1", "P2"),
        ("P2", "P5"),
        ("P5", "P3"),
        ("P2", "P4"),
        ("P0", "P5"),
        ("P0", "P3"),
        ("P1", "P4"),
    ]
    for left_lm, right_lm in zip(left_keys, right_keys, strict=True):
        prefix = left_lm.replace("_L", "").lower()
        for sp_top, sp_bot in B_SP:
            name = f"{prefix}_{sp_top.lower()}_{sp_bot.lower()}"
            regions.append(_def(name, left_lm, sp_top, sp_bot, right_lm))

    # ── C. Cross-level bands ──
    regions.append(_def("nr_sp_band", "NR_L", "SP_L", "P1", "SP_R", "NR_R"))
    regions.append(_def("st_ax_band", "ST_L", "AX_L", "P2", "AX_R", "ST_R"))
    regions.append(_def("sp_wa_band", "SP_L", "WA_L", "P3", "WA_R", "SP_R"))
    regions.append(_def("ax_wl_band", "AX_L", "WL_L", "P4", "WL_R", "AX_R"))
    regions.append(_def("nr_ax_band", "NR_L", "AX_L", "P2", "AX_R", "NR_R"))
    regions.append(_def("st_wa_band", "ST_L", "WA_L", "P3", "WA_R", "ST_R"))
    regions.append(_def("sp_wl_band", "SP_L", "WL_L", "P4", "WL_R", "SP_R"))
    regions.append(_def("nr_wa_band", "NR_L", "WA_L", "P3", "WA_R", "NR_R"))
    regions.append(_def("st_wl_band", "ST_L", "WL_L", "P4", "WL_R", "ST_R"))

    # ── D. Tall vertical strips (full height span) ──
    regions.append(_def("full_outer_band", "ST_L", "AX_L", "WA_L", "WL_L", "WL_R", "WA_R", "AX_R", "ST_R"))
    regions.append(_def("full_inner_band", "NR_L", "SP_L", "AX_L", "WA_L", "WA_R", "AX_R", "SP_R", "NR_R"))
    regions.append(_def("nr_wl_band", "NR_L", "AX_L", "WL_L", "WL_R", "AX_R", "NR_R"))
    regions.append(_def("st_ax_p5", "ST_L", "AX_L", "P5", "AX_R", "ST_R"))
    regions.append(_def("nr_p3_band", "NR_L", "WA_L", "P3", "WA_R", "NR_R"))
    regions.append(_def("st_p4_band", "ST_L", "WL_L", "P4", "WL_R", "ST_R"))

    # ── E. P5-centred ──
    regions.append(_def("nr_p5_band", "NR_L", "AX_L", "P5", "AX_R", "NR_R"))
    regions.append(_def("st_p5_band", "ST_L", "AX_L", "P5", "AX_R", "ST_R"))
    regions.append(_def("sp_p5_band", "SP_L", "AX_L", "P5", "AX_R", "SP_R"))
    regions.append(_def("p5_wa_band", "AX_L", "WA_L", "P3", "P5", "WA_R", "AX_R"))

    # ── F. P4-centred ──
    regions.append(_def("ax_p4_band", "AX_L", "WL_L", "P4", "WL_R", "AX_R"))

    # ── G. Asymmetric cross ──
    regions.append(_def("nrl_axr", "NR_L", "P0", "P2", "AX_R"))
    regions.append(_def("stl_war", "ST_L", "P0", "P3", "WA_R"))
    regions.append(_def("axl_nrr", "AX_L", "P2", "P0", "NR_R"))
    regions.append(_def("wal_str", "WA_L", "P3", "P2", "ST_R"))
    regions.append(_def("nrl_war", "NR_L", "P0", "P3", "WA_R"))

    return regions


_CACHED_CANDIDATES: list[tuple[str, tuple[str, ...]]] | None = None


def _get_candidates() -> list[tuple[str, tuple[str, ...]]]:
    global _CACHED_CANDIDATES
    if _CACHED_CANDIDATES is None:
        _CACHED_CANDIDATES = _gen_candidate_regions()
    return _CACHED_CANDIDATES


# ═══════════════════════════════════════════════════════════════════════════
# V2 — Bilateral region pairs (current, 225 pairs)
# ═══════════════════════════════════════════════════════════════════════════
# Each asymmetry candidate is a *pair* of strictly unilateral UV polygons:
#   • Left polygon  → only left-edge landmarks (eg NR_L, ST_L, …, WL_L) + spine
#   • Right polygon → only right-edge landmarks (NR_R, ST_R, …, WL_R) + same spine
#
# Spine landmarks (P0-P5) sit on the midline (U=0) and are allowed in both
# left and right polygons.  Edge landmarks are strictly unilateral.
#
# Asymmetry = |mean(inside left polygon) - mean(inside right polygon)|

# Edge landmarks (top → bottom) and their short name prefixes
_LEFT_EDGE = ["NR_L", "ST_L", "SP_L", "AX_L", "WA_L", "WL_L"]
_RIGHT_EDGE = ["NR_R", "ST_R", "SP_R", "AX_R", "WA_R", "WL_R"]
_EDGE_SHORT = ["nr", "st", "sp", "ax", "wa", "wl"]

# Spine landmarks sorted top → bottom
_SPINE = ["P0", "P1", "P2", "P5", "P3", "P4"]
_SPINE_SHORT = ["p0", "p1", "p2", "p5", "p3", "p4"]
_N_SPINE = len(_SPINE)

# All spine pairs (top_index, bottom_index, short_name_str) — 15 对
_SPINE_PAIRS: list[tuple[int, int, str]] = [
    (i, j, f"{_SPINE_SHORT[i]}_{_SPINE_SHORT[j]}") for i in range(_N_SPINE) for j in range(i + 1, _N_SPINE)
]

# Short → full landmark name lookup (独立于 V1 的 _S，因为 V2 不需要反向解析）
_SHORT_TO_FULL: dict[str, str] = {
    "NR_L": "neck_root_L",
    "NR_R": "neck_root_R",
    "ST_L": "shoulder_transition_L",
    "ST_R": "shoulder_transition_R",
    "SP_L": "scapular_peaks_L",
    "SP_R": "scapular_peaks_R",
    "AX_L": "axilla_L",
    "AX_R": "axilla_R",
    "WA_L": "waist_L",
    "WA_R": "waist_R",
    "WL_L": "waist_lower_L",
    "WL_R": "waist_lower_R",
    "P0": "spine_P0",
    "P1": "spine_P1",
    "P2": "spine_P2",
    "P5": "spine_P5",
    "P3": "spine_P3",
    "P4": "spine_P4",
}


def _bilat_poly(left_keys: tuple[str, ...], right_keys: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Build left and right UV polygon arrays from short landmark key sequences."""
    return (
        np.array([_LM[_SHORT_TO_FULL[k]] for k in left_keys], dtype=np.float64),
        np.array([_LM[_SHORT_TO_FULL[k]] for k in right_keys], dtype=np.float64),
    )


def _gen_bilateral_pairs() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Generate (name, left_uv_poly, right_uv_poly) for all candidate pairs.

    Every entry satisfies:
      * Left polygon contains only ``_L`` edge landmarks + spine (U=0) points.
      * Right polygon contains only ``_R`` edge landmarks + the **same** spine points.
      * No polygon mixes ``_L`` and ``_R`` edge landmarks.
    """
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _add(name: str, left_keys: tuple[str, ...], right_keys: tuple[str, ...]) -> None:
        lpoly, rpoly = _bilat_poly(left_keys, right_keys)
        if len(lpoly) >= 3 and len(rpoly) >= 3:  # noqa: PLR2004
            pairs.append((name, lpoly, rpoly))

    # ── A. Single-edge triangles (1 edge landmark + 2 spine points) ──
    # 6 edges × 15 spine_pairs = 90
    for e_idx in range(6):
        elk = _LEFT_EDGE[e_idx]
        erk = _RIGHT_EDGE[e_idx]
        en = _EDGE_SHORT[e_idx]
        for si, sj, spn in _SPINE_PAIRS:
            _add(
                f"{en}_{spn}",
                (elk, _SPINE[sj], _SPINE[si]),  # left:  edge → lower → upper
                (_SPINE[sj], erk, _SPINE[si]),  # right: lower → edge → upper
            )

    # ── B. Two-edge vertical bands (2 adjacent edges + 2 spine points) ──
    # 5 edge-pairs × 15 spine_pairs = 75
    for e_idx in range(5):
        elk1, elk2 = _LEFT_EDGE[e_idx], _LEFT_EDGE[e_idx + 1]
        erk1, erk2 = _RIGHT_EDGE[e_idx], _RIGHT_EDGE[e_idx + 1]
        en = f"{_EDGE_SHORT[e_idx]}_{_EDGE_SHORT[e_idx + 1]}"
        for si, sj, spn in _SPINE_PAIRS:
            _add(f"{en}_{spn}", (elk1, elk2, _SPINE[sj], _SPINE[si]), (erk1, erk2, _SPINE[sj], _SPINE[si]))

    # ── C. Cross-level bands (skip-1 edge pairs + 2 spine points) ──
    # 4 edge-pairs × 15 spine_pairs = 60
    for e_idx in range(4):
        elk1, elk2 = _LEFT_EDGE[e_idx], _LEFT_EDGE[e_idx + 2]
        erk1, erk2 = _RIGHT_EDGE[e_idx], _RIGHT_EDGE[e_idx + 2]
        en = f"{_EDGE_SHORT[e_idx]}_{_EDGE_SHORT[e_idx + 2]}"
        for si, sj, spn in _SPINE_PAIRS:
            _add(f"{en}_{spn}", (elk1, elk2, _SPINE[sj], _SPINE[si]), (erk1, erk2, _SPINE[sj], _SPINE[si]))

    return pairs  # 90 + 75 + 60 = 225


_BILATERAL_CACHE: list[tuple[str, np.ndarray, np.ndarray]] | None = None


def _get_bilateral() -> list[tuple[str, np.ndarray, np.ndarray]]:
    global _BILATERAL_CACHE
    if _BILATERAL_CACHE is None:
        _BILATERAL_CACHE = _gen_bilateral_pairs()
    return _BILATERAL_CACHE


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
