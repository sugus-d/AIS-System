"""Shared CI decomposition utilities.

Maps CI feature SHAP values back to categories by their measure type.
Used by both feature importance analysis and SHAP waterfall plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COMPOSITE_CSV = Path("results/modeling/composite/results_compressed.csv")

# CI feature → measure type mapping
CI_MEASURE = {
    "ci_normal_angle_dm":      "normal_angle",
    "ci_normal_vector_cos_pw": "normal_vector_cos",
    "ci_height_pw":            "height",
    "ci_mean_curv_dm":         "mean_curv",
}

_MEASURE_TO_CATEGORY = {
    "normal_angle":      "Normal Angle",
    "normal_vector":     "Normal Angle",
    "normal_vector_cos": "Normal Angle",
    "normal_vector_sin": "Normal Angle",
    "height":            "Height",
    "mean_curv":         "Curvature",
    "gauss_curv":        "Curvature",
    "roughness":         "Roughness",
}
_SHAP_EPS = 1e-6
"""视为零贡献的 SHAP 绝对值阈值。"""
_MIN_POLY_VERTICES = 3
"""多边形有效所需的最少顶点数。"""
_BAND_MAJORITY = 0.70
"""区域归属判定中主导 band 的占比阈值。"""


def load_ci_formulas() -> dict:
    """Load CI formula metadata from CSV.

    Returns:
        {group_name: {"measure": str, "n_feats": int, "r": float}} or empty dict.
    """
    if not COMPOSITE_CSV.exists():
        return {}
    df = pd.read_csv(COMPOSITE_CSV)
    formulas = {}
    for _, r in df.iterrows():
        if r["n_feats"] == 0:
            continue
        group = r["group"]
        method = "pw" if group.endswith("_pw") else "dm"
        measure = group.replace(f"_{method}", "")
        formulas[group] = {"measure": measure, "n_feats": r["n_feats"], "r": r["full_r"]}
    return formulas


def ci_category(ci_name: str) -> str:
    """Map a CI feature name to its merged category."""
    measure = CI_MEASURE.get(ci_name, "")
    return _MEASURE_TO_CATEGORY.get(measure, "Other")


def decompose_ci_shap(shap_values_row: np.ndarray, feature_names: list[str],
                      ci_indices: list[int], group_names: list[str]) -> dict[str, float]:
    """Decompose CI feature SHAP values to category-level contributions.

    Args:
        shap_values_row: 1D array of SHAP values for one subject.
        feature_names: list of feature names.
        ci_indices: indices of CI features in feature_names.
        group_names: ordered list of all category names.

    Returns:
        dict mapping category name → total SHAP contribution (including CI decomposed).
        Only includes categories with non-zero total contribution.
    """
    result = {g: 0.0 for g in group_names}

    # Aggregate direct region features
    for i, name in enumerate(feature_names):
        if i in ci_indices:
            continue
        g = _assign_group(name)
        if g in result:
            result[g] += shap_values_row[i]

    # Decompose CI features
    for ci_idx in ci_indices:
        ci_name = feature_names[ci_idx]
        cat = ci_category(ci_name)
        if cat in result:
            result[cat] += shap_values_row[ci_idx]

    return {k: v for k, v in result.items() if abs(v) > _SHAP_EPS}


# ── Feature grouping (shared with analyze script) ──

_MORPH_NAMES = {
    "spine_P0_P1_length", "spine_P0_P1_angle_vertical", "spine_P0_P1_lateral_deviation",
    "spine_P1_P2_length", "spine_P1_P2_angle_vertical",
    "spine_P2_P5_angle_vertical",
    "spine_P3_P4_len_ratio", "spine_P3_P4_lateral_deviation",
    "spine_curvature_P0P1_vs_P3P4",
    "scapular_peaks_slope_angle", "scapular_peaks_distance_3d",
    "scapular_peaks_anterior_diff", "scapular_peaks_vertical_diff",
    "waist_slope_angle", "waist_vertical_diff", "waist_distance_3d", "waist_anterior_diff",
    "neck_root_slope_angle", "neck_root_vertical_diff", "neck_root_distance_3d", "neck_root_anterior_diff",
    "axilla_vertical_diff", "axilla_distance_3d", "axilla_anterior_diff", "axilla_slope_angle",
    "shoulder_transition_vertical_diff", "shoulder_transition_slope_angle",
    "shoulder_transition_distance_3d", "shoulder_transition_anterior_diff",
    "width_waist_axilla_ratio", "trunk_length_ratio",
}


def _assign_group(name: str) -> str:
    if name in ("Height", "Weight", "BMI", "Gender", "Height_x_Weight"):
        return "Clinical"
    if name in _MORPH_NAMES:
        return "Morph"
    if any(x in name for x in ("normal_angle", "normal_vector_cos", "normal_vector", "normal_vector_sin")):
        return "Normal Angle"
    if "gauss_curv" in name or "mean_curv" in name:
        return "Curvature"
    if "_height" in name:
        return "Height"
    if "roughness" in name:
        return "Roughness"
    return "Other"


# ── Feature display label (shared by waterfall + importance plots) ──

_CI_DISPLAY = {
    "ci_normal_angle_dm": "Normal Angle Asymmetric Index",
    "ci_normal_vector_cos_pw": "Normal Angle Asymmetric Index",
    "ci_height_pw": "Height Asymmetric Index",
    "ci_mean_curv_dm": "Curvature Asymmetric Index",
    "normal_angle_pw": "Normal Angle Asymmetric Index",
    "normal_vector_cos_pw": "Normal Angle Asymmetric Index",
    "height_dm": "Height Asymmetric Index",
    "mean_curv_dm": "Curvature Asymmetric Index",
    "ci10_normal": "Normal-Targeted Asymmetric Index",
    "ci20_mild": "Mild-Targeted Asymmetric Index",
}

# 5 anatomical regions with v-band boundaries
_BANDS = [
    ("Shoulder", 1.75, 2.0),
    ("Scapula", 0, 1.75),
    ("Axilla", -1.5, 0),
    ("Waist", -3.0, -1.5),
    ("Pelvis", -4.0, -3.0),
]

_MEASURE_LABEL = {
    "normal_angle": "Normal Angle",
    "normal_vector_cos": "Normal Angle",
    "normal_vector": "Normal Angle",
    "gauss_curv": "Curvature",
    "mean_curv": "Curvature",
    "roughness": "Roughness",
    "height": "Height",
}

_MORPH_LABEL = {
    "spine_P0_P1_length": "Spine Length (Neck-Scapula)",
    "spine_P0_P1_angle_vertical": "Spine Vertical Angle (Neck-Scapula)",
    "spine_P0_P1_lateral_deviation": "Spine Lateral Deviation (Neck-Scapula)",
    "spine_P1_P2_length": "Spine Length (Scapula-Axilla)",
    "spine_P1_P2_angle_vertical": "Spine Vertical Angle (Scapula-Axilla)",
    "spine_P2_P5_angle_vertical": "Spine Vertical Angle (Axilla-Waist)",
    "spine_P3_P4_len_ratio": "Spine Length Ratio (Waist-Pelvis)",
    "spine_P3_P4_lateral_deviation": "Spine Lateral Deviation (Waist-Pelvis)",
    "spine_curvature_P0P1_vs_P3P4": "Spine Curvature Ratio (Neck vs Waist)",
    "neck_root_slope_angle": "Neck Slope Angle",
    "neck_root_vertical_diff": "Neck Vertical Diff",
    "shoulder_transition_vertical_diff": "Shoulder Vertical Diff",
    "shoulder_transition_slope_angle": "Shoulder Slope Angle",
    "scapular_peaks_slope_angle": "Scapular Slope Angle",
    "scapular_peaks_distance_3d": "Scapular Distance",
    "scapular_peaks_anterior_diff": "Scapular Anterior Diff",
    "axilla_vertical_diff": "Axilla Vertical Diff",
    "waist_slope_angle": "Waist Slope Angle",
    "waist_vertical_diff": "Waist Vertical Diff",
    "waist_distance_3d": "Waist Distance",
}

_CLINICAL_LABEL = {
    "Height": "Height", "Weight": "Weight", "BMI": "BMI",
    "Gender": "Gender", "Height_x_Weight": "H×W",
}


def _sutherland_hodgman(pts: np.ndarray, v_lo: float, v_hi: float) -> np.ndarray | None:
    """Clip convex polygon to v-band [v_lo, v_hi), return clipped polygon or None."""
    for bound, keep_above in [(v_lo, True), (v_hi, False)]:
        if len(pts) < _MIN_POLY_VERTICES:
            return None
        out = []
        for i in range(len(pts)):
            cur = pts[i]
            prev = pts[i - 1]
            cur_in = cur[1] >= bound if keep_above else cur[1] < bound
            prev_in = prev[1] >= bound if keep_above else prev[1] < bound
            # Edge enters band: add intersection point (unless it's the vertex itself)
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
    """Shoelace formula."""
    if pts is None or len(pts) < _MIN_POLY_VERTICES:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _build_region_area_cache() -> dict[str, dict[str, float]]:
    """Pre-compute UV area distribution for all 225 regions."""
    from features.extractors.asymmetry.regions import _get_pairs
    rng = np.random.default_rng(42)
    cache = {}
    for name, left, right in _get_pairs():
        total = _poly_area(left) + _poly_area(right)
        if total == 0:
            continue
        # Sample points inside left+right polygons
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
        # Ray casting for left polygon
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
        # Ray casting for right polygon
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
    """Get full area distribution of a feature's polygon across all 5 regions.

    Returns dict like {"Shoulder": 0.44, "Scapula": 0.56} for nr_p0_p1.
    Empty dict if the feature name doesn't match any region polygon.
    """
    global _REGION_AREA_CACHE
    if _REGION_AREA_CACHE is None:
        _REGION_AREA_CACHE = _build_region_area_cache()

    import re
    key = re.sub(r"_(height|mean_curv|gauss_curv|roughness|normal_angle|normal_vector_cos|normal_vector|normal_vector_sin)(__pw|_pw|_dm)?$", "", name)
    return _REGION_AREA_CACHE.get(key, {})


def _get_region_display(name: str) -> str:
    """Get region display string from feature name using area-based mapping."""
    global _REGION_AREA_CACHE
    if _REGION_AREA_CACHE is None:
        _REGION_AREA_CACHE = _build_region_area_cache()

    # Extract region key from feature name: strip measure suffix
    # e.g. "wa_wl_p0_p4_normal_angle__pw" → "wa_wl_p0_p4"
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


def feature_display_name(name: str) -> str:
    """Map raw feature name to readable label.

    Returns: "{region} {measure}" for region features,
             readable name for morph/clinical features,
             "Asymmetric Index" label for CI features.
    """
    # Clinical
    if name in _CLINICAL_LABEL:
        return _CLINICAL_LABEL[name]

    # CI
    if name in _CI_DISPLAY:
        return _CI_DISPLAY[name]

    # Morph
    if name in _MORPH_LABEL:
        return _MORPH_LABEL[name]

    # Region asymmetry feature
    fn_lower = name.lower()
    measure = "Morph"
    for pat, lbl in _MEASURE_LABEL.items():
        if pat in fn_lower:
            measure = lbl
            break

    region = _get_region_display(name)
    if region:
        return f"{region} {measure}"
    return measure
