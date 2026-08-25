"""Region-based mesh evaluation: landmark split lines, triangle classification, delta counting.

Compares algorithm-output meshes against ground-truth meshes by counting
triangles per anatomical region and reporting signed deltas.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
# 5 regions: neck / core / hem / side_L / side_R
REGION_NAMES = ["neck", "core", "hem", "side_L", "side_R"]
_N_REGIONS = len(REGION_NAMES)

_MIN_CSV_PARTS = 2  # CSV 坐标至少包含 x,y 两个分量

_LANDMARK_FIELDS = [
    "neck_root_L(x,y,z)",
    "neck_root_R(x,y,z)",
    "shoulder_transition_L(x,y,z)",
    "shoulder_transition_R(x,y,z)",
    "axilla_L(x,y,z)",
    "axilla_R(x,y,z)",
    "waist_L(x,y,z)",
    "waist_R(x,y,z)",
]

# ---------------------------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------------------------


def load_landmarks(csv_path: str) -> dict[str, float]:
    """Load GT landmarks from a per-subject CSV into a flat dict of floats.

    Returns keys like ``neck_root_L_y``, ``shoulder_transition_R_x``, etc.
    Returns an empty dict if the file is missing or unreadable.
    """
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)
    except Exception:
        return {}

    out: dict[str, float] = {}
    for field in _LANDMARK_FIELDS:
        raw = row.get(field, "").strip("()")
        if not raw:
            continue
        parts = raw.split(",")
        if len(parts) < _MIN_CSV_PARTS:
            continue
        short = field.split("(")[0]  # e.g. "neck_root_L"
        out[f"{short}_x"] = float(parts[0])
        out[f"{short}_y"] = float(parts[1])
        out[f"{short}_z"] = float(parts[2]) if len(parts) > _MIN_CSV_PARTS else 0.0
    return out


def compute_landmark_splits(
    landmarks: dict[str, float] | None,
    gt_v: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute region split lines from GT landmarks, with fallback to vertex extent.

    When landmarks are available, uses them for precise anatomical boundaries.
    Falls back to simple Y/X percentile ratios when landmarks are missing.

    Args:
        landmarks: Output of :func:`load_landmarks`, or ``None``.
        gt_v: (N, 3) GT vertex array — used for fallback / neck Y refinement.

    Returns:
        Dictionary with keys ``Y_neck``, ``Y_waist``, ``X_left``, ``X_right``.
    """
    has_lm = landmarks and len(landmarks) > 0

    # ── Y splits ──
    if has_lm:
        nr_y = np.mean(
            [
                landmarks.get("neck_root_L_y", 0),
                landmarks.get("neck_root_R_y", 0),
            ]
        )
        wa_y = np.mean(
            [
                landmarks.get("waist_L_y", 0),
                landmarks.get("waist_R_y", 0),
            ]
        )
        # neck: use neck_root Y directly; waist: use waist Y directly
        Y_neck = float(nr_y)
        Y_waist = float(wa_y)
    else:
        # fallback to ratio-based
        if gt_v is not None and len(gt_v) > 0:
            y_min = float(gt_v[:, 1].min())
            y_max = float(gt_v[:, 1].max())
            y_span = y_max - y_min
            Y_neck = y_max - y_span * 0.20
            Y_waist = y_min + y_span * 0.15
        else:
            Y_neck = 0.0
            Y_waist = -999.0

    # ── X splits ──
    if has_lm:
        # 颈根→肩转点的三等分点（更靠近颈根）作为脖子 X 边界
        nr_lx = landmarks.get("neck_root_L_x", -999)
        st_lx = landmarks.get("shoulder_transition_L_x", -999)
        nr_rx = landmarks.get("neck_root_R_x", 999)
        st_rx = landmarks.get("shoulder_transition_R_x", 999)
        X_left = float(nr_lx + (st_lx - nr_lx) / 3.0)
        X_right = float(nr_rx + (st_rx - nr_rx) / 3.0)
    else:
        if gt_v is not None and len(gt_v) > 0:
            x_min = float(gt_v[:, 0].min())
            x_max = float(gt_v[:, 0].max())
            x_span = x_max - x_min
            x_center = (x_min + x_max) / 2.0
            X_left = x_center - x_span * 0.35
            X_right = x_center + x_span * 0.35
        else:
            X_left = -999.0
            X_right = 999.0

    return {
        "Y_neck": Y_neck,
        "Y_waist": Y_waist,
        "X_left": X_left,
        "X_right": X_right,
    }


# ---------------------------------------------------------------------------
# Triangle classification
# ---------------------------------------------------------------------------


def classify_triangles(
    tri_centers: np.ndarray,
    splits: dict[str, float],
) -> np.ndarray:
    """Assign each triangle center to an anatomical region (0-4).

    Priority order: neck -> side_L -> side_R -> hem -> core.
    Unassigned → -1.

    Args:
        tri_centers: (N, 3) — X, Y, Z.
        splits: Split lines from :func:`compute_landmark_splits`.

    Returns:
        (N,) int32 labels.
    """
    if len(tri_centers) == 0:
        return np.empty((0,), dtype=np.int32)

    x, y = tri_centers[:, 0], tri_centers[:, 1]
    Yn, Yw, Xl, Xr = splits["Y_neck"], splits["Y_waist"], splits["X_left"], splits["X_right"]

    labels = np.full(len(tri_centers), -1, dtype=np.int32)

    # 1) neck: Y above neck_root AND X between shoulder transitions
    mask = (y > Yn) & (x >= Xl) & (x <= Xr)
    labels[mask] = 0

    # 2) side_L: X < shoulder_transition_L
    mask = (labels == -1) & (x < Xl)
    labels[mask] = 3

    # 3) side_R: X > shoulder_transition_R
    mask = (labels == -1) & (x > Xr)
    labels[mask] = 4

    # 4) hem: Y < waist_Y (not neck, not side)
    mask = (labels == -1) & (y < Yw)
    labels[mask] = 2

    # 5) core: everything else
    labels[labels == -1] = 1

    return labels


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def compute_region_deltri(
    algo_v: np.ndarray,
    algo_t: np.ndarray,
    gt_v: np.ndarray,
    gt_t: np.ndarray,
    splits: dict[str, float] | None = None,
    landmarks: dict[str, float] | None = None,
) -> list[dict]:
    """Compare triangle counts per region between algorithm and GT meshes.

    Args:
        algo_v: (N, 3) algo vertices.
        algo_t: (M, 3) algo triangle indices.
        gt_v: (K, 3) GT vertices.
        gt_t: (L, 3) GT triangle indices.
        splits: Pre-computed splits.  If ``None``, computed from *gt_v*
            (using landmarks if provided, otherwise fallback ratios).
        landmarks: Optional landmark dict for improved split computation.

    Returns:
        List of dicts with keys ``region``, ``algo``, ``gt``, ``delta``
        (signed: positive = algo has *more* than GT), ``delta_pct``.
    """
    if splits is None:
        splits = compute_landmark_splits(landmarks, gt_v)

    algo_centers = algo_v[algo_t].mean(axis=1)
    gt_centers = gt_v[gt_t].mean(axis=1)

    algo_labels = classify_triangles(algo_centers, splits)
    gt_labels = classify_triangles(gt_centers, splits)

    algo_counts = np.bincount(algo_labels[algo_labels >= 0], minlength=_N_REGIONS)
    gt_counts = np.bincount(gt_labels[gt_labels >= 0], minlength=_N_REGIONS)

    results: list[dict] = []
    for ri in range(_N_REGIONS):
        ac = int(algo_counts[ri])
        gc = int(gt_counts[ri])
        delta = ac - gc  # signed: + = algo more, - = algo less
        dp = round(delta / max(gc, 1) * 100.0, 1)
        results.append(
            {
                "region": REGION_NAMES[ri],
                "algo": ac,
                "gt": gc,
                "delta": delta,
                "delta_pct": dp,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def region_report_text(results: list[dict], subject: str = "") -> str:
    """Format region results as a human-readable table."""
    header = f"Region delta report{f' for {subject}' if subject else ''}"
    sep = "-" * len(header)
    lines = [header, sep]
    lines.append(f"{'Region':<8} {'Algo':>7} {'GT':>7} {'Δtri':>7} {'Δtri%':>8}")
    lines.append("-" * 40)
    for e in results:
        lines.append(f"{e['region']:<8} {e['algo']:>7d} {e['gt']:>7d} {e['delta']:+>7d} {e['delta_pct']:+>7.1f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Threshold management
# ---------------------------------------------------------------------------


def compute_thresholds(all_results: list[list[dict]], output_path: str) -> dict:
    """Compute per-region delta_pct percentiles across subjects.

    Saves JSON and returns the dict.
    """
    rv: dict[str, list[float]] = {}
    for subj in all_results:
        for e in subj:
            rv.setdefault(e["region"], []).append(e["delta_pct"])

    thresholds: dict[str, dict] = {}
    for name, vals in rv.items():
        arr = np.array(vals, dtype=np.float64)
        thresholds[name] = {
            "p50": round(float(np.percentile(arr, 50)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "p95": round(float(np.percentile(arr, 95)), 1),
            "n": len(arr),
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(thresholds, indent=2))
    return thresholds


def load_thresholds(path: str = "results/region_thresholds.json") -> dict | None:
    p = Path(path)
    return dict(json.loads(p.read_text())) if p.exists() else None
