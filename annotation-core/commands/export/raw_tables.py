#!/usr/bin/env python3
"""生成原始数据 CSV：体征参数（Table 2）+ 各表原始数据。

用法:
    uv run python -m commands.export.raw_tables
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from commands.export.config import (
    CLINICAL_FILE,
    ENSEMBLE_PRED_PATH,
    FEATURE_DIR,
    GROUND_TRUTH_DIR,
    TABLES_DIR,
)
from commands.export.tables import _compute_indices
from utils.logger import logger

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
_COBB_MILD = 10
_COBB_MODERATE = 20
_COBB_SEVERE = 40


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
    sp = [_parse_landmark(row.get(f"spine_P{i}(x,y,z)", "(0,0,0)")) for i in range(6)]

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


_BILATERAL_COLS = [
    "neck_root",
    "shoulder_transition",
    "scapular_peaks",
    "axilla",
    "waist",
    "waist_lower",
]


def _gt_to_csv_row(gt: dict, subject_id: str) -> dict:
    """把 ground_truth.json 转为 compute_cosmetic 可读的 CSV 行风格 dict。"""
    row: dict = {"subject_id": subject_id}
    for name in _BILATERAL_COLS:
        pair = gt.get(name)
        if not isinstance(pair, dict):
            continue
        for side in ("L", "R"):
            pt = pair.get(side)
            if pt is not None:
                x, y, z = (float(v) for v in pt)
                row[f"{name}_{side}(x,y,z)"] = f"({x},{y},{z})"
    spine = gt.get("spine_points")
    if isinstance(spine, list):
        for i, pt in enumerate(spine[:6]):
            x, y, z = (float(v) for v in pt)
            row[f"spine_P{i}(x,y,z)"] = f"({x},{y},{z})"
    return row


def generate_table2() -> pd.DataFrame:
    gt_paths = list(Path("results/ground-truth").glob("*/ground_truth.json"))
    raw_rows = []
    for p in gt_paths:
        gt = json.loads(p.read_text())
        raw_rows.append(_gt_to_csv_row(gt, p.parent.name))

    all_vals = []
    for row in raw_rows:
        vals = compute_cosmetic(row)
        vals["subject_id"] = row["subject_id"]
        all_vals.append(vals)

    df = pd.DataFrame(all_vals)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "table2_raw.csv", index=False)
    logger.info(f"Table 2 raw: {TABLES_DIR / 'table2_raw.csv'}")
    return df


def generate_raw_csvs(table1_df: pd.DataFrame, table3_raw: pd.DataFrame | None = None) -> None:
    # Table 1 raw
    with open(CLINICAL_FILE) as f:
        clin = json.load(f)
    exported = {d.name for d in GROUND_TRUTH_DIR.iterdir() if d.is_dir()}
    t1_rows = []
    for sid, cd in clin.items():
        if sid not in exported:
            continue
        h = cd.get("height_cm")
        w = cd.get("weight_kg")
        bmi = round(w / ((h / 100) ** 2), 1) if h and w else None
        mc = cd.get("max_cobb")
        ais = "AIS" if (mc is not None and mc >= _COBB_MILD) else "Non-AIS" if mc is not None else ""
        sev = (
            "Normal"
            if mc is None or mc < _COBB_MILD
            else "Mild"
            if mc < _COBB_MODERATE
            else "Moderate"
            if mc < _COBB_SEVERE
            else "Severe"
        )
        t1_rows.append(
            {
                "subject_id": sid,
                "gender": cd.get("gender", ""),
                "max_cobb": mc,
                "ais": ais,
                "severity": sev,
                "height_cm": h,
                "weight_kg": w,
                "bmi": bmi,
            }
        )
    pd.DataFrame(t1_rows).to_csv(TABLES_DIR / "table1_raw.csv", index=False)
    logger.info(f"Table 1 raw: {TABLES_DIR / 'table1_raw.csv'} ({len(t1_rows)} subjects)")

    # Table 3 raw
    if table3_raw is not None:
        table3_raw[["subject_id", "y", "ai", "curvature_index", "height_index", "nai", "ri"]].rename(
            columns={"y": "max_cobb"}
        ).to_csv(TABLES_DIR / "table3_raw.csv", index=False)
        logger.info(f"Table 3 raw: {TABLES_DIR / 'table3_raw.csv'}")
        table4 = table3_raw[["subject_id", "y", "ai", "curvature_index", "height_index", "nai", "ri"]].rename(
            columns={"y": "max_cobb"}
        )
        table4.to_csv(TABLES_DIR / "table4_raw.csv", index=False)
        logger.info(f"Table 4 raw: {TABLES_DIR / 'table4_raw.csv'}")

    # Table 5/6 raw — from ensemble predictions + clinical
    if ENSEMBLE_PRED_PATH.exists():
        pred_df = pd.read_csv(ENSEMBLE_PRED_PATH)
        df5 = pd.DataFrame(
            {
                "subject_id": pred_df["subject_id"],
                "max_cobb": pred_df["max_cobb_true"],
                "pred_max_cobb": pred_df["max_cobb_pred"],
                "class_true": pred_df["class_true"],
                "class_pred": pred_df["class_pred"],
            }
        )

        with open(CLINICAL_FILE) as f:
            clinical = json.load(f)
        curve_data = {}
        for sid, cd in clinical.items():
            curves = cd.get("curves", [])
            row = {}
            for i in range(4):
                cv = curves[i] if i < len(curves) else {}
                for k in ["cobb", "level", "direction", "apex"]:
                    row[f"curv{i + 1}_{k}"] = cv.get(k, "")
            curve_data[sid] = row
        for col in [f"curv{i + 1}_{k}" for i in range(4) for k in ["cobb", "level", "direction", "apex"]]:
            df5[col] = df5["subject_id"].map(lambda sid, c=col: curve_data.get(sid, {}).get(c))

        curve_map = {}
        for sid, cd in clinical.items():
            curves = cd.get("curves", [])
            primary, max_cb = None, -1
            for cv in curves:
                cb = cv.get("cobb") or 0
                if cb > max_cb:
                    max_cb, primary = cb, cv
            if primary and primary.get("level"):
                lv = primary["level"]
                curve_map[sid] = (
                    "Double"
                    if ("T" in lv and "L" in lv)
                    else "Thoracic"
                    if "T" in lv
                    else "Lumbar"
                    if "L" in lv
                    else "None"
                )
            else:
                curve_map[sid] = "None"
        df5["curve_type"] = df5["subject_id"].map(curve_map)
        df5.to_csv(TABLES_DIR / "table5_raw.csv", index=False)
        logger.info(f"Table 5 raw: {TABLES_DIR / 'table5_raw.csv'}")

        df5[["subject_id", "max_cobb", "pred_max_cobb", "class_true", "class_pred"]].to_csv(
            TABLES_DIR / "table6_raw.csv", index=False
        )
        logger.info(f"Table 6 raw: {TABLES_DIR / 'table6_raw.csv'}")


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    df_t2 = generate_table2()
    logger.info(f"\n  Cosmetic parameters (Auto): {len(df_t2)} subjects")
    for col in COSMETIC_PARAMS:
        vals = df_t2[col].dropna()
        logger.info(f"  {col}: {vals.mean():.2f}±{vals.std():.2f}  (n={len(vals)})")

    t1 = pd.read_csv(TABLES_DIR / "table1_demographics.csv")
    try:
        df_2700 = pd.read_csv(FEATURE_DIR / "features_2700d.csv").dropna(subset=["max_cobb"])
        df_idx = _compute_indices(df_2700)
    except Exception:
        df_idx = None

    generate_raw_csvs(t1, df_idx)
    logger.info("\n全部原始数据 CSV 已生成。")


if __name__ == "__main__":
    main()
