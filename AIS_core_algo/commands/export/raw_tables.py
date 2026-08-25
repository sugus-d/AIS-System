#!/usr/bin/env python3
"""生成原始数据 CSV：体征参数（Table 2）+ 各表原始数据。

用法:
    uv run python -m commands.export.raw_tables
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.body_params import (
    COBB_MILD,
    COBB_MODERATE,
    COBB_SEVERE,
    compute_cosmetic,
    COSMETIC_PARAMS,
    gt_to_csv_row,
)
from commands.export.tables import _compute_indices
from utils.logger import logger
from utils.paths import (
    CLINICAL_DATA,
    ENSEMBLE_PRED_PATH,
    EVAL_TABLES_DIR,
    FEATURES_DIR,
    GROUND_TRUTH_INPUT_DIR,
)


def generate_table2(out_dir: Path = EVAL_TABLES_DIR) -> pd.DataFrame:
    gt_paths = list(Path("results/ground-truth").glob("*/ground_truth.json"))
    raw_rows = []
    for p in gt_paths:
        gt = json.loads(p.read_text())
        raw_rows.append(gt_to_csv_row(gt, p.parent.name))

    all_vals = []
    for row in raw_rows:
        vals = compute_cosmetic(row)
        vals["subject_id"] = row["subject_id"]
        all_vals.append(vals)

    df = pd.DataFrame(all_vals)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "table2_raw.csv", index=False)
    logger.info(f"Table 2 raw: {out_dir / 'table2_raw.csv'}")
    return df


def generate_raw_csvs(
    table1_df: pd.DataFrame,
    table3_raw: pd.DataFrame | None = None,
    out_dir: Path = EVAL_TABLES_DIR,
    pred_path: Path = ENSEMBLE_PRED_PATH,
) -> None:
    """生成各表原始数据 CSV（Table 1/3/4/5/6 raw），数据源/输出目录可切换。"""
    target = out_dir
    target.mkdir(parents=True, exist_ok=True)
    # Table 1 raw
    with open(CLINICAL_DATA) as f:
        clin = json.load(f)
    exported = {d.name for d in GROUND_TRUTH_INPUT_DIR.iterdir() if d.is_dir()}
    t1_rows = []
    for sid, cd in clin.items():
        if sid not in exported:
            continue
        h = cd.get("height_cm")
        w = cd.get("weight_kg")
        bmi = round(w / ((h / 100) ** 2), 1) if h and w else None
        mc = cd.get("max_cobb")
        ais = "AIS" if (mc is not None and mc >= COBB_MILD) else "Non-AIS" if mc is not None else ""
        sev = (
            "Normal"
            if mc is None or mc < COBB_MILD
            else "Mild"
            if mc < COBB_MODERATE
            else "Moderate"
            if mc < COBB_SEVERE
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
    pd.DataFrame(t1_rows).to_csv(target / "table1_raw.csv", index=False)
    logger.info(f"Table 1 raw: {target / 'table1_raw.csv'} ({len(t1_rows)} subjects)")

    # Table 3 raw
    if table3_raw is not None:
        table3_raw[["subject_id", "y", "ai", "curvature_index", "height_index", "nai", "ri"]].rename(
            columns={"y": "max_cobb"}
        ).to_csv(target / "table3_raw.csv", index=False)
        logger.info(f"Table 3 raw: {target / 'table3_raw.csv'}")
        table4 = table3_raw[["subject_id", "y", "ai", "curvature_index", "height_index", "nai", "ri"]].rename(
            columns={"y": "max_cobb"}
        )
        table4.to_csv(target / "table4_raw.csv", index=False)
        logger.info(f"Table 4 raw: {target / 'table4_raw.csv'}")

    # Table 5/6 raw — from predictions + clinical
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        df5 = pd.DataFrame(
            {
                "subject_id": pred_df["subject_id"],
                "max_cobb": pred_df["max_cobb_true"],
                "pred_max_cobb": pred_df["max_cobb_pred"],
                "class_true": pred_df["class_true"],
                "class_pred": pred_df["class_pred"],
            }
        )

        with open(CLINICAL_DATA) as f:
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
        df5.to_csv(target / "table5_raw.csv", index=False)
        logger.info(f"Table 5 raw: {target / 'table5_raw.csv'}")

        df5[["subject_id", "max_cobb", "pred_max_cobb", "class_true", "class_pred"]].to_csv(
            target / "table6_raw.csv", index=False
        )
        logger.info(f"Table 6 raw: {target / 'table6_raw.csv'}")


def main(
    pred_csv: Path | None = None,
    region_csv: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    """原始数据 CSV（表 1/2/3/4/5/6 raw），数据源与输出目录可切换。

    Args:
        pred_csv: 预测 CSV；None 时用 v0.1.0 ``ENSEMBLE_PRED_PATH``。
        region_csv: 2700 维 region 特征 CSV；None 时用 v0.1.0 ``features_2700d.csv``。
        out_dir: 输出目录；None 时用 ``EVAL_TABLES_DIR``。
    """
    target = Path(out_dir) if out_dir else EVAL_TABLES_DIR
    pred_path = Path(pred_csv) if pred_csv else ENSEMBLE_PRED_PATH
    region_path = Path(region_csv) if region_csv else (FEATURES_DIR / "features_2700d.csv")
    target.mkdir(parents=True, exist_ok=True)

    df_t2 = generate_table2(target)
    logger.info(f"\n  Cosmetic parameters (Auto): {len(df_t2)} subjects")
    for col in COSMETIC_PARAMS:
        vals = df_t2[col].dropna()
        logger.info(f"  {col}: {vals.mean():.2f}±{vals.std():.2f}  (n={len(vals)})")

    t1 = pd.read_csv(target / "table1_demographics.csv")
    try:
        df_2700 = pd.read_csv(region_path).dropna(subset=["max_cobb"])
        df_idx = _compute_indices(df_2700)
    except Exception:
        df_idx = None

    generate_raw_csvs(t1, df_idx, target, pred_path)
    logger.info("\n全部原始数据 CSV 已生成。")


if __name__ == "__main__":
    main()
