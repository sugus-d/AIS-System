#!/usr/bin/env python3
"""数据摸底：统计 ground_truth 中所有 subject 的数据完整性。

用法:
    uv run python -m commands.export.inventory
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from utils.constants import classify_cobb
from utils.logger import logger
from utils.paths import CLINICAL_DATA, EVAL_TABLES_DIR, GROUND_TRUTH_INPUT_DIR, GROUND_TRUTH_OUTPUT_DIR


def classify_curve_type(curves: list[dict]) -> str:
    if not curves:
        return "None"
    max_cobb = -1
    primary = None
    for c in curves:
        cb = c.get("cobb") or 0
        if cb > max_cobb:
            max_cobb = cb
            primary = c
    if primary is None or not primary.get("level"):
        return "None"
    level = primary["level"]
    has_t = "T" in level
    has_l = "L" in level
    if has_t and not has_l:
        return "Thoracic"
    if has_l and not has_t:
        return "Lumbar"
    if has_t and has_l:
        return "Double"
    return "Other"


def main():
    with open(CLINICAL_DATA) as f:
        clinical = json.load(f)

    rows = []
    for sid in sorted(Path(GROUND_TRUTH_INPUT_DIR).iterdir()):
        if not sid.is_dir():
            continue
        sid_name = sid.name
        lm = GROUND_TRUTH_OUTPUT_DIR / sid_name / "ground_truth.json"
        mesh = sid / "roi.ply"

        has_lm = lm.exists()
        has_mesh = mesh.exists()
        has_clin = sid_name in clinical

        cd = clinical.get(sid_name, {})
        max_cobb = cd.get("max_cobb", None)
        gender = cd.get("gender", "")
        height = cd.get("height_cm", None)
        weight = cd.get("weight_kg", None)
        bmi = round(weight / ((height / 100) ** 2), 1) if height and weight else None
        curves = cd.get("curves", [])
        curve_type = classify_curve_type(curves)

        severity = "Normal" if max_cobb is None else classify_cobb(max_cobb)

        rows.append(
            {
                "subject_id": sid_name,
                "has_landmark": has_lm,
                "has_mesh": has_mesh,
                "has_clinical": has_clin,
                "max_cobb": max_cobb,
                "gender": gender,
                "height_cm": height,
                "weight_kg": weight,
                "bmi": bmi,
                "severity": severity,
                "curve_type": curve_type,
            }
        )

    df = pd.DataFrame(rows)
    logger.info(f"Total subjects: {len(df)}")
    logger.info(f"With landmark: {df['has_landmark'].sum()}")
    logger.info(f"With mesh:     {df['has_mesh'].sum()}")
    logger.info(f"With clinical: {df['has_clinical'].sum()}")
    logger.info("\nSeverity distribution:")
    logger.info(df["severity"].value_counts().sort_index().to_string())
    logger.info("\nCurve type distribution:")
    logger.info(df["curve_type"].value_counts().to_string())
    logger.info("\nGender distribution:")
    logger.info(df["gender"].value_counts().to_string())

    EVAL_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_TABLES_DIR / "data_inventory.csv"
    df.to_csv(out, index=False)
    logger.info(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
