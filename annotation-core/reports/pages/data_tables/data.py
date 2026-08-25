"""加载论文数据表的 CSV 和相关公式。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

TABLES_DIR = Path("results/eval/tables")
FORMULA_DIR = Path("results/formulas")

AIS_THRESHOLD = 10  # Cobb 角 ≥10° 判定为 AIS
BMI_UNDERWEIGHT_MAX = 18.5
BMI_NORMAL_MAX = 25.0
BMI_OVERWEIGHT_MAX = 30.0
MILD_THRESHOLD = 20
MODERATE_THRESHOLD = 40


@st.cache_data
def load_table(name: str) -> pd.DataFrame | None:
    path = TABLES_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_formula(name: str) -> dict | None:
    path = FORMULA_DIR / f"{name}_formula.json"
    if not path.exists():
        return None
    import json
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_clinical_stats() -> dict:
    """Load clinical data and compute BMI categories, curve types, severity counts."""
    import json
    from pathlib import Path

    exported = {d.name for d in Path("data/ground_truth").iterdir() if d.is_dir()}
    with open("data/form/clinical_data.json") as f:
        clinical = json.load(f)

    # Detailed categories for Table 1: category × gender × AIS status
    # Structure: nested dict of category, gender, AIS status, count
    bmi_detail: dict[str, dict[str, dict[str, int]]] = {}
    ct_detail: dict[str, dict[str, dict[str, int]]] = {}
    sev_detail: dict[str, dict[str, dict[str, int]]] = {}

    for sid, cd in clinical.items():
        if sid not in exported:
            continue
        g = cd.get("gender", "Unknown")
        mc = cd.get("max_cobb")
        ais = "AIS" if (mc is not None and mc >= AIS_THRESHOLD) else "Non-AIS" if mc is not None else "Unknown"

        # BMI
        h = cd.get("height_cm")
        w = cd.get("weight_kg")
        if h and w:
            bmi = w / ((h / 100) ** 2)
            cat = ("Underweight" if bmi < BMI_UNDERWEIGHT_MAX else "Normal weight"
                   if bmi < BMI_NORMAL_MAX else "Overweight" if bmi < BMI_OVERWEIGHT_MAX else "Obesity")
            bmi_detail.setdefault(cat, {}).setdefault(g, {}).setdefault(ais, 0)
            bmi_detail[cat][g][ais] = bmi_detail[cat][g].get(ais, 0) + 1

        # Curve type
        curves = cd.get("curves", [])
        primary, max_cb = None, -1
        for c in curves:
            cb = c.get("cobb") or 0
            if cb > max_cb:
                max_cb, primary = cb, c
        if primary and primary.get("level"):
            lv = primary["level"]
            has_t, has_l = "T" in lv, "L" in lv
            ct = ("Double" if (has_t and has_l) else "Thoracic" if has_t
                  else "Lumbar" if has_l else "Other")
            ct_detail.setdefault(ct, {}).setdefault(g, {}).setdefault(ais, 0)
            ct_detail[ct][g][ais] = ct_detail[ct][g].get(ais, 0) + 1

        # Severity
        if mc is not None:
            sev = ("Normal" if mc < AIS_THRESHOLD else "Mild" if mc < MILD_THRESHOLD
                   else "Moderate" if mc < MODERATE_THRESHOLD else "Severe")
            sev_detail.setdefault(sev, {}).setdefault(g, {}).setdefault(ais, 0)
            sev_detail[sev][g][ais] = sev_detail[sev][g].get(ais, 0) + 1

    # Compute overall counts from detail
    def sum_detail(d: dict) -> dict[str, int]:
        out: dict[str, int] = {}
        for cat, genders in d.items():
            total = 0
            for _, aises in genders.items():
                total += sum(aises.values())
            out[cat] = total
        return out

    bmi_cats = sum_detail(bmi_detail)
    curve_types = sum_detail(ct_detail)
    severities = sum_detail(sev_detail)

    return {"bmi": bmi_cats, "bmi_detail": bmi_detail,
            "curve_types": curve_types, "ct_detail": ct_detail,
            "severities": severities, "sev_detail": sev_detail}


def load_all() -> dict[str, pd.DataFrame | dict | None]:
    return {
        "table1": load_table("table1_demographics.csv"),
        "table2": load_table("table2_raw.csv"),
        "table3": load_table("table3_indices_by_severity.csv"),
        "table4": load_table("table4_correlation.csv"),
        "table5": load_table("table5_prediction.csv"),
        "table6": load_table("table6_classification.csv"),
        "ai_formula": load_formula("ai"),
        "curvature_formula": load_formula("curvature_index"),
        "height_formula": load_formula("height_index"),
        "nvi_formula": load_formula("nvi"),
        "ri_formula": load_formula("ri"),
        "inventory": load_table("data_inventory.csv"),
        "clinical": load_clinical_stats(),
    }
