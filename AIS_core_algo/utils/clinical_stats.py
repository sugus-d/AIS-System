"""临床数据统计 — 论文 Table 1 的 BMI/曲线类型/严重度分布。

从 reports/pages/data_tables/data.py 迁出（reports 已删除），供 commands/export 使用。
"""

from __future__ import annotations

import json
from pathlib import Path

AIS_THRESHOLD = 10  # Cobb 角 ≥10° 判定为 AIS
BMI_UNDERWEIGHT_MAX = 18.5
BMI_NORMAL_MAX = 25.0
BMI_OVERWEIGHT_MAX = 30.0
MILD_THRESHOLD = 20
MODERATE_THRESHOLD = 40


def load_clinical_stats() -> dict:
    """Load clinical data and compute BMI categories, curve types, severity counts.

    Returns:
        dict 含 bmi/curve_types/severities 分布 + 各自 gender×AIS 明细。
    """
    exported = {d.name for d in Path("data/ground_truth").iterdir() if d.is_dir()}
    with open("data/form/clinical_data.json") as f:
        clinical = json.load(f)

    # Detailed categories for Table 1: category × gender × AIS status
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
            cat = (
                "Underweight"
                if bmi < BMI_UNDERWEIGHT_MAX
                else "Normal weight"
                if bmi < BMI_NORMAL_MAX
                else "Overweight"
                if bmi < BMI_OVERWEIGHT_MAX
                else "Obesity"
            )
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
            ct = "Double" if (has_t and has_l) else "Thoracic" if has_t else "Lumbar" if has_l else "Other"
            ct_detail.setdefault(ct, {}).setdefault(g, {}).setdefault(ais, 0)
            ct_detail[ct][g][ais] = ct_detail[ct][g].get(ais, 0) + 1

        # Severity
        if mc is not None:
            sev = (
                "Normal"
                if mc < AIS_THRESHOLD
                else "Mild"
                if mc < MILD_THRESHOLD
                else "Moderate"
                if mc < MODERATE_THRESHOLD
                else "Severe"
            )
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

    return {
        "bmi": sum_detail(bmi_detail),
        "bmi_detail": bmi_detail,
        "curve_types": sum_detail(ct_detail),
        "severities": sum_detail(sev_detail),
    }
