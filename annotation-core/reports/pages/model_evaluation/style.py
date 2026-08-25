"""模型评估表格样式与阈值。"""

from __future__ import annotations

import pandas as pd

CLINICAL = 20.0
F1_THRESHOLD = 0.85
SENS_THRESHOLD = 0.6
SPEC_THRESHOLD = 0.55
RMSE_VALID_THRESHOLD = 999  # RMSE 低于该值才认为有有效预测，参与最小值高亮


def passes(r: dict) -> bool:
    return (
        r.get("f1", 0) >= F1_THRESHOLD
        and r.get("sens", 0) >= SENS_THRESHOLD
        and r.get("spec", 0) >= SPEC_THRESHOLD
    )


def style_table(df: pd.DataFrame) -> pd.DataFrame.style:
    s = df.style
    for col in ["F1", "Sens", "Spec"]:
        if col in df.columns:
            best = df[col].max()
            if best > 0:
                s = s.map(
                    lambda v, b=best: "background-color:#d4edda;font-weight:bold" if v == b else "",
                    subset=[col],
                )
    for col in ["RMSE"]:
        if col in df.columns:
            best = df[col].min()
            if best < RMSE_VALID_THRESHOLD:
                s = s.map(
                    lambda v, b=best: "background-color:#d4edda;font-weight:bold" if v == b else "",
                    subset=[col],
                )
    col_th = {"F1": F1_THRESHOLD, "Sens": SENS_THRESHOLD, "Spec": SPEC_THRESHOLD}
    for col, th in col_th.items():
        if col in df.columns:
            s = s.map(
                lambda v, t=th: "color:red;font-weight:bold"
                if isinstance(v, (int, float)) and v < t
                else "",
                subset=[col],
            )
    return s
