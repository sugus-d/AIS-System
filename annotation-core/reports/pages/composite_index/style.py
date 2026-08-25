"""Composite Index 表格样式。"""

from __future__ import annotations

import pandas as pd

CV5_GOOD = 0.6  # cv5_r 良好阈值
CV5_PASS = 0.4  # cv5_r 通过阈值
GAP_WARN = 0.15  # gap 警告阈值


def fmt_table(df: pd.DataFrame) -> pd.DataFrame.style:
    s = df.style

    def _color_cv5(v: float) -> str:
        if v > CV5_GOOD:
            return "color: #16a34a; font-weight: bold"
        elif v > CV5_PASS:
            return "color: #ca8a04; font-weight: bold"
        return "color: #dc2626; font-weight: bold"

    if "cv5_r" in df.columns:
        s = s.map(_color_cv5, subset=["cv5_r"])

    def _color_gap(v: float) -> str:
        return "color: #dc2626" if v > GAP_WARN else ""

    if "gap" in df.columns:
        s = s.map(_color_gap, subset=["gap"])

    return s
