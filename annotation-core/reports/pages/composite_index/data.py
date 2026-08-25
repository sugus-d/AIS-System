"""Composite Index 数据加载。"""

from __future__ import annotations

import csv
from pathlib import Path

import streamlit as st

COMPOSITE_DIR = Path("results/modeling/composite")


@st.cache_data
def load_data(mode: str) -> list[dict]:
    filename = "results_compressed.csv" if mode == "compressed" else "results.csv"
    path = COMPOSITE_DIR / filename
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "group": row["group"],
                "n_feats": int(row["n_feats"]),
                "cv5_r": float(row["cv5_r"]),
                "full_r": float(row["full_r"]),
                "gap": float(row["gap"]),
                "formula": row.get("formula", ""),
            })
    return rows
