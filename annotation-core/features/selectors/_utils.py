"""内部工具函数 — 数据加载与特征方案数据组装。

本模块从 modeling/schemes.py 抽取而来，用于打破 modeling ↔ features 的循环依赖。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_DIR = Path("results/extraction/features")

# features 层本地严重度分箱（不 import modeling，避免反向依赖回归；与 metrics.SEVERITY_BINS 同值）
_SEVERITY_BINS = [0, 10, 20, 40, np.inf]


def load_parquet_or_csv(path_stem: str) -> pd.DataFrame:
    """加载 parquet（优先）或 CSV。"""
    p = FEATURE_DIR / f"{path_stem}.parquet"
    if p.exists():
        return pd.read_parquet(p).dropna(subset=["max_cobb"])
    return pd.read_csv(FEATURE_DIR / f"{path_stem}.csv").dropna(subset=["max_cobb"])


def to_float(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.nan_to_num(X[cols].values.astype(float), nan=0.0)


def make_data_dict(y: np.ndarray, X_basic: np.ndarray | None = None,
                   X_morph: np.ndarray | None = None,
                   X_region_full: np.ndarray | None = None,
                   region_col_names: list[str] | None = None) -> dict:
    return {
        "y": y,
        "X_basic": X_basic,
        "X_morph": X_morph,
        "X_region_full": X_region_full,
        "region_col_names": region_col_names,
    }
