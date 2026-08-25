"""内部工具函数 — 数据加载、特征方案数据组装与共享筛选原语。

本模块从 modeling/schemes.py 抽取而来，用于打破 modeling ↔ features 的循环依赖。
筛选原语（_dedup_by_r/_anova_groups/_pass_r_or_anova）供特征方案
（scheme_morph_region_ci_35d）与特征合成器（features.synthesis）共享，避免复制。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, pearsonr

FEATURE_DIR = Path("results/extraction/features")

# ANOVA 检验所需的最少分组数
_MIN_ANOVA_GROUPS = 2


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


def _dedup_by_r(X: np.ndarray, y: np.ndarray, cols: list[str],
                corr_threshold: float) -> list[str]:
    """按 |r| 降序贪心去高相关（从 scheme_morph_region_ci_35d 迁移）。"""
    n = X.shape[1]
    if n <= 1:
        return cols
    r_vals = np.array([abs(pearsonr(X[:, i], y)[0]) for i in range(n)])
    order = np.argsort(-r_vals)
    corr = np.abs(np.corrcoef(X.T))
    keep: list[int] = []
    for idx in order:
        redundant = False
        for j in keep:
            if corr[idx, j] > corr_threshold:
                redundant = True
                break
        if not redundant:
            keep.append(idx)
    return [cols[i] for i in sorted(keep)]


def _anova_groups(x: np.ndarray, y4: np.ndarray) -> list[np.ndarray]:
    """按 Cobblestone 分级分组，每组至少 2 个样本（从 scheme_morph_region_ci_35d 迁移）。"""
    return [x[y4 == g] for g in range(4) if (y4 == g).sum() > 1]


def _pass_r_or_anova(r: float, groups: list[np.ndarray], r_threshold: float,
                     anova_p: float | None) -> bool:
    """|r| 超过阈值即保留; anova_p 非 None 时另接受 ANOVA p < anova_p。

    从 scheme_morph_region_ci_35d 迁移。
    """
    if abs(r) > r_threshold:
        return True
    if anova_p is None:
        return False
    return len(groups) >= _MIN_ANOVA_GROUPS and f_oneway(*groups)[1] < anova_p
