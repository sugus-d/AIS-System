"""特征方案 loader — SELECTION_REGISTRY 的加载实现（纯函数，无注册表逻辑）。

从 results/features_* 读 CSV 构建 make_data_dict 分块数据。
注册表定义在 :mod:`features.selectors.schemes`。

拆分说明：CI 相关函数移至 _loaders_ci.py，canonical 相关函数移至 _loaders_canonical.py。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.selectors._loaders_canonical import _load_canonical_44d, _load_canonical_union_64d  # noqa: F401
from features.selectors._loaders_ci import (  # noqa: F401
    _load_dual_ci,
    _load_morph_region_ci_27d,
)
from features.selectors._utils import make_data_dict


def _load_selection(version: str) -> dict:
    """加载特征工程阶段筛选后的数据，返回 X_basic 作为全量特征矩阵。"""
    d = Path("results/extraction/features_selection") / version
    df_b = pd.read_csv(Path("results/extraction/features_extraction") / version / "basic.csv").dropna(subset=["max_cobb"])
    df_m = pd.read_csv(d / "morphology.csv").dropna(subset=["max_cobb"])
    df_r = pd.read_csv(d / "region_asymmetry.csv").dropna(subset=["max_cobb"])
    df_c = pd.read_csv(d / "ci.csv")

    y = df_b["max_cobb"].values.astype(float)
    basic_cols = [c for c in df_b.columns if c not in ("subject_id", "max_cobb")]
    morph_cols = [c for c in df_m.columns if c not in ("subject_id", "max_cobb")]
    region_cols = [c for c in df_r.columns if c not in ("subject_id", "max_cobb")]
    ci_cols = [c for c in df_c.columns if c != "subject_id"]

    df_all = df_b[["subject_id", "max_cobb"] + basic_cols].merge(
        df_m[["subject_id"] + morph_cols], on="subject_id", how="left")
    df_all = df_all.merge(df_r[["subject_id"] + region_cols], on="subject_id", how="left")
    df_all = df_all.merge(df_c, on="subject_id", how="left")

    feature_names = basic_cols + morph_cols + region_cols + ci_cols
    X = np.nan_to_num(df_all[feature_names].values.astype(float), nan=0.0)
    return make_data_dict(y, X, None, None, None)
