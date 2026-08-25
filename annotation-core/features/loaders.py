"""特征数据加载 — 兼容旧 schemeB 特征（仅 reports/pages/model_evaluation 使用）。注意与 features/selectors/_loaders.py（方案加载实现）区分。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_scheme_b_features(feature_dir: str | Path = "results/extraction/features") -> pd.DataFrame:
    """加载旧 schemeB 特征数据（供模型评估报告使用）。

    Args:
        feature_dir: 特征目录，默认 results/features。

    Returns:
        特征表，删除 max_cobb 缺失行。
    """
    fdir = Path(feature_dir)
    parquet_path = fdir / "features_schemeB.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path).dropna(subset=["max_cobb"])
    return pd.read_csv(fdir / "features_schemeB.csv").dropna(subset=["max_cobb"])
