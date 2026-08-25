"""特征评估框架：评估每个特征与目标的相关性。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_MIN_VALID_SAMPLES = 5


def evaluate_features(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """评估每个特征与目标变量的相关性。

    对每个特征计算：
      - Pearson 相关系数 + p 值
      - Spearman 秩相关系数 + p 值
      - 方差（表示特征本身的离散程度）

    Args:
        X: 特征 DataFrame，每列为一个特征，每行为一个样本。
        y: 目标变量数组，长度与 X 的行数一致。

    Returns:
        DataFrame，每行为一个特征，包含以下列：
            feature_name, pearson_r, pearson_p, spearman_r, spearman_p, variance。
    """
    results: list[dict] = []
    y = np.asarray(y, dtype=np.float64)

    for col in X.columns:
        x = X[col].values.astype(np.float64)

        # 跳过全 NaN 或常数列
        finite_mask = np.isfinite(x) & np.isfinite(y)
        x_valid = x[finite_mask]
        y_valid = y[finite_mask]
        if len(x_valid) < _MIN_VALID_SAMPLES:
            continue

        # Pearson
        pr, pp = pearsonr(x_valid, y_valid)

        # Spearman
        sr, sp = spearmanr(x_valid, y_valid)

        results.append(
            {
                "feature": col,
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_r": float(sr),
                "spearman_p": float(sp),
                "variance": float(np.nanvar(x)),
            }
        )

    df = pd.DataFrame(results)
    # 按 |Pearson r| 降序排列
    df["abs_r"] = df["pearson_r"].abs()
    df = df.sort_values("abs_r", ascending=False).reset_index(drop=True)
    df = df.drop(columns=["abs_r"])
    return df
