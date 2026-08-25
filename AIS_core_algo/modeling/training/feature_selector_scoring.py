"""特征评分与去重工具 — 混合评分、高相关去重策略。

从 feature_selector 拆分出的独立模块，供 select/ci 子模块调用。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

from modeling._shared import CLINICAL


def _hybrid_scores(X: np.ndarray, y: np.ndarray, w_auc: float = 2.0,
                   clinical: float = CLINICAL,
                   min_valid: int = 5) -> np.ndarray:
    """混合评分: |Pearson r| + w_auc × |AUROC-0.5|×2。

    同时考虑连续相关（r）和 20° 二分类分离能力（AUC）。

    Args:
        min_valid: 有效样本数下限，低于该值的特征列跳过。
    """
    y_bin = (y > clinical).astype(int)
    scores = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        valid = np.isfinite(X[:, i])
        if valid.sum() < min_valid:
            continue
        r = abs(pearsonr(X[valid, i], y[valid])[0])
        auc = abs(roc_auc_score(y_bin[valid], X[valid, i]) - 0.5) * 2
        scores[i] = r + w_auc * auc
    return scores


def _dedup_by_corr(X: np.ndarray, y: np.ndarray,
                   corr_threshold: float = 0.85,
                   scores: np.ndarray | None = None) -> np.ndarray:
    """贪心去高相关：从高相关对中保留评分更高的特征。

    Args:
        X: 特征矩阵 (n, m)。
        y: 目标值 (n,)。
        corr_threshold: 高相关判定阈值。
        scores: 预计算的评分数组 (m,)。None 时用 |Pearson r|。
    """
    n = X.shape[1]
    if n <= 1:
        return np.arange(n)
    if scores is None:
        scores = np.array([abs(pearsonr(X[:, i], y)[0]) for i in range(n)])
    corr = np.abs(np.corrcoef(X.T))
    final = []
    for i in range(n):
        redundant = False
        for j in final:
            if corr[i, j] > corr_threshold:
                if scores[i] > scores[j]:
                    final.remove(j)
                    final.append(i)
                redundant = True
                break
        if not redundant:
            final.append(i)
    return np.array(sorted(final), dtype=int)


def _dedup_keep_first(corr: np.ndarray, threshold: float = 0.85) -> list[int]:
    """按序遍历去高相关：保留每个未被已有成员压制的特征。

    Args:
        corr: 特征相关矩阵（按当前遍历顺序，corr[i, j] 为 i 与已保留 j 的相关）。
        threshold: 高相关判定阈值。

    Returns:
        保留的索引列表（int）。
    """
    keep: list[int] = []
    for i in range(corr.shape[0]):
        redundant = False
        for j in keep:
            if corr[i, j] > threshold:
                redundant = True
                break
        if not redundant:
            keep.append(i)
    return keep


def _dedup_replace_better(corr: np.ndarray, scores: np.ndarray,
                          threshold: float = 0.85) -> list[int]:
    """去高相关：冗余时若新特征评分更高则替换已有成员。

    Args:
        corr: 特征相关矩阵（按当前遍历顺序）。
        scores: 每个特征的评分 (n,)。
        threshold: 高相关判定阈值。

    Returns:
        保留的索引列表（int）。
    """
    keep: list[int] = []
    for i in range(corr.shape[0]):
        redundant = False
        for j in keep:
            if corr[i, j] > threshold:
                if scores[i] > scores[j]:
                    keep.remove(j)
                    keep.append(i)
                redundant = True
                break
        if not redundant:
            keep.append(i)
    return keep
