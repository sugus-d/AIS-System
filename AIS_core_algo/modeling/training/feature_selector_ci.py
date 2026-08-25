"""Per-fold CI 计算 — 分组 Lasso + dm/pw 2选1 + 去高相关 + top 3。

从 feature_selector 拆分出的独立模块，依赖 feature_selector_scoring。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from modeling._shared import CLINICAL
from modeling.training.feature_selector_scoring import _dedup_replace_better


def _compute_ci_per_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    group_ix: dict[str, list[int]],
    alpha: float = 1.5,
    w_auc: float = 2.0,
    min_features: int = 2,
    min_samples: int = 10,
    std_eps: float = 1e-8,
    min_groups: int = 2,
    corr_threshold: float = 0.85,
) -> tuple[np.ndarray, callable]:
    """per-fold CI: 分组 Lasso → dm/pw 2选1(混合评分) → 去高相关 → top 3。

    Args:
        min_features: 单组特征数下限，不足则跳过该组。
        min_samples: 样本数下限，不足则跳过该组。
        std_eps: CI 标准差下限，视为退化（无区分度）。
        min_groups: 有效 measure 组数下限，不足时返回全零。
        corr_threshold: 高相关判定阈值。

    Returns:
        ci_tr: (N_train, 3) 训练集 CI 值。
        ci_te_fn: (X_te) → (N_test, 3) 测试集 CI 计算函数。
    """
    y_bin = (y_tr > CLINICAL).astype(int)

    # Step 1: 12 组分别拟合 Lasso，计算 CI + 混合评分。
    # 每组存储 method、ci 值、scaler、lasso 系数、列索引、混合评分六项。
    all_ci: dict[str, tuple] = {}

    for gkey in sorted(group_ix.keys()):
        measure, method = gkey.split("|")
        gix = group_ix[gkey]
        Xg = X_tr[:, gix]
        if Xg.shape[1] < min_features or Xg.shape[0] < min_samples:
            continue
        scaler = StandardScaler()
        Xgs = scaler.fit_transform(Xg)
        lasso = Lasso(alpha=alpha, max_iter=50000, random_state=42)
        lasso.fit(Xgs, y_tr)
        ci = Xgs @ lasso.coef_
        if np.std(ci) > std_eps:
            r_val = abs(pearsonr(ci, y_tr)[0])
            ci_bin = (ci > CLINICAL).astype(int)
            auc_val = abs(roc_auc_score(y_bin, ci_bin) - 0.5) * 2
            h_val = r_val + w_auc * auc_val
        else:
            h_val = 0.0
        if measure not in all_ci or h_val > all_ci[measure][5]:
            all_ci[measure] = (method, ci, scaler, lasso.coef_, gix, h_val)

    if len(all_ci) < min_groups:
        ci_tr = np.zeros((X_tr.shape[0], 3))
        return ci_tr, lambda x: np.zeros((x.shape[0], 3))

    # Step 2: 去高相关（混合评分）
    ci_names = sorted(all_ci.keys())
    ci_vals = np.column_stack([all_ci[m][1] for m in ci_names])
    hs = np.array([all_ci[m][5] for m in ci_names])
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(ci_vals.T))
    corr = np.nan_to_num(corr, nan=0.0)  # 零方差列 → NaN 相关系数，视为不相关
    final = _dedup_replace_better(corr, hs, corr_threshold)
    kept = [ci_names[i] for i in sorted(final)]
    kept_hs = hs[[i for i in sorted(final)]]

    # Step 3: 混合评分 top 3
    k = min(3, len(kept))
    top = np.argsort(-kept_hs)[:k]

    ci_meta = []
    ci_parts = []
    for ti in top:
        m_name = kept[ti]
        method, ci, scaler, coefs, gix, _ = all_ci[m_name]
        ci_parts.append(ci)
        ci_meta.append((gix, scaler, coefs))

    ci_tr = np.column_stack(ci_parts) if len(ci_parts) > 1 else ci_parts[0]

    def _ci_te(X_te: np.ndarray) -> np.ndarray:
        te_parts = []
        for gix, scaler, coefs in ci_meta:
            Xgs = scaler.transform(X_te[:, gix])
            te_parts.append(Xgs @ coefs)
        return np.column_stack(te_parts) if te_parts else np.zeros((X_te.shape[0], k))

    return ci_tr, _ci_te
