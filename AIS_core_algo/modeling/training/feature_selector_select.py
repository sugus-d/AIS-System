"""特征筛选策略 — 2700D 分组解析、morph/region 筛选。

从 feature_selector 拆分出的独立模块，依赖 feature_selector_scoring。
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from modeling.training.feature_selector_scoring import _dedup_keep_first, _hybrid_scores

# 2700D 的 8 种测量类型（列名匹配用）
MEASUREMENTS = [
    "height", "mean_curv", "gauss_curv", "roughness",
    "normal_angle", "normal_vector_cos",
]


def _parse_2700d_groups(columns: list[str]) -> dict[str, list[int]]:
    """将 2700D 列名按 (measurement, diff_method) 分组。

    Returns:
        {"height_dm": [0, 3, 15, ...], "height_pw": [1, 7, ...], ...}
    """
    groups: dict[str, list[int]] = {}
    for i, col in enumerate(columns):
        is_pw = "__pw" in col
        method = "pw" if is_pw else "dm"
        # 去掉 __pw 后缀以提取测量名
        base = col.replace("__pw", "")
        # 匹配测量类型
        matched = None
        for m in MEASUREMENTS:
            # 列名如 ax_p0_p1_height → 匹配 height, 不在更长的前缀里
            # 用 endswith 或完整单词匹配
            if base.endswith(f"_{m}") or (
                base.endswith(m) and (len(base) == len(m) or base[-len(m) - 1] == "_")
            ):
                # 检查不是更长测量的前缀 (normal_vector 不应匹配 normal_vector_cos)
                # 但 normal_vector 列就是 normal_vector，不会带多余后缀
                matched = m
                break
            # 对于 normal_vector_cos:
            # 列可能是 ax_p0_p1_normal_vector_cos
            # base 去掉 _cos/_sin 后匹配 normal_vector
            for suffix in ["_cos", "_sin"]:
                if base.endswith(suffix):
                    stem = base[: -len(suffix)]
                    if stem.endswith(f"_{m}"):
                        matched = m + suffix
                        break
            if matched:
                break
        if matched:
            g = f"{matched}|{method}"
            groups.setdefault(g, []).append(i)
    return groups


def _select_morph(X: np.ndarray, y: np.ndarray,
                  n_max: int = 10,
                  w_auc: float = 2.0,
                  corr_threshold: float = 0.85) -> np.ndarray:
    """混合评分排序 → 去高相关 → top n_max。

    Args:
        corr_threshold: 高相关判定阈值。
    """
    scores = _hybrid_scores(X, y, w_auc)
    order = np.argsort(-scores)
    X_sorted = X[:, order]
    # 去高相关（高分优先）；零方差列 → NaN 相关系数，视为不相关
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(X_sorted.T))
    corr = np.nan_to_num(corr, nan=0.0)
    keep = _dedup_keep_first(corr, corr_threshold)
    kept = order[np.array(keep)]
    if len(kept) > n_max:
        kept = kept[np.argsort(-scores[kept])[:n_max]]
    return np.sort(kept)


def _select_region(X: np.ndarray, y: np.ndarray,
                   n_target: int = 20,
                   w_auc: float = 2.0,
                   n_top_dedup: int = 500,
                   corr_threshold: float = 0.85,
                   coef_eps: float = 1e-6) -> np.ndarray:
    """混合评分排序 → 去高相关 → top N → LassoCV → top n_target。

    Args:
        corr_threshold: 高相关判定阈值。
        coef_eps: Lasso 系数视为非零的绝对值下限。
    """
    # Step 1: 全量混合评分 → 排序 → 去高相关
    scores = _hybrid_scores(X, y, w_auc)
    order = np.argsort(-scores)
    X_sorted = X[:, order]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs(np.corrcoef(X_sorted.T))
    corr = np.nan_to_num(corr, nan=0.0)  # 零方差列 → NaN，视为不相关
    keep = _dedup_keep_first(corr, corr_threshold)
    kept_orig = order[np.array(keep)]

    # Step 2: top N 候选池
    n_cand = min(n_top_dedup, len(kept_orig))
    cand = kept_orig[np.argsort(-scores[kept_orig])[:n_cand]]

    # Step 3: LassoCV 筛选
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X[:, cand])
    lcv = LassoCV(cv=min(5, max(2, len(y) // 5)),
                  max_iter=100000, random_state=42, n_jobs=1)
    lcv.fit(Xs, y)
    coef_abs = np.abs(lcv.coef_)
    nz = np.sum(coef_abs > coef_eps)

    if nz == 0:
        top = np.argsort(-scores[cand])[:n_target]
        return np.sort(cand[top])
    n_lasso = min(n_target * 2, int(nz))
    sel = [s for s in np.argsort(-coef_abs)[:n_lasso] if coef_abs[s] > coef_eps]
    if not sel:
        return np.sort(cand[:n_target])
    sel_cols = cand[np.array(sel)]

    # Step 4: 混合评分 top n_target
    if len(sel_cols) > n_target:
        sel_cols = sel_cols[np.argsort(-scores[sel_cols])[:n_target]]
    return np.sort(sel_cols)
