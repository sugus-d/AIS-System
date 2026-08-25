"""Per-fold 嵌入式特征筛选 — 每折在训练集上选特征 + CI 合成。

自原 ml/cross_verification/ 迁入（逻辑原样，防信息泄漏），作为 Trainer 的可选组件
（TrainingConfig.feature_selector="per_fold" 启用）。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from modeling._shared import CLINICAL

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

# 2700D 的 8 种测量类型（列名匹配用）
MEASUREMENTS = [
    "height", "mean_curv", "gauss_curv", "roughness",
    "normal_angle", "normal_vector_cos",
]
DIFF_METHODS = ["dm", "pw"]


# ---------------------------------------------------------------------------
# 2700D 列分组解析
# ---------------------------------------------------------------------------


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
    # 去高相关（高分优先）
    corr = np.abs(np.corrcoef(X_sorted.T))
    keep = _dedup_keep_first(corr, corr_threshold)
    kept = order[np.array(keep)]
    if len(kept) > n_max:
        kept = kept[np.argsort(-scores[kept])[:n_max]]
    return np.sort(kept)


# ---------------------------------------------------------------------------
# 2. Region 筛选: 2700D → ~20
# ---------------------------------------------------------------------------


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
    corr = np.abs(np.corrcoef(X_sorted.T))
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
    corr = np.abs(np.corrcoef(ci_vals.T))
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




# ---------------------------------------------------------------------------
# PerFoldFeatureSelector — Trainer 集成组件
# ---------------------------------------------------------------------------


class PerFoldFeatureSelector:
    """per-fold 嵌入式特征筛选：basic 全保留 + morph top10 + region topN + CI 合成。

    在每折训练集上执行筛选（fit_transform），测试折用已拟合的索引/CI 变换（transform）。
    与旧 cross_verification 行为一致，供 Trainer 在 scaler 之前调用。
    """

    def __init__(self, w_auc: float = 2.0) -> None:
        self.w_auc = w_auc
        self._morph_ix: np.ndarray | None = None
        self._reg_ix: np.ndarray | None = None
        self._ci_fn = None

    def fit_transform(
        self,
        raw_blocks: dict[str, np.ndarray],
        y_tr: np.ndarray,
        region_column_names: list[str] | None = None,
    ) -> np.ndarray:
        """在训练折上筛选并返回拼接后的训练特征。

        Args:
            raw_blocks: 原始特征块 {"basic": (N,5), "morph": (N,~58), "region": (N,2700)}。
            y_tr: 训练折目标（原始空间，未变换）。
            region_column_names: 2700D 列名（启用 CI 合成时必填）。

        Returns:
            (N, n_sel) 拼接特征：basic + morph_sel + region_sel + ci。
        """
        parts: list[np.ndarray] = []
        basic = raw_blocks.get("basic")
        morph = raw_blocks.get("morph")
        region = raw_blocks.get("region")
        if basic is not None:
            parts.append(basic)
        if morph is not None:
            self._morph_ix = _select_morph(morph, y_tr, w_auc=self.w_auc)
            parts.append(morph[:, self._morph_ix])
        if region is not None:
            self._reg_ix = _select_region(region, y_tr, w_auc=self.w_auc)
            parts.append(region[:, self._reg_ix])
        if region is not None and region_column_names:
            groups = _parse_2700d_groups(region_column_names)
            ci_tr, self._ci_fn = _compute_ci_per_fold(region, y_tr, groups, w_auc=self.w_auc)
            if ci_tr.shape[1] > 0:
                parts.append(ci_tr)
        if not parts:
            raise ValueError("raw_blocks 至少需要一个特征块")
        return np.column_stack(parts) if len(parts) > 1 else parts[0]

    def transform(self, raw_blocks: dict[str, np.ndarray]) -> np.ndarray:
        """用已拟合的筛选索引/CI 变换测试折。"""
        parts: list[np.ndarray] = []
        basic = raw_blocks.get("basic")
        morph = raw_blocks.get("morph")
        region = raw_blocks.get("region")
        if basic is not None:
            parts.append(basic)
        if morph is not None and self._morph_ix is not None:
            parts.append(morph[:, self._morph_ix])
        if region is not None and self._reg_ix is not None:
            parts.append(region[:, self._reg_ix])
        if region is not None and self._ci_fn is not None:
            parts.append(self._ci_fn(region))
        if not parts:
            raise ValueError("raw_blocks 至少需要一个特征块")
        return np.column_stack(parts) if len(parts) > 1 else parts[0]
