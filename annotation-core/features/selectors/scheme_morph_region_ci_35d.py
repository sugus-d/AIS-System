"""morph_region_ci 特征筛选 — 35d/36d 变体参数化合并实现。

流程:
  Morph(31) → |r|>0.15 (36d 另加 ANOVA p<0.05) → |r|降序去高相关 → top 10
  Region(2700) → |r|>0.2 (36d 另加 ANOVA p<0.01) → 去高相关 → LassoCV → |coef|×|r| top 20
  CI(6 measure × 2 method = 12 组, 每组内条件筛选→去高相关→Lasso→线性组合)
    → 12 CI 去高相关 → pw/dm 择优 → top 6 (35d) / top 4 (36d)

变体差异: 35d 纯 |r| 单条件, CI top 6; 36d |r| OR ANOVA 双条件, CI top 4。
其余参数两变体完全相同。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, pearsonr
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

from features.selectors._utils import _SEVERITY_BINS, make_data_dict

MORPH_R = 0.15
MORPH_ANOVA_P = 0.05
MORPH_CORR = 0.85
MORPH_TOP = 10

REGION_R = 0.2
REGION_ANOVA_P = 0.01
REGION_CORR = 0.85
REGION_TOP = 20

CI_R = 0.2
CI_ANOVA_P = 0.05
CI_GROUP_CORR = 0.85
CI_ALPHA = {"mean_curv_dm": 2.0, "mean_curv_pw": 0.5}

CI_SELECT_CORR = 0.85
CI_TOP_35D = 6
CI_TOP_36D = 4
_MIN_ANOVA_GROUPS = 2
"""ANOVA 检验所需的最少分组数。"""
_COEF_EPS = 1e-6
"""Lasso/LassoCV 系数视为零的阈值。"""

MORPH_PREFIXES = [
    "neck_root_","shoulder_transition_","scapular_peaks_",
    "axilla_","waist_","waist_lower_","spine_",
]

CI_MEASURES = ["normal_angle", "normal_vector_cos", "height", "mean_curv", "gauss_curv", "roughness"]


def _dedup_by_r(X: np.ndarray, y: np.ndarray, cols: list[str],
                corr_threshold: float) -> list[str]:
    """按 |r| 降序贪心去高相关。"""
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
    """按 Cobblestone 分级分组，每组至少 2 个样本。"""
    return [x[y4 == g] for g in range(4) if (y4 == g).sum() > 1]


def _pass_r_or_anova(r: float, groups: list[np.ndarray], r_threshold: float,
                     anova_p: float | None) -> bool:
    """|r| 超过阈值即保留; anova_p 非 None 时另接受 ANOVA p < anova_p。"""
    if abs(r) > r_threshold:
        return True
    if anova_p is None:
        return False
    return len(groups) >= _MIN_ANOVA_GROUPS and f_oneway(*groups)[1] < anova_p


def _filter_morphology(df: pd.DataFrame, y: np.ndarray, *,
                       anova_p: float | None = MORPH_ANOVA_P) -> list[str]:
    """Morph: 去冗余 → 条件筛选(默认含 ANOVA) → |r|降序去高相关 → top 10"""
    cols = [c for c in df.columns if any(c.startswith(p) for p in MORPH_PREFIXES)]
    cols = [c for c in cols if not (c.endswith("_v_diff_ratio") or c.endswith("_len_ratio"))]

    y4 = np.digitize(y, _SEVERITY_BINS[1:-1]) if anova_p is not None else None
    keep: list[str] = []
    for c in cols:
        r, _ = pearsonr(df[c].values, y)
        groups = _anova_groups(df[c].values, y4) if y4 is not None else []
        if _pass_r_or_anova(r, groups, MORPH_R, anova_p):
            keep.append(c)

    if not keep:
        return []
    X = df[keep].values.astype(float)
    keep = _dedup_by_r(X, y, keep, MORPH_CORR)
    r_sorted = sorted(keep, key=lambda c: -abs(pearsonr(df[c].values, y)[0]))
    return r_sorted[:MORPH_TOP]


def _filter_region(df: pd.DataFrame, y: np.ndarray, *,
                   anova_p: float | None = REGION_ANOVA_P) -> list[str]:
    """Region: 条件筛选(默认含 ANOVA) → |r|降序去高相关 → LassoCV → |coef|×|r| top 20"""
    cols = [c for c in df.columns if c not in ("subject_id", "max_cobb")]
    X = df[cols].values.astype(float)
    y4 = np.digitize(y, _SEVERITY_BINS[1:-1]) if anova_p is not None else None

    keep_idx: list[int] = []
    for i in range(len(cols)):
        r, _ = pearsonr(X[:, i], y)
        groups = _anova_groups(X[:, i], y4) if y4 is not None else []
        if _pass_r_or_anova(r, groups, REGION_R, anova_p):
            keep_idx.append(i)
    if len(keep_idx) == 0:
        return []
    keep_cols = [cols[i] for i in keep_idx]
    keep_cols = _dedup_by_r(X[:, keep_idx], y, keep_cols, REGION_CORR)
    keep_idx = [cols.index(c) for c in keep_cols]

    X_filt = X[:, keep_idx]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_filt)
    lcv = LassoCV(cv=5, max_iter=100000, random_state=42, n_jobs=1)
    lcv.fit(Xs, y)
    nz = np.where(np.abs(lcv.coef_) > _COEF_EPS)[0]
    if len(nz) == 0:
        return []

    r_vals = np.array([abs(pearsonr(X[:, keep_idx[i]], y)[0]) for i in nz])
    scores = np.abs(lcv.coef_[nz]) * r_vals
    sel = nz[np.argsort(-scores)]
    return [keep_cols[i] for i in sel[:REGION_TOP]]


def _build_one_ci(df_r: pd.DataFrame, y: np.ndarray, measure: str, method: str,
                  alpha: float, *, anova_p: float | None = CI_ANOVA_P) -> np.ndarray | None:
    """构建单个 CI group: 条件筛选 → 去高相关 → Lasso → 线性组合"""
    suffix = f"_{measure}__pw" if method == "pw" else f"_{measure}"
    cols = [c for c in df_r.columns if c.endswith(suffix)
            and c not in ("subject_id", "max_cobb")]
    if len(cols) == 0:
        return None
    X_all = df_r[cols].values.astype(float)
    y4 = np.digitize(y, _SEVERITY_BINS[1:-1]) if anova_p is not None else None

    keep = []
    for i, _ in enumerate(cols):
        r, _ = pearsonr(X_all[:, i], y)
        groups = _anova_groups(X_all[:, i], y4) if y4 is not None else []
        if _pass_r_or_anova(r, groups, CI_R, anova_p):
            keep.append(i)
    if len(keep) == 0:
        return None
    keep_cols = [cols[i] for i in keep]
    keep_cols = _dedup_by_r(X_all[:, keep], y, keep_cols, CI_GROUP_CORR)
    keep_idx = [cols.index(c) for c in keep_cols]
    Xk = X_all[:, keep_idx]
    if Xk.shape[1] == 0:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xk)
    lasso = Lasso(alpha=alpha, max_iter=200000, random_state=42)
    lasso.fit(Xs, y)
    nz = np.where(np.abs(lasso.coef_) > _COEF_EPS)[0]
    if len(nz) == 0:
        return None

    nz_cols = [keep_cols[i] for i in nz]
    Xnz = df_r[nz_cols].values.astype(float)
    pm = np.nanmean(Xnz, axis=0)
    ps = np.maximum(np.nanstd(Xnz, axis=0), 1e-8)
    return (Xnz - pm) / ps @ lasso.coef_[nz]


def _build_ci_features(df_r: pd.DataFrame, y: np.ndarray, *,
                       anova_p: float | None = CI_ANOVA_P) -> pd.DataFrame:
    """构建 12 组 CI → 返回 subject_id + 12 CI 列。"""
    df_b = pd.read_csv(Path("results/extraction/features_extraction/back_v1") / "basic.csv").dropna(subset=["max_cobb"])
    result = {"subject_id": df_b["subject_id"].values}
    for measure in CI_MEASURES:
        for method in ["dm", "pw"]:
            group = f"{measure}_{method}"
            alpha = CI_ALPHA.get(group, 0.5)
            ci = _build_one_ci(df_r, y, measure, method, alpha, anova_p=anova_p)
            if ci is not None:
                result[group] = ci
    return pd.DataFrame(result)


def _filter_ci(df_ci: pd.DataFrame, y: np.ndarray, *,
               ci_top: int = CI_TOP_36D) -> list[str]:
    """CI 筛选：去高相关 → pw/dm 每组择优 → top ci_top"""
    ci_cols = [c for c in df_ci.columns if c != "subject_id"]
    if len(ci_cols) == 0:
        return []
    Xc = df_ci[ci_cols].values.astype(float)
    r_ci = np.array([abs(pearsonr(Xc[:, i], y)[0]) for i in range(Xc.shape[1])])

    keep = _dedup_by_r(Xc, y, ci_cols, CI_SELECT_CORR)
    # pw/dm 每组择优
    measure_map: dict[str, list[str]] = {}
    for c in keep:
        measure = c.rsplit("_", 1)[0] if c.endswith(("_pw", "_dm")) else c
        measure_map.setdefault(measure, []).append(c)
    selected = []
    for _measure, candidates in measure_map.items():
        best = max(candidates, key=lambda x: r_ci[ci_cols.index(x)])
        selected.append(best)
    return sorted(selected[:ci_top])


def _load(use_anova: bool, ci_top: int) -> dict:
    """按变体参数运行完整筛选管线，返回数据 dict。"""
    d = Path("results/extraction/features_extraction/back_v1")
    df_b = pd.read_csv(d / "basic.csv").dropna(subset=["max_cobb"])
    df_m = pd.read_csv(d / "morphology.csv").dropna(subset=["max_cobb"])
    df_r = pd.read_csv(d / "region_asymmetry.csv").dropna(subset=["max_cobb"])

    y = df_b["max_cobb"].values.astype(float)
    basic_cols = [c for c in df_b.columns if c not in ("subject_id", "max_cobb")]

    morph_final = _filter_morphology(df_m, y, anova_p=(MORPH_ANOVA_P if use_anova else None))
    region_final = _filter_region(df_r, y, anova_p=(REGION_ANOVA_P if use_anova else None))
    df_ci = _build_ci_features(df_r, y, anova_p=(CI_ANOVA_P if use_anova else None))
    ci_final = _filter_ci(df_ci, y, ci_top=ci_top)

    feature_names = basic_cols + morph_final + region_final + ci_final
    df_all = df_b[["subject_id", "max_cobb"] + basic_cols].merge(
        df_m[["subject_id"] + morph_final], on="subject_id", how="left")
    df_all = df_all.merge(df_r[["subject_id"] + region_final], on="subject_id", how="left")
    df_all = df_all.merge(df_ci[["subject_id"] + ci_final], on="subject_id", how="left")

    X = np.nan_to_num(df_all[feature_names].values.astype(float), nan=0.0)
    return make_data_dict(y, X, None, None, None)


def load() -> dict:
    """35d 变体: 纯 |r| 单条件, CI top 6。"""
    return _load(use_anova=False, ci_top=CI_TOP_35D)


def load_36d() -> dict:
    """36d 变体: |r| OR ANOVA 双条件, CI top 4。"""
    return _load(use_anova=True, ci_top=CI_TOP_36D)
