"""Feature selection pipeline.

Filters morphology (56D) and region (2700D) features independently,
producing compact feature subsets for training.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from utils.logger import logger

MORPH_PREFIXES = [
    "neck_root_",
    "shoulder_transition_",
    "scapular_peaks_",
    "axilla_",
    "waist_",
    "waist_lower_",
    "spine_",
]
_MIN_VALID_SAMPLES = 10
"""单列参与相关性计算所需的最少有效样本数。"""
_MIN_SCORE_SAMPLES = 5
"""单列计算 hybrid score 所需的最少有效样本数。"""
_CLINICAL_COBB = 20
"""临床分级阈值（Cobb 角 20°）。"""
_CORR_SIGNAL = 0.2
"""Step 2 保留特征所需的 |r| 下界。"""
_AUC_SIGNAL = 0.10
"""Step 2 保留特征所需的 |AUC-0.5| 下界。"""
_COEF_EPS = 1e-6
"""LassoCV 系数视为零的阈值。"""
_FINAL_CORR_THRESHOLD = 0.85
"""最终去高相关的相关系数阈值。"""


def get_morphology_columns(df: pd.DataFrame) -> list[str]:
    """从 DataFrame 中选出 morphology 列（以 MORPH_PREFIXES 开头）。"""
    return [c for c in df.columns if any(c.startswith(p) for p in MORPH_PREFIXES)]


def step1_remove_redundant(cols: list[str]) -> list[str]:
    """Step 1: 删除计算冗余特征（_v_diff_ratio ×6, _len_ratio ×7）。

    _v_diff_ratio = vertical_diff / distance_3d, 信息已被 vertical_diff + distance_3d 覆盖。
    _len_ratio = segment_length / trunk_length, 在 spine 各段中携带相同分母噪音。
    """
    return [
        c
        for c in cols
        if not (c.endswith("_v_diff_ratio") or c.endswith("_len_ratio"))
    ]


def step2_filter_by_correlation(
    X: pd.DataFrame, y: np.ndarray, threshold: float = 0.2
) -> list[str]:
    """Step 2: 保留 Pearson |r| > threshold 的特征。

    只考虑有限值（finite）样本足够的列（>= 10 个有效值）。
    """
    cols = X.columns.tolist()
    Xv = X.values.astype(float)
    keep = []
    for i, c in enumerate(cols):
        valid = np.isfinite(Xv[:, i])
        if valid.sum() < _MIN_VALID_SAMPLES:
            continue
        r, _ = pearsonr(Xv[valid, i], y[valid])
        if abs(r) > threshold:
            keep.append(c)
    return keep


def step3_dedup_high_corr(
    X: pd.DataFrame, y: np.ndarray, corr_threshold: float = 0.85
) -> list[str]:
    """Step 3: 去除高度相关的冗余特征。

    计算 hybrid score = |r| + |AUC-0.5|*2 作为特征重要性评分。
    贪心保留：从高相关对中保留分数更高的特征。
    """
    cols = X.columns.tolist()
    Xv = X.values.astype(float)
    y_bin = (y > _CLINICAL_COBB).astype(int)

    # 计算每条特征的 hybrid score
    scores: dict[str, float] = {}
    for i, c in enumerate(cols):
        valid = np.isfinite(Xv[:, i])
        if valid.sum() < _MIN_SCORE_SAMPLES:
            scores[c] = 0.0
            continue
        r_val = abs(pearsonr(Xv[valid, i], y[valid])[0])
        auc_val = abs(roc_auc_score(y_bin[valid], Xv[valid, i]) - 0.5) * 2
        scores[c] = r_val + auc_val

    # 贪心去冗余
    corr_mat = np.abs(np.corrcoef(Xv.T))
    keep: list[int] = []
    for i in range(len(cols)):
        redundant = False
        for j in keep:
            if corr_mat[i, j] > corr_threshold:
                if scores[cols[i]] > scores[cols[j]]:
                    keep.remove(j)
                    keep.append(i)
                redundant = True
                break
        if not redundant:
            keep.append(i)
    return [cols[i] for i in keep]


def filter_morphology(df: pd.DataFrame, y: np.ndarray) -> list[str]:
    """三步筛选 Morphology 56 维 -> ~12 维。

    步骤:
      1. 删除计算冗余（_v_diff_ratio, _len_ratio）
      2. 保留 |r| > 0.2 的列
      3. 去高相关（corr > 0.85），保留 hybrid score 更高的
    """
    cols = get_morphology_columns(df)
    logger.info(f"  Morphology 原始: {len(cols)}")
    cols = step1_remove_redundant(cols)
    logger.info(f"  去计算冗余: {len(cols)}")
    cols = step2_filter_by_correlation(df[cols], y, 0.2)
    logger.info(f"  筛弱信号 |r|>0.2: {len(cols)}")
    cols = step3_dedup_high_corr(df[cols], y)
    logger.info(f"  去高相关: {len(cols)}")
    return cols


def filter_region(
    feature_dir: str = "results/extraction/features", n_target: int = 20
) -> list[str]:
    """三步筛选 Region 2700 维 -> ~20 维。

    Step 1: 使用已准备好的 2700d（已去低方差）。
    Step 2: (|r| > 0.2) AND (|AUC-0.5| > 0.10) 联合筛选。
    Step 3: LassoCV + 特征间去冗余，压缩到 ~20 维。
    """
    path_2250 = os.path.join(feature_dir, "features_2700d.parquet")
    if os.path.exists(path_2250):
        df = pd.read_parquet(path_2250)
    else:
        path_csv = os.path.join(feature_dir, "features_2700d.csv")
        if os.path.exists(path_csv):
            df = pd.read_csv(path_csv)
        else:
            path_csv = "results/extraction/features/features_2700d.csv"
            df = pd.read_csv(path_csv)

    df = df.dropna(subset=["max_cobb"])
    y = df["max_cobb"].values.astype(float)
    cols = [c for c in df.columns if c not in ("subject_id", "max_cobb")]
    X = df[cols].values.astype(float)
    y_bin = (y > _CLINICAL_COBB).astype(int)

    logger.info(f"  Region 原始: {len(cols)}")

    # Step 2: 双条件联合筛选
    keep: list[int] = []
    for i, _ in enumerate(cols):
        valid = np.isfinite(X[:, i])
        if valid.sum() < _MIN_VALID_SAMPLES:
            continue
        r_val = abs(pearsonr(X[valid, i], y[valid])[0])
        auc_val = abs(roc_auc_score(y_bin[valid], X[valid, i]) - 0.5) * 2
        if r_val > _CORR_SIGNAL and auc_val > _AUC_SIGNAL:
            keep.append(i)
    logger.info(f"  双条件筛选后: {len(keep)}")

    if len(keep) == 0:
        logger.warning("  WARNING: 无特征通过双条件筛选!")
        return []

    # Step 3: LassoCV
    X_filt = X[:, keep]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_filt)
    lcv = LassoCV(cv=5, max_iter=100000, random_state=42, n_jobs=1)
    lcv.fit(Xs, y)
    nz = np.sum(np.abs(lcv.coef_) > _COEF_EPS)
    logger.info(f"  LassoCV 非零系数: {nz}")

    # 按系数绝对值排序，取 top_k
    coef_abs = np.abs(lcv.coef_)
    top_k = min(n_target * 2, nz if nz > 0 else n_target)
    sel = np.argsort(-coef_abs)[:top_k]
    sel = [s for s in sel if coef_abs[s] > _COEF_EPS]

    final = [cols[keep[s]] for s in sel]
    logger.info(f"  LassoCV 选出: {len(final)}")

    # 最终去高相关
    if len(final) > n_target:
        X_final = df[final].values.astype(float)
        corr_f = np.abs(np.corrcoef(X_final.T))
        drop: set[str] = set()
        for i in range(len(final)):
            for j in range(i + 1, len(final)):
                if corr_f[i, j] > _FINAL_CORR_THRESHOLD:
                    r_i = abs(pearsonr(X_final[:, i], y)[0])
                    r_j = abs(pearsonr(X_final[:, j], y)[0])
                    drop.add(final[j] if r_i >= r_j else final[i])
        final = [c for c in final if c not in drop]
        logger.info(f"  最终去高相关后: {len(final)}")

    return final


def build_scheme_features(
    df_raw: pd.DataFrame,
    morph_cols: list[str],
    region_cols: list[str],
    ci_cols: list[str] | None = None,
) -> pd.DataFrame:
    """构建方案特征 DataFrame。

    Args:
        df_raw: 原始特征 DataFrame（含 basic, morphology 等）。
        morph_cols: 筛选后的 morphology 列名。
        region_cols: 筛选后的 region 列名。
        ci_cols: CI 列名（None = 不用 CI）。

    Returns:
        方案特征的 DataFrame（含 subject_id, max_cobb, 特征列）。
    """
    keep = [
        "subject_id",
        "max_cobb",
        "Height",
        "Weight",
        "BMI",
        "Gender",
        "Height_x_Weight",
    ]
    keep += [c for c in morph_cols if c in df_raw.columns]
    if ci_cols:
        keep += [c for c in ci_cols if c in df_raw.columns]
    if region_cols:
        # region 特征在单独的 2700d 文件中
        df_region = pd.read_csv("results/extraction/features/features_2700d.csv")
        dropna_cols = [c for c in region_cols if c in df_region.columns]
        df_region = df_region[["subject_id"] + dropna_cols]
        result = df_raw[keep].merge(df_region, on="subject_id")
        return result
    return df_raw[keep]
