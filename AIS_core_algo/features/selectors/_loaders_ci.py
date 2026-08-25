"""CI 特征加载器 — _compute_ci_features / _load_dual_ci / _load_morph_region_ci_27d。

从 _loaders.py 拆分而来，保持等价。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── 特征筛选阈值（各 scheme 共用的超参，历史调优值，勿随意改动） ──
_R_MORPH = 0.15              # morphology |Pearson r| 保留阈值
_R_REGION = 0.2              # region |Pearson r| 保留阈值
_P_VALUE = 0.05              # ANOVA F 检验 p 值阈值
_COEF_EPS = 1e-6             # Lasso 系数视为非零的绝对值下限
_TOP_MORPH = 10              # morphology 保留 top N
_MIN_ANOVA_GROUPS = 2        # ANOVA 至少需要的等级组数
_MIN_GROUP_FEATURES = 2      # 每组至少保留的特征数
_COBB_LIGHT = 10             # Cobb 10°（轻度阈值，用于二分类目标）
_COBB_CLINICAL = 20          # Cobb 20°（临床阈值，用于二分类目标）


def _load_dual_ci(version: str = "v0.1.0") -> dict:
    """加载 v0.1.0 方案（构成语义名）：10° CI + 20° CI。

    Args:
        version: 特征提取目录名（v0.1.0=算法 ROI，v1.0.0=人工 ROI）。
    """
    import warnings

    from scipy.stats import f_oneway, pearsonr
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")
    import logging

    logging.disable(logging.CRITICAL)
    import numpy as np
    from sklearn.linear_model import LassoCV

    from features.selectors._utils import _dedup_by_r
    from features.selectors._utils import make_data_dict as _make_data_dict
    from features.selectors.scheme_morph_region_ci_35d import (
        _build_ci_features,
        _filter_ci,
        _filter_morphology,
    )
    from utils.constants import SEVERITY_BINS

    d = Path("results/extraction/features_extraction") / version
    df_b = pd.read_csv(d / "basic.csv").dropna(subset=["max_cobb"])
    df_m = pd.read_csv(d / "morphology.csv").dropna(subset=["max_cobb"])
    df_r = pd.read_csv(d / "region_asymmetry.csv").dropna(subset=["max_cobb"])
    y = df_b["max_cobb"].values.astype(float)
    y4 = np.digitize(y, SEVERITY_BINS[1:-1])
    basic_cols = [c for c in df_b.columns if c not in ("subject_id", "max_cobb")]

    morph_final = _filter_morphology(df_m, y)

    # Region: 放松 r>0.15 → dedup → LassoCV → top25
    rc = [c for c in df_r.columns if c not in ("subject_id", "max_cobb")]
    Xr = df_r[rc].values.astype(float)
    k_idx = [
        i
        for i in range(len(rc))
        if abs(pearsonr(Xr[:, i], y)[0]) > _R_MORPH
        or (
            len([Xr[y4 == g, i] for g in range(4) if (y4 == g).sum() > 1]) >= _MIN_ANOVA_GROUPS
            and f_oneway(*[Xr[y4 == g, i] for g in range(4) if (y4 == g).sum() > 1])[1] < _P_VALUE
        )
    ]
    kc = _dedup_by_r(Xr[:, k_idx], y, [rc[i] for i in k_idx], 0.85)
    ki = [rc.index(c) for c in kc]
    Xf = Xr[:, ki]
    lcv = LassoCV(cv=5, max_iter=100000, random_state=42, n_jobs=1)
    lcv.fit(StandardScaler().fit_transform(Xf), y)
    nz = np.where(np.abs(lcv.coef_) > _COEF_EPS)[0]
    rv = np.array([abs(pearsonr(Xf[:, i], y)[0]) for i in nz])
    sel = nz[np.argsort(-np.abs(lcv.coef_[nz]) * rv)]
    region_final = [kc[i] for i in sel[:25]]

    df_ci = _build_ci_features(df_r, y)
    ci_final = _filter_ci(df_ci, y)

    # 候选池
    all_cols = []
    for m in ["normal_angle", "normal_vector_cos", "height", "mean_curv", "gauss_curv", "roughness"]:
        for mt in ["dm", "pw"]:
            sfx = f"_{m}__pw" if mt == "pw" else f"_{m}"
            all_cols.extend(c for c in df_r.columns if c.endswith(sfx) and c not in ("subject_id", "max_cobb"))
    Xr_all = df_r[all_cols].values.astype(float)

    def _build_ci_for_target(target: np.ndarray, C: float = 0.2,
                             thr: float = 0.1) -> np.ndarray:
        """当场拟合 ci10/ci20 合成参数并算值（复用 CiTargetSynthesizer，0.85 去高相关）。"""
        from features.synthesis import CiTargetSynthesizer

        synth = CiTargetSynthesizer().fit(all_cols, Xr_all, target, C, thr)
        return synth.transform_ndarray(Xr_all, all_cols)

    yb10 = (y < _COBB_LIGHT).astype(float)
    yb20 = (y > _COBB_CLINICAL).astype(float)
    ci10 = _build_ci_for_target(yb10, C=0.1, thr=0.05)
    ci20 = _build_ci_for_target(yb20, C=0.2, thr=0.1)

    feature_names = basic_cols + morph_final + region_final + ci_final + ["ci10_normal", "ci20_mild"]
    df_all = df_b[["subject_id", "max_cobb"] + basic_cols].merge(
        df_m[["subject_id"] + morph_final], on="subject_id", how="left"
    )
    df_all = df_all.merge(df_r[["subject_id"] + region_final], on="subject_id", how="left")
    df_all = df_all.merge(df_ci[["subject_id"] + ci_final], on="subject_id", how="left")
    df_all["ci10_normal"] = ci10
    df_all["ci20_mild"] = ci20
    X = np.nan_to_num(df_all[feature_names].values.astype(float), nan=0.0)
    result = _make_data_dict(y, X, None, None, None)
    result["feature_names"] = feature_names
    result["ci_feature_names"] = ci_final + ["ci10_normal", "ci20_mild"]
    return result


def _load_morph_region_ci_27d() -> dict:
    """morph_region_ci_27d: CI 占用剔除→剩余池 LassoCV 非零全量（42D）"""
    import logging
    import warnings

    from scipy.stats import f_oneway, pearsonr
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    import numpy as np

    from features.selectors._utils import _dedup_by_r
    from features.selectors._utils import make_data_dict as _make_data_dict
    from features.selectors.scheme_morph_region_ci_35d import (
        _build_ci_features,
        _filter_ci,
        _filter_morphology,
    )
    from utils.constants import SEVERITY_BINS

    d = Path("results/extraction/features_extraction/v0.1.0")
    df_b = pd.read_csv(d / "basic.csv").dropna(subset=["max_cobb"])
    df_m = pd.read_csv(d / "morphology.csv").dropna(subset=["max_cobb"])
    df_r = pd.read_csv(d / "region_asymmetry.csv").dropna(subset=["max_cobb"])
    y = df_b["max_cobb"].values.astype(float)
    y4 = np.digitize(y, SEVERITY_BINS[1:-1])
    basic_cols = [c for c in df_b.columns if c not in ("subject_id", "max_cobb")]
    morph_final = _filter_morphology(df_m, y)

    # Step 1: 构建全部 CI
    df_ci = _build_ci_features(df_r, y)
    ci4_names = _filter_ci(df_ci, y)

    all_cols = []
    for m in ["normal_angle", "normal_vector_cos", "height", "mean_curv", "gauss_curv", "roughness"]:
        for mt in ["dm", "pw"]:
            sfx = f"_{m}__pw" if mt == "pw" else f"_{m}"
            all_cols.extend(c for c in df_r.columns if c.endswith(sfx) and c not in ("subject_id", "max_cobb"))
    Xa = df_r[all_cols].values.astype(float)

    # 标准 CI 的 Lasso 非零特征
    ci_used = set()
    for measure in ["normal_angle", "normal_vector_cos", "height", "mean_curv", "gauss_curv", "roughness"]:
        for method in ["dm", "pw"]:
            sfx = f"_{measure}__pw" if method == "pw" else f"_{measure}"
            cols = [c for c in df_r.columns if c.endswith(sfx) and c not in ("subject_id", "max_cobb")]
            if len(cols) < _MIN_GROUP_FEATURES:
                continue
            Xc = df_r[cols].values.astype(float)
            keep_idx = [i for i in range(len(cols)) if abs(pearsonr(Xc[:, i], y)[0]) > _R_REGION]
            if len(keep_idx) < _MIN_GROUP_FEATURES:
                continue
            kc = [cols[i] for i in keep_idx]
            kc = _dedup_by_r(Xc[:, keep_idx], y, kc, 0.85)
            ki = [cols.index(c) for c in kc]
            lasso = Lasso(alpha=0.5, max_iter=200000, random_state=42)
            lasso.fit(StandardScaler().fit_transform(Xc[:, ki]), y)
            nz = np.where(np.abs(lasso.coef_) > _COEF_EPS)[0]
            for i in nz:
                ci_used.add(kc[i])

    def _ci(target: np.ndarray, C: float, thr: float) -> tuple[set[str], np.ndarray]:
        """当场拟合 ci10/ci20 并返回 (选中特征集合, 合成值)（复用 CiTargetSynthesizer）。"""
        from features.synthesis import CiTargetSynthesizer

        synth = CiTargetSynthesizer().fit(all_cols, Xa, target, C, thr)
        params = synth.to_params()
        feats = {params["columns"][i] for i in params["nz"]}
        return feats, synth.transform_ndarray(Xa, all_cols)

    ci10_feats, ci10 = _ci((y < _COBB_LIGHT).astype(float), 0.1, 0.05)
    ci20_feats, ci20 = _ci((y > _COBB_CLINICAL).astype(float), 0.2, 0.1)
    ci_used |= ci10_feats | ci20_feats

    # Step 2: 在剔除 CI 特征的池中找 region
    rc_pool = [c for c in df_r.columns if c not in ("subject_id", "max_cobb") and c not in ci_used]
    Xr = df_r[rc_pool].values.astype(float)
    k_idx = [
        i
        for i in range(len(rc_pool))
        if abs(pearsonr(Xr[:, i], y)[0]) > _R_MORPH
        or (
            len([Xr[y4 == g, i] for g in range(4) if (y4 == g).sum() > 1]) >= _MIN_ANOVA_GROUPS
            and f_oneway(*[Xr[y4 == g, i] for g in range(4) if (y4 == g).sum() > 1])[1] < _P_VALUE
        )
    ]
    kc = _dedup_by_r(Xr[:, k_idx], y, [rc_pool[i] for i in k_idx], 0.85)
    ki = [rc_pool.index(c) for c in kc]
    Xf = Xr[:, ki]
    lcv = LassoCV(cv=5, max_iter=100000, random_state=42, n_jobs=1)
    lcv.fit(StandardScaler().fit_transform(Xf), y)
    nz = np.where(np.abs(lcv.coef_) > _COEF_EPS)[0]
    rv = np.array([abs(pearsonr(Xf[:, i], y)[0]) for i in nz])
    # region 保留 LassoCV 全部非零特征（截断到 top6 实测性能更差，保留全量）
    region_final = [kc[i] for i in nz[np.argsort(-np.abs(lcv.coef_[nz]) * rv)]]

    feature_names = basic_cols + morph_final + region_final + ci4_names + ["ci10_normal", "ci20_mild"]
    df_all = df_b[["subject_id", "max_cobb"] + basic_cols].merge(
        df_m[["subject_id"] + morph_final], on="subject_id", how="left"
    )
    df_all = df_all.merge(df_r[["subject_id"] + region_final], on="subject_id", how="left")
    df_all = df_all.merge(df_ci[["subject_id"] + ci4_names], on="subject_id", how="left")
    df_all["ci10_normal"] = ci10
    df_all["ci20_mild"] = ci20
    X = np.nan_to_num(df_all[feature_names].values.astype(float), nan=0.0)
    return _make_data_dict(y, X, None, None, None)
