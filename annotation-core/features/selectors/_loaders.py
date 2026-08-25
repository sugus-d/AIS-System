"""特征方案 loader — SELECTION_REGISTRY 的加载实现（纯函数，无注册表逻辑）。

从 results/features_* 读 CSV 构建 make_data_dict 分块数据。
注册表定义在 :mod:`features.selectors.schemes`。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.selectors._utils import make_data_dict

# ── 特征筛选阈值（各 scheme 共用的超参，历史调优值，勿随意改动） ──
_R_MORPH = 0.15              # morphology |Pearson r| 保留阈值
_R_REGION = 0.2              # region |Pearson r| 保留阈值
_P_VALUE = 0.05              # ANOVA F 检验 p 值阈值
_CORR_DEDUP = 0.85           # 去高相关阈值
_COEF_EPS = 1e-6             # Lasso 系数视为非零的绝对值下限
_TOP_MORPH = 10              # morphology 保留 top N
_MIN_ANOVA_GROUPS = 2        # ANOVA 至少需要的等级组数
_MIN_GROUP_FEATURES = 2      # 每组至少保留的特征数
_COBB_LIGHT = 10             # Cobb 10°（轻度阈值，用于二分类目标）
_COBB_CLINICAL = 20          # Cobb 20°（临床阈值，用于二分类目标）


def _load_extraction(version: str) -> dict:
    """加载特征提取阶段的原始数据。"""
    d = Path("results/extraction/features_extraction") / version
    df_b = pd.read_csv(d / "basic.csv").dropna(subset=["max_cobb"])
    df_m = pd.read_csv(d / "morphology.csv").dropna(subset=["max_cobb"])
    df_r = pd.read_csv(d / "region_asymmetry.csv").dropna(subset=["max_cobb"])

    y = df_b["max_cobb"].values.astype(float)
    basic_cols = [c for c in df_b.columns if c not in ("subject_id", "max_cobb")]
    morph_cols = [c for c in df_m.columns if c not in ("subject_id", "max_cobb")]
    region_cols = [c for c in df_r.columns if c not in ("subject_id", "max_cobb")]

    Xb = df_b[basic_cols].values.astype(float)
    Xm = df_m[morph_cols].values.astype(float)
    Xr = df_r[region_cols].values.astype(float)

    return make_data_dict(y, Xb, Xm, Xr, region_cols)

def _compute_ci_features(subject_ids: pd.Series, features_df: pd.DataFrame) -> pd.DataFrame:
    """从 region 特征实时构建 6 measure × 2(pw/dm) = 12 组 CI 特征。

    6 measure: normal_angle, normal_vector_cos, height, mean_curv, gauss_curv, roughness
    （不含 normal_vector、normal_vector_sin）
    """
    from features.extractors.asymmetry import compute_ci, load_formulas
    formulas = load_formulas("results/modeling/composite/results_compressed.csv")
    # 只保留 6 core measure
    core = {"normal_angle", "normal_vector_cos", "height", "mean_curv", "gauss_curv", "roughness"}
    rows: list[dict] = []
    for sid in subject_ids:
        row: dict = {"subject_id": sid}
        for group, (feats, coefs) in formulas.items():
            measure = group.rsplit("_", 1)[0] if group.endswith(("_pw", "_dm")) else group
            if measure in core:
                row[group] = compute_ci(sid, group, feats, coefs, features_df)
        rows.append(row)
    return pd.DataFrame(rows)

def _load_canonical_union_64d(_version: str) -> dict:
    """并集：morph_region_ci_37d ∪ canonical_44d = 64D。

    CI 的 canonical 前缀对齐（normal_angle_dm → ci_normal_angle_dm），
    2 个 canonical CI 不存在于提取，去重后 union = morph 16 + region 39 + CI 4 + basic 5 = 64D。
    """
    d = Path("results/extraction/features_extraction/back_v1")
    ds = Path("results/extraction/features_selection/back_v1")
    df_b = pd.read_csv(d / "basic.csv").dropna(subset=["max_cobb"])
    df_m_ext = pd.read_csv(d / "morphology.csv").dropna(subset=["max_cobb"])
    df_r_ext = pd.read_csv(d / "region_asymmetry.csv").dropna(subset=["max_cobb"])
    df_m = pd.read_csv(ds / "morphology.csv").dropna(subset=["max_cobb"])
    df_r = pd.read_csv(ds / "region_asymmetry.csv").dropna(subset=["max_cobb"])
    df_c = pd.read_csv(ds / "ci.csv")

    y = df_b["max_cobb"].values.astype(float)
    basic_cols = [c for c in df_b.columns if c not in ("subject_id", "max_cobb")]
    morph_sel = [c for c in df_m.columns if c not in ("subject_id", "max_cobb")]
    region_sel = [c for c in df_r.columns if c not in ("subject_id", "max_cobb")]
    ci_cols = [c for c in df_c.columns if c != "subject_id"]

    # canonical_47d morph 存在于 back_v1 提取的
    canon_morph = ["neck_root_vertical_diff", "neck_root_slope_angle",
        "shoulder_transition_vertical_diff", "scapular_peaks_anterior_diff",
        "scapular_peaks_vertical_diff", "scapular_peaks_slope_angle",
        "axilla_anterior_diff", "axilla_vertical_diff", "waist_distance_3d",
        "spine_P0_P1_length", "spine_curvature_P0P1_vs_P3P4"]
    morph_all = sorted(set(morph_sel) | set(canon_morph))

    # canonical_47d region 存在于 back_v1 提取的
    canon_region = ["wa_wl_p0_p5_normal_angle", "ax_wa_p0_p3_normal_angle",
        "wl_p2_p4_normal_angle__pw", "sp_wa_p0_p1_normal_vector_cos",
        "nr_sp_p0_p2_height", "nr_p0_p5_height", "wa_p5_p3_mean_curv__pw",
        "nr_sp_p0_p1_roughness", "st_ax_p0_p1_normal_vector_cos",
        "ax_wa_p2_p5_normal_vector_cos__pw", "ax_wl_p5_p4_normal_angle__pw",
        "st_ax_p1_p2_normal_angle", "wa_wl_p0_p4_gauss_curv", "wa_p5_p4_mean_curv",
        "st_sp_p3_p4_height", "ax_p0_p1_roughness__pw", "nr_sp_p1_p3_normal_angle",
        "sp_ax_p5_p3_roughness__pw", "ax_wa_p0_p1_mean_curv", "wl_p1_p5_normal_angle",
        "ax_p5_p3_normal_angle", "st_sp_p3_p4_roughness", "sp_ax_p5_p3_gauss_curv",
        "wa_p0_p3_normal_angle"]
    region_all = sorted(set(region_sel) | set(canon_region))

    # CI union: canonical CI 前缀对齐到 ci_（只取 back_v1 中存在的）
    canon_ci = ["ci_normal_angle_dm", "ci_roughness_pw", "ci_gauss_curv_dm"]
    canon_ci_avail = [c for c in canon_ci if c in ci_cols]
    ci_all = sorted(set(ci_cols) | set(canon_ci_avail))

    feature_names = basic_cols + morph_all + region_all + ci_all
    df_all = df_b[["subject_id", "max_cobb"] + basic_cols].merge(
        df_m_ext[["subject_id"] + morph_all], on="subject_id", how="left")
    df_all = df_all.merge(df_r_ext[["subject_id"] + region_all], on="subject_id", how="left")
    df_all = df_all.merge(df_c[["subject_id"] + ci_all], on="subject_id", how="left")

    X = np.nan_to_num(df_all[feature_names].values.astype(float), nan=0.0)
    return make_data_dict(y, X, None, None, None)

def _load_dual_ci() -> dict:
    """加载 morph_region_ci_40d 方案（构成语义名）：10° CI + 20° CI"""
    import warnings

    from scipy.stats import f_oneway, pearsonr
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")
    import logging

    logging.disable(logging.CRITICAL)
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LassoCV, LogisticRegression

    from features.selectors._utils import _SEVERITY_BINS
    from features.selectors._utils import make_data_dict as _make_data_dict
    from features.selectors.scheme_morph_region_ci_35d import (
        _build_ci_features,
        _dedup_by_r,
        _filter_ci,
        _filter_morphology,
    )

    d = Path("results/extraction/features_extraction/back_v1")
    df_b = pd.read_csv(d / "basic.csv").dropna(subset=["max_cobb"])
    df_m = pd.read_csv(d / "morphology.csv").dropna(subset=["max_cobb"])
    df_r = pd.read_csv(d / "region_asymmetry.csv").dropna(subset=["max_cobb"])
    y = df_b["max_cobb"].values.astype(float)
    y4 = np.digitize(y, _SEVERITY_BINS[1:-1])
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
        rv = np.array([abs(pearsonr(Xr_all[:, i], target)[0]) for i in range(2700)])
        keep = np.where(rv > thr)[0]
        fcols = [all_cols[i] for i in keep]
        order = np.argsort(-rv[keep])
        cr = np.abs(np.corrcoef(Xr_all[:, keep].T))
        dd = [order[0]]
        for idx in order[1:]:
            if not any(cr[idx, j] > _CORR_DEDUP for j in dd):
                dd.append(idx)
        fcols = [fcols[i] for i in sorted(dd)]
        keep2 = [keep[i] for i in sorted(dd)]
        sc = StandardScaler()
        Xs = sc.fit_transform(Xr_all[:, keep2])
        lr = LogisticRegression(
            C=C, l1_ratio=0.95, solver="saga", max_iter=10000, class_weight="balanced", random_state=42
        )
        lr.fit(Xs, target)
        nz = np.where(np.abs(lr.coef_[0]) > _COEF_EPS)[0]
        return Xs[:, nz] @ lr.coef_[0][nz]

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
    from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    import numpy as np
    import pandas as pd

    from features.selectors._utils import _SEVERITY_BINS
    from features.selectors._utils import make_data_dict as _make_data_dict
    from features.selectors.scheme_morph_region_ci_35d import (
        _build_ci_features,
        _dedup_by_r,
        _filter_ci,
        _filter_morphology,
    )

    d = Path("results/extraction/features_extraction/back_v1")
    df_b = pd.read_csv(d / "basic.csv").dropna(subset=["max_cobb"])
    df_m = pd.read_csv(d / "morphology.csv").dropna(subset=["max_cobb"])
    df_r = pd.read_csv(d / "region_asymmetry.csv").dropna(subset=["max_cobb"])
    y = df_b["max_cobb"].values.astype(float)
    y4 = np.digitize(y, _SEVERITY_BINS[1:-1])
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
        rv = np.array([abs(pearsonr(Xa[:, i], target)[0]) for i in range(2700)])
        ki = np.where(rv > thr)[0]
        o = np.argsort(-rv[ki])
        cr = np.abs(np.corrcoef(Xa[:, ki].T))
        dd = [o[0]]
        for idx in o[1:]:
            if not any(cr[idx, j] > _CORR_DEDUP for j in dd):
                dd.append(idx)
        k2 = [ki[i] for i in sorted(dd)]
        sc = StandardScaler()
        Xs = sc.fit_transform(Xa[:, k2])
        lr = LogisticRegression(
            C=C, l1_ratio=0.95, solver="saga", max_iter=10000, class_weight="balanced", random_state=42
        )
        lr.fit(Xs, target)
        nz = np.where(np.abs(lr.coef_[0]) > _COEF_EPS)[0]
        return set([all_cols[k2[i]] for i in nz]), Xs[:, nz] @ lr.coef_[0][nz]

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

def _load_canonical_44d(_version: str) -> dict:
    """加载 canonical 固定参考集，从 back_v1 提取数据中取可用特征。

    canonical_47d 原数据仅 N=60，从 back_v1 提取（N=122）中取它的 45 个特征，
    去掉 back_v1 不存在的 3 个（spine_P3_P4_angle_vertical, waist_Y_asymmetry, axilla_Y_asymmetry），
    CI 前缀对齐（normal_angle_dm → ci_normal_angle_dm）后 = 42D。
    """
    d = Path("results/extraction/features_extraction/back_v1")
    ds = Path("results/extraction/features_selection/back_v1")
    df_b = pd.read_csv(d / "basic.csv").dropna(subset=["max_cobb"])
    df_m = pd.read_csv(d / "morphology.csv").dropna(subset=["max_cobb"])
    df_r = pd.read_csv(d / "region_asymmetry.csv").dropna(subset=["max_cobb"])
    df_c = pd.read_csv(ds / "ci.csv")

    y = df_b["max_cobb"].values.astype(float)
    basic_cols = [c for c in df_b.columns if c not in ("subject_id", "max_cobb")]

    # canonical_47d morph list, minus ones not in extraction
    canon_morph = ["neck_root_vertical_diff", "neck_root_slope_angle",
        "shoulder_transition_vertical_diff", "scapular_peaks_anterior_diff",
        "scapular_peaks_vertical_diff", "scapular_peaks_slope_angle",
        "axilla_anterior_diff", "axilla_vertical_diff", "waist_distance_3d",
        "spine_P0_P1_length", "spine_curvature_P0P1_vs_P3P4"]
    morph_cols = [c for c in canon_morph if c in df_m.columns]

    # canonical_47d region list, minus ones not in extraction
    canon_region = ["wa_wl_p0_p5_normal_angle", "ax_wa_p0_p3_normal_angle",
        "wl_p2_p4_normal_angle__pw", "sp_wa_p0_p1_normal_vector_cos",
        "nr_sp_p0_p2_height", "nr_p0_p5_height", "wa_p5_p3_mean_curv__pw",
        "nr_sp_p0_p1_roughness", "st_ax_p0_p1_normal_vector_cos",
        "ax_wa_p2_p5_normal_vector_cos__pw", "ax_wl_p5_p4_normal_angle__pw",
        "st_ax_p1_p2_normal_angle", "wa_wl_p0_p4_gauss_curv", "wa_p5_p4_mean_curv",
        "st_sp_p3_p4_height", "ax_p0_p1_roughness__pw", "nr_sp_p1_p3_normal_angle",
        "sp_ax_p5_p3_roughness__pw", "ax_wa_p0_p1_mean_curv", "wl_p1_p5_normal_angle",
        "ax_p5_p3_normal_angle", "st_sp_p3_p4_roughness", "sp_ax_p5_p3_gauss_curv",
        "wa_p0_p3_normal_angle"]
    region_cols = [c for c in canon_region if c in df_r.columns]

    ci_cols = [c for c in df_c.columns if c != "subject_id"]

    feature_names = basic_cols + morph_cols + region_cols + ci_cols
    df_all = df_b[["subject_id", "max_cobb"] + basic_cols].merge(
        df_m[["subject_id"] + morph_cols], on="subject_id", how="left")
    df_all = df_all.merge(df_r[["subject_id"] + region_cols], on="subject_id", how="left")
    df_all = df_all.merge(df_c, on="subject_id", how="left")

    X = np.nan_to_num(df_all[feature_names].values.astype(float), nan=0.0)
    return make_data_dict(y, X, None, None, None)

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
