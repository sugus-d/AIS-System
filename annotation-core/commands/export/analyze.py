#!/usr/bin/env python3
"""Compute feature importance for morph_region_ci_40d: Permutation Importance → CI decomposition → aggregation.

Usage:
    uv run python -m commands.export.analyze
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402  # 需在 warnings 过滤后导入
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from commands.export.config import FEATURE_IMPORTANCE_DIR  # noqa: E402
from features.utils.ci_decompose import _assign_group, _get_region_distribution  # noqa: E402
from modeling.models import _oversample, REGISTRY  # noqa: E402
from utils.logger import logger  # noqa: E402

# 特征筛选阈值：|r| 高于此值视为共线
_CORR_COLLINEAR = 0.85
# 系数绝对值低于此值视为零（Lasso/Logistic 稀疏解）
_COEF_EPS = 1e-6
# Cobb 角严重度分级阈值
_COBB_MILD = 10
_COBB_MODERATE = 20
# 特征名按 "_" 切分后的最小/完整段数（如 "nr_normal_angle__pw"）
_NAME_PARTS_MIN = 3
_NAME_PARTS_FULL = 4

# ── CI formula loading ──


def _load_ci_formulas() -> dict[str, list[tuple[str, float]]]:
    """Load CI formulas from results_compressed.csv.

    Returns: {ci_feature_name: [(region_feature, coefficient), ...]}
    Only includes the 4 CI features used in morph_region_ci_40d.
    """
    df = pd.read_csv("results/modeling/composite/results_compressed.csv")
    # Map group+method to ci_feature names used in the model
    used = {
        "height_dm": "ci_height_dm",
        "mean_curv_dm": "ci_mean_curv_dm",
        "normal_angle_pw": "ci_normal_angle_pw",
        "normal_vector_cos_pw": "ci_normal_vector_cos_pw",
    }
    formulas = {}
    for _, r in df.iterrows():
        key = f"{r['group']}_{r['method']}"
        key = key.replace("_orchestrated", "")
        ci_name = used.get(key)
        if ci_name is None:
            continue
        feats = r["feats"].split("|")
        coefs = [float(x) for x in r["coefs"].split("|")]
        formulas[ci_name] = list(zip(feats, coefs, strict=True))
    return formulas


def _compute_dual_ci_formulas() -> dict[str, list[tuple[str, float]]]:
    """Compute ci10/ci20 formulas via LogisticRegression.

    Returns: {"ci10_normal": [(feat, coef), ...], "ci20_mild": [(feat, coef), ...]}
    """
    from scipy.stats import pearsonr
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    d = Path("results/extraction/features_extraction/back_v1")
    df_b = pd.read_csv(d / "basic.csv").dropna(subset=["max_cobb"])
    df_r = pd.read_csv(d / "region_asymmetry.csv").dropna(subset=["max_cobb"])
    y = df_b["max_cobb"].values.astype(float)

    # All 2700 region features (6 measures × 2 modes)
    all_cols = []
    for m in ["normal_angle", "normal_vector_cos", "height", "mean_curv", "gauss_curv", "roughness"]:
        for mt in ["dm", "pw"]:
            sfx = f"_{m}__pw" if mt == "pw" else f"_{m}"
            all_cols.extend(c for c in df_r.columns if c.endswith(sfx) and c not in ("subject_id", "max_cobb"))
    Xr_all = df_r[all_cols].values.astype(float)

    def _build(target: np.ndarray, C: float, thr: float, label: str) -> list[tuple[str, float]]:
        rv = np.array([abs(pearsonr(Xr_all[:, i], target)[0]) for i in range(len(all_cols))])
        keep = np.where(rv > thr)[0]
        order = np.argsort(-rv[keep])
        cr = np.abs(np.corrcoef(Xr_all[:, keep].T))
        dd = [order[0]]
        for idx in order[1:]:
            if not any(cr[idx, j] > _CORR_COLLINEAR for j in dd):
                dd.append(idx)
        keep2 = [keep[i] for i in sorted(dd)]
        fcols = [all_cols[i] for i in keep2]
        sc = StandardScaler()
        Xs = sc.fit_transform(Xr_all[:, keep2])
        lr = LogisticRegression(
            C=C, l1_ratio=0.95, solver="saga", max_iter=10000, class_weight="balanced", random_state=42
        )
        lr.fit(Xs, target)
        nz = np.where(np.abs(lr.coef_[0]) > _COEF_EPS)[0]
        return [(fcols[i], lr.coef_[0][i]) for i in nz]

    yb10 = (y < _COBB_MILD).astype(float)
    yb20 = (y > _COBB_MODERATE).astype(float)
    return {
        "ci10_normal": _build(yb10, 0.1, 0.05, "ci10"),
        "ci20_mild": _build(yb20, 0.2, 0.1, "ci20"),
    }


# ── CI decomposition ──


def _decompose_ci_importance(
    ci_importance: dict[str, float],
    ci_formulas: dict[str, list[tuple[str, float]]],
    dual_ci_formulas: dict[str, list[tuple[str, float]]],
) -> dict[str, float]:
    """Decompose CI feature importance to constituent region features.

    Each CI's importance is allocated proportionally by |coef|.
    """
    all_formulas = {**ci_formulas, **dual_ci_formulas}
    # Add non-ci-prefixed aliases for formulas that have them
    for key in list(all_formulas.keys()):
        if key.startswith("ci_"):
            all_formulas[key[3:]] = all_formulas[key]
    decomposed: dict[str, float] = {}
    for ci_name, imp in ci_importance.items():
        if ci_name not in all_formulas:
            continue
        feats_coefs = all_formulas[ci_name]
        total_abs = sum(abs(c) for _, c in feats_coefs)
        if total_abs == 0:
            continue
        for feat, coef in feats_coefs:
            share = imp * abs(coef) / total_abs
            decomposed[feat] = decomposed.get(feat, 0.0) + share
    return decomposed


# ── Aggregation ──


def _aggregate_by_measurement(feature_imp: dict[str, float]) -> dict[str, float]:
    """Aggregate feature importance by 6 measurement type groups."""
    groups: dict[str, float] = {}
    for name, imp in feature_imp.items():
        g = _assign_group(name)
        if g == "Other":
            g = "Morph"  # ponytail: fallback for unknown features
        groups[g] = groups.get(g, 0.0) + imp
    return groups


def _aggregate_by_region(feature_imp: dict[str, float]) -> dict[str, float]:
    """Aggregate feature importance by 5 anatomical regions + Morph + Clinical.

    Uses area-based proportional splitting for region features.
    """
    regions: dict[str, float] = {}
    for name, imp in feature_imp.items():
        g = _assign_group(name)
        if g == "Other":
            g = "Morph"
        if g in ("Morph", "Clinical"):
            regions[g] = regions.get(g, 0.0) + imp
            continue
        # Region feature: split by area proportion
        dist = _get_region_distribution(name)
        if dist:
            for band, ratio in dist.items():
                regions[band] = regions.get(band, 0.0) + imp * ratio
        else:
            regions["Morph"] = regions.get("Morph", 0.0) + imp
    return regions


# ── Main ──


def main():
    FEATURE_IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Loading morph_region_ci_40d scheme...")
    from features.selectors.schemes import _load_dual_ci

    data = _load_dual_ci()
    y = data["y"]
    X = data["X_basic"]
    feat_names = data.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
    logger.info(f"  Features: {len(feat_names)}D")
    ci_formulas = _load_ci_formulas()
    dual_ci_formulas = _compute_dual_ci_formulas()
    all_ci_formula_keys = set(ci_formulas) | set(dual_ci_formulas)
    ci_names = [
        n for n in feat_names if n in all_ci_formula_keys or f"ci_{n}" in all_ci_formula_keys or n.startswith("ci")
    ]
    ci_set = set(ci_names)
    logger.info(f"  CI features: {ci_names}")

    logger.info("\nTraining HistGBRT...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = REGISTRY["HistGBRT"]()
    w = model._build_weight(y)  # noqa: SLF001  # modeling.models 训练辅助函数
    X_a, y_a = _oversample(Xs, y, w)
    model._reg.fit(X_a, y_a)  # noqa: SLF001  # HistGBRT 内部 sklearn 回归器
    logger.info(f"  Trained: {model._reg.n_iter_} trees")  # noqa: SLF001

    logger.info("\nComputing Permutation Importance (30 repeats)...")
    result = permutation_importance(
        model._reg, Xs, y, n_repeats=30, random_state=42, n_jobs=1, scoring="neg_mean_absolute_error"  # noqa: SLF001
    )
    importance = result.importances_mean

    # Build importance dict
    feat_imp = dict(zip(feat_names, importance, strict=True))

    # Separate CI vs non-CI features
    direct_imp: dict[str, float] = {}
    ci_imp: dict[str, float] = {}
    for name, imp in feat_imp.items():
        if name in ci_set:
            ci_imp[name] = imp
        else:
            direct_imp[name] = imp

    logger.info(f"  Direct features: {len(direct_imp)}")
    logger.info(f"  CI features: {len(ci_imp)}")
    for name, imp in sorted(ci_imp.items(), key=lambda x: -abs(x[1])):
        logger.info(f"    {name}: {imp:.4f}")

    # Load CI formulas and decompose
    logger.info("\nDecomposing CI features...")
    ci_formulas = _load_ci_formulas()
    dual_ci_formulas = _compute_dual_ci_formulas()
    decomposed = _decompose_ci_importance(ci_imp, ci_formulas, dual_ci_formulas)
    logger.info(f"  Decomposed to {len(decomposed)} region features")

    # Merge direct + decomposed
    all_imp = dict(direct_imp)
    for name, imp in decomposed.items():
        all_imp[name] = all_imp.get(name, 0.0) + imp

    # Sort by importance
    sorted_imp = sorted(all_imp.items(), key=lambda x: -abs(x[1]))
    total_imp = sum(abs(v) for v in all_imp.values())

    # ── Output 1: per-feature ranking ──
    rows = []
    for rank, (name, imp) in enumerate(sorted_imp, 1):
        share = imp / total_imp * 100 if total_imp > 0 else 0
        g = _assign_group(name)
        # subgroup: for region features, parse region + measure
        subgroup = ""
        if g not in ("Morph", "Clinical", "Normal Angle", "Curvature", "Roughness", "Height"):
            # Try to extract region+measure from the name
            parts = name.split("_")
            if len(parts) >= _NAME_PARTS_MIN and parts[0] in ("nr", "st", "sp", "ax", "wa", "wl"):
                prefix = parts[0]
                if len(parts) >= _NAME_PARTS_FULL and parts[1] in ("nr", "st", "sp", "ax", "wa", "wl"):
                    prefix = f"{parts[0]}_{parts[1]}"
                # Get measure from name
                for m_pat, m_name in [
                    ("normal_angle", "normal_angle"),
                    ("normal_vector_cos", "normal_vector_cos"),
                    ("gauss_curv", "gauss_curv"),
                    ("mean_curv", "mean_curv"),
                    ("roughness", "roughness"),
                    ("_height", "height"),
                ]:
                    if m_pat in name:
                        subgroup = f"{prefix} + {m_name}"
                        break
        rows.append(
            {
                "rank": rank,
                "feature": name,
                "importance": imp,
                "share_pct": round(share, 1),
                "group": g,
                "subgroup": subgroup,
            }
        )
    df_out = pd.DataFrame(rows)
    df_out.to_csv(FEATURE_IMPORTANCE_DIR / "feature_importance_decomposed.csv", index=False)
    logger.info(f"\n  Saved: feature_importance_decomposed.csv ({len(df_out)} features)")

    # ── Output 2: by measurement type ──
    by_measure = _aggregate_by_measurement(all_imp)
    total_m = sum(by_measure.values())
    mt_rows = []
    for g in ["Normal Angle", "Curvature", "Roughness", "Height", "Morph", "Clinical"]:
        imp = by_measure.get(g, 0.0)
        share = imp / total_m * 100 if total_m > 0 else 0.0
        mt_rows.append({"new_group": g, "importance": round(imp, 4), "share_pct": f"{share:.1f}"})
    pd.DataFrame(mt_rows).to_csv(FEATURE_IMPORTANCE_DIR / "importance_by_group.csv", index=False)
    logger.info(f"  Saved: importance_by_group.csv ({len(mt_rows)} groups)")

    # ── Output 3: by anatomical region ──
    by_region = _aggregate_by_region(all_imp)
    total_r = sum(by_region.values())
    uv_rows = []
    for band in ["Shoulder", "Scapula", "Axilla", "Waist", "Pelvis", "Morph", "Clinical"]:
        imp = by_region.get(band, 0.0)
        share = imp / total_r * 100 if total_r > 0 else 0.0
        uv_rows.append({"horizontal_band": band, "importance": round(imp, 4), "share_pct": f"{share:.1f}"})
    pd.DataFrame(uv_rows).to_csv(FEATURE_IMPORTANCE_DIR / "importance_by_horizontal_band.csv", index=False)
    logger.info(f"  Saved: importance_by_horizontal_band.csv ({len(uv_rows)} bands)")

    logger.info(f"\nDone!  Total importance: {total_imp:.4f}")


if __name__ == "__main__":
    main()
