"""Orchestration: SHAP waterfall + convergence plots for 20 subjects.

Usage:
    uv run python -m commands.export.charts_waterfall
"""

from __future__ import annotations

from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler

from features.utils.ci_decompose import CI_MEASURE, decompose_waterfall, load_ci_formulas
from features.utils.ci_display import feature_display_name
from modeling.models import _oversample, REGISTRY
from modeling.training.result_paths import scheme_results_path
from utils.logger import logger
from utils.paths import EXPORT_SHAP_DIR, FEATURES_EXTRACTION_DIR, FEATURES_SELECTION_DIR
from visualization._style import ACADEMIC_STYLE
from visualization.waterfall_panels import render_residual_convergence, render_tree_structure, render_waterfall

# ── Typography (academic-figure-skill baseline) ──
mpl.rcParams.update(ACADEMIC_STYLE)

MM = 1 / 25.4

GROUP_NAMES = ["Normal Angle", "Morph", "Clinical", "Curvature", "Height", "Roughness", "Other"]
FIXED_ORDER = ("Normal Angle", "Curvature", "Height", "Roughness", "Morph", "Clinical", "Other")

SEVERITY_COLORS = {
    "Normal":   "#5A8C5A",
    "Mild":   "#C88A4A",
    "Moderate": "#4A72A0",
    "Severe":   "#B06060",
}

OUT_DIR = EXPORT_SHAP_DIR
PRED_CSV = scheme_results_path("ensemble_composite_v7_ai60") / "Ensemble" / "predictions.csv"

SELECTED_SUBJECTS = {
    "Normal": [
        ("S0020",  "correct, 9 -> 6"),
        ("S0087",  "correct, 8 -> 5"),
        ("S0060",  "over-pred, 4 -> 19, +15"),
        ("S0097",  "over-pred, 6 -> 13, +7"),
        ("S0012",  "over-pred, 8 -> 12, +4"),
    ],
    "Mild": [
        ("S0029",  "correct, 10 -> 10"),
        ("S0037",  "correct, 13 -> 13"),
        ("S0018",  "over-pred, 18 -> 30, +12"),
        ("S0114",  "under-pred, 17 -> 8, -9"),
        ("S0047",  "over-pred, 18 -> 24, +6"),
    ],
    "Moderate": [
        ("S0066",  "correct, 22 -> 22"),
        ("S0002",  "correct, 38 -> 38"),
        ("S0064",  "under-pred, 34 -> 14, -20"),
        ("S0040",  "under-pred, 28 -> 19, -9"),
        ("S0083",  "over-pred, 23 -> 36, +13"),
    ],
    "Severe": [
        ("S0069",  "correct, 51 -> 51"),
        ("S0085",  "correct, 41 -> 42"),
        ("S0052",  "under-pred, 48 -> 38, -11"),
        ("S0027",  "under-pred, 40 -> 37, -3"),
        ("S0005",  "under-pred, 72 -> 54, -18"),
    ],
}


# ── Data loading ──

def build_feature_matrix() -> tuple:
    from features.selectors.schemes import _load_selection
    data = _load_selection("v0.1.0")
    y, X = data["y"], data["X_basic"]

    d = FEATURES_SELECTION_DIR / "v0.1.0"
    df_b = pd.read_csv(FEATURES_EXTRACTION_DIR / "v0.1.0" / "basic.csv").dropna(subset=["max_cobb"])
    df_m = pd.read_csv(d / "morphology.csv").dropna(subset=["max_cobb"])
    df_r = pd.read_csv(d / "region_asymmetry.csv").dropna(subset=["max_cobb"])
    df_c = pd.read_csv(d / "ci.csv")
    basic_cols = [c for c in df_b.columns if c not in ("subject_id", "max_cobb")]
    morph_cols = [c for c in df_m.columns if c not in ("subject_id", "max_cobb")]
    region_cols = [c for c in df_r.columns if c not in ("subject_id", "max_cobb")]
    ci_cols = [c for c in df_c.columns if c != "subject_id"]
    all_names = basic_cols + morph_cols + region_cols + ci_cols
    ci_indices = list(range(len(all_names) - len(ci_cols), len(all_names)))

    pred_df = pd.read_csv(PRED_CSV)
    sids = set(pred_df["subject_id"])
    df_all = df_b[["subject_id"]].merge(
        df_m[["subject_id"]], on="subject_id", how="left")
    df_all = df_all.merge(df_r[["subject_id"]], on="subject_id", how="left")
    df_all = df_all.merge(df_c[["subject_id"]], on="subject_id", how="left")
    mask = df_all["subject_id"].isin(sids)
    return X[mask], y[mask], all_names, df_all[mask]["subject_id"].tolist(), ci_indices


# ── Main ──

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Loading features (v0.1.0 selection)...")
    X, y, feature_names, subject_ids, ci_indices = build_feature_matrix()

    gc = Counter()
    for fn in feature_names:
        from features.utils.ci_decompose import _assign_group
        gc[_assign_group(fn)] += 1
    logger.info(f"  X: {X.shape}, y: {y.shape}, subjects: {len(subject_ids)}")
    for g in GROUP_NAMES:
        if gc.get(g, 0) > 0:
            logger.info(f"    {g}: {gc[g]} features")

    formulas = load_ci_formulas()
    if formulas:
        logger.info(f"  CI features ({len(ci_indices)}):")
        for idx in ci_indices:
            logger.info(f"    {feature_names[idx]} -> {CI_MEASURE.get(feature_names[idx], '?')}")

    logger.info("\nTraining full-data HistGBRT...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = REGISTRY["HistGBRT"]()
    w = model._build_weight(y)  # noqa: SLF001  # modeling.models 训练辅助函数
    X_a, y_a = _oversample(Xs, y, w)
    model._reg.fit(X_a, y_a)  # noqa: SLF001  # HistGBRT 内部 sklearn 回归器
    logger.info(f"  Trained: {model._reg.n_iter_} trees")  # noqa: SLF001

    logger.info("\nComputing SHAP...")
    explainer = shap.TreeExplainer(model._reg)  # noqa: SLF001
    shap_values = explainer.shap_values(Xs)
    expected_val = float(explainer.expected_value.item())
    logger.info(f"  SHAP: {shap_values.shape}, baseline={expected_val:.1f}")

    pred_df = pd.read_csv(PRED_CSV)
    sid_to_pred = {r["subject_id"]: r for _, r in pred_df.iterrows()}
    sid_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
    # 位置索引 → 名字集合（decompose_waterfall 按名匹配 CI 特征，保等）
    ci_names = {feature_names[i] for i in ci_indices}

    logger.info("\nGenerating waterfall plots (20 subjects)...")
    subjects_info = []
    for sev, subjects in SELECTED_SUBJECTS.items():
        for sid, label in subjects:
            if sid not in sid_to_pred or sid not in sid_to_idx:
                logger.info(f"  WARNING: {sid} not found, skipping")
                continue

            row = sid_to_pred[sid]
            idx = sid_to_idx[sid]
            # decompose_waterfall 内部用 SHAP 可加性取真实模型输出（expected + sum(shap)）
            contrib, pred_val = decompose_waterfall(
                shap_values[idx], expected_val, feature_names, ci_names, GROUP_NAMES)

            fig, ax = plt.subplots(figsize=(max(100, 8 * 18) * MM, 65 * MM))
            render_waterfall(
                ax, contrib, expected_val,
                row["max_cobb_true"], pred_val,
                sid, sev, label,
                fixed_order=FIXED_ORDER,
            )
            out_path = OUT_DIR / "瀑布图" / sev / f"瀑布图_{sid}_{sev}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300)
            fig.savefig(out_path.with_suffix(".pdf"), dpi=300)
            plt.close(fig)
            logger.info(f"  瀑布图_{sid}_{sev}.png")

            subjects_info.append({
                "sid": sid, "severity": sev,
                "true_cobb": row["max_cobb_true"], "pred_cobb": pred_val,
            })

    # ── Residual convergence + tree structure (for selected subjects) ──
    logger.info("\nGenerating residual convergence & tree structure plots...")
    CONV_DIR = EXPORT_SHAP_DIR
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    # Pick one subject per severity level
    for sev, subs in SELECTED_SUBJECTS.items():
        sid, label = subs[0]
        if sid not in sid_to_idx:
            continue
        idx = sid_to_idx[sid]
        true_c = sid_to_pred[sid]["max_cobb_true"]

        # Residual convergence
        stages = list(model._reg.staged_predict(Xs[idx:idx + 1]))  # noqa: SLF001
        sv_full = np.array([float(s[0]) for s in stages])
        bl = float(model._reg._baseline_prediction)  # noqa: SLF001  # sklearn 内部基线预测值
        tc_full = np.array([sv_full[0] - bl] + [sv_full[i] - sv_full[i - 1] for i in range(1, len(sv_full))])
        res_full = np.array([abs(true_c - sv_full[i]) for i in range(len(sv_full))])

        fig, ax1 = plt.subplots(figsize=(130 * MM, 65 * MM))
        ax2 = ax1.twinx()
        render_residual_convergence(ax1, ax2, sv_full, tc_full, res_full, true_c, sid)
        plt.subplots_adjust(bottom=0.32)
        (CONV_DIR / "残差收敛图" / sev).mkdir(parents=True, exist_ok=True)
        fig.savefig(CONV_DIR / "残差收敛图" / sev / f"残差收敛图_{sid}.png", dpi=300)
        fig.savefig(CONV_DIR / "残差收敛图" / sev / f"残差收敛图_{sid}.pdf", dpi=300)
        plt.close(fig)
        logger.info(f"  残差收敛图_{sid}.png")

        # Tree structure
        fig, axes = plt.subplots(3, 1, figsize=(100 * MM, 160 * MM))
        fig.subplots_adjust(hspace=0.15)
        tree_labels = [feature_display_name(f) for f in feature_names]
        render_tree_structure(axes, model._reg._predictors, Xs[idx], feature_names, feature_labels=tree_labels, subject_id=sid)  # noqa: SLF001
        (CONV_DIR / "树结构图" / sev).mkdir(parents=True, exist_ok=True)
        fig.savefig(CONV_DIR / "树结构图" / sev / f"树结构图_{sid}.png", dpi=300)
        fig.savefig(CONV_DIR / "树结构图" / sev / f"树结构图_{sid}.pdf", dpi=300)
        plt.close(fig)
        logger.info(f"  树结构图_{sid}.png")

    logger.info(f"\nDone!  {OUT_DIR}/")


if __name__ == "__main__":
    main()
