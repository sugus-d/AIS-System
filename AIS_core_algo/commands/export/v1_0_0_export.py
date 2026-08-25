#!/usr/bin/env python3
"""v1.0.0 方案批量导出：indices 表 + 特征重要性 + 单 case 瀑布图。

新模型（v1.0.0, boundary_ensemble_ridge.joblib）训练时已在模型目录落盘
可读 JSON/图表产物；本模块从模型包读取，对 v1.0.0 特征数据批量复算
批量 indices 表、全局特征重要性图与单 case 瀑布图（复用 v0.1.0 的
tables._compute_indices / permutation importance / SHAP 框架）。

用法:
    uv run python -m commands.export.v1_0_0_export
    uv run python -m commands.export --scheme v1.0.0   # 挂在主导出管线（CI 反解在步骤 1）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from commands.export.tables import _compute_indices
from features.selectors.schemes import SELECTION_REGISTRY
from modeling.model_package import load_model_package
from utils.logger import logger
from utils.paths import EXPORT_DIR

# ── v1.0.0 数据源（与 modeling/ensemble_boundary.py 常量一致） ──
MODEL_PKG = Path("results/modeling/models/v1.0.0/boundary_ensemble_ridge.joblib")
REGION_CSV = Path("results/extraction/features_extraction/v1.0.0/region_asymmetry.csv")
PRED_CSV = Path(
    "results/modeling/prediction/v1.0.0/ensemble-ai_refit_ridge_boundary/Ensemble/predictions.csv"
)
_SCHEME = "v1.0.0"
_COBB_BOUNDARIES = (10, 20, 40)
_SEVERITY_LABELS = ["Normal", "Mild", "Moderate", "Severe"]
_OUT_DIR = EXPORT_DIR / "v1.0.0"
_GROUP_NAMES = ["Normal Angle", "Morph", "Clinical", "Curvature", "Height", "Roughness", "Other"]
_FIXED_ORDER = ("Normal Angle", "Curvature", "Height", "Roughness", "Morph", "Clinical", "Other")


def _load_model_pkg() -> dict:
    """加载模型包（展平结构：模型/scaler/feature_names/CI 参数统一在顶层）。"""
    return load_model_package(str(MODEL_PKG))


def _load_scheme() -> tuple[np.ndarray, list[str], np.ndarray]:
    """加载 v1.0.0 30D 特征（含 CI）+ 特征名 + 真实 Cobb（对齐 predictions 行序）。"""
    pred_df = pd.read_csv(PRED_CSV)
    y = pred_df["max_cobb_true"].values.astype(float)
    scheme_data = SELECTION_REGISTRY[_SCHEME].load()
    X = np.asarray(scheme_data["X_basic"], dtype=np.float64)
    if len(X) != len(y):
        raise ValueError(f"scheme X_basic {len(X)} 行与 predictions {len(y)} 不一致")
    return X, scheme_data["feature_names"], y


def _indices_table() -> pd.DataFrame:
    """批量 5 不对称指数表（复用 v0.1.0 tables._compute_indices 同口径拟合）。"""
    df = pd.read_csv(REGION_CSV).dropna(subset=["max_cobb"])
    out = _compute_indices(df)
    out_path = _OUT_DIR / "v1.0.0_indices.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info(f"批量 indices 表: {out_path}（{len(out)} subjects × 6 列）")
    return out


def _select_subjects(pred_df: pd.DataFrame) -> list[tuple[str, str]]:
    """每个 severity 类选 3 个代表 subject：正确 / 高估 / 低估。"""
    selected: list[tuple[str, str]] = []
    for sev in _SEVERITY_LABELS:
        sub = pred_df[pred_df["class_true"] == sev].copy()
        if len(sub) == 0:
            continue
        sub["err"] = sub["max_cobb_pred"] - sub["max_cobb_true"]
        correct = sub.iloc[np.argmin(np.abs(sub["err"].values))]
        over = sub.iloc[np.argmax(sub["err"].values)]
        under = sub.iloc[np.argmin(sub["err"].values)]
        selected.append(
            (correct["subject_id"], f"correct, {correct['max_cobb_true']:.0f} -> {correct['max_cobb_pred']:.0f}")
        )
        selected.append(
            (
                over["subject_id"],
                f"over-pred, {over['max_cobb_true']:.0f} -> {over['max_cobb_pred']:.0f}, +{over['err']:.0f}",
            )
        )
        selected.append(
            (
                under["subject_id"],
                f"under-pred, {under['max_cobb_true']:.0f} -> {under['max_cobb_pred']:.0f}, {under['err']:.0f}",
            )
        )
    return selected


def _waterfalls(composite: dict, X: np.ndarray, y: np.ndarray) -> None:
    """批量单 case 瀑布图（SHAP 分解 composite 分量，每 severity 选 3 代表）。"""
    import shap

    from features.utils.ci_decompose import decompose_waterfall
    from modeling.metrics import SEVERITY_LABELS
    from visualization.waterfall_panels import render_waterfall

    pred_df = pd.read_csv(PRED_CSV)
    sids = pred_df["subject_id"].tolist()
    sid_to_idx = {sid: i for i, sid in enumerate(sids)}
    feature_names = composite["feature_names"]
    ci_names = set((composite.get("ci_formula_params") or {}).keys()) | {
        "height_dm",
        "mean_curv_dm",
        "mean_curv_pw",
        "normal_angle_pw",
        "normal_vector_cos_pw",
        "ci10_normal",
        "ci20_mild",
    }

    estimator = composite["model"]._reg  # noqa: SLF001  # 包装类内部回归器
    explainer = shap.TreeExplainer(estimator)
    X_s = composite["scaler"].transform(X)
    expected_val = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    from matplotlib import pyplot as plt

    for sid, label in _select_subjects(pred_df):
        if sid not in sid_to_idx:
            logger.info(f"  WARNING: {sid} 不在特征矩阵，跳过")
            continue
        idx = sid_to_idx[sid]
        shap_row = np.asarray(explainer.shap_values(X_s[idx : idx + 1])[0]).reshape(-1)
        # decompose_waterfall 内部用 SHAP 可加性取真实模型输出（expected + sum(shap)）
        contrib, pred_val = decompose_waterfall(shap_row, expected_val, feature_names, ci_names, _GROUP_NAMES)
        true_cobb = float(y[idx])
        severity = SEVERITY_LABELS[int(np.clip(np.digitize(true_cobb, _COBB_BOUNDARIES), 0, 3))]
        fig, ax = plt.subplots(figsize=(8, 5))
        render_waterfall(
            ax, contrib, expected_val, true_cobb, pred_val, sid, severity, label=label, fixed_order=_FIXED_ORDER
        )
        fig.tight_layout()
        out_dir = _OUT_DIR / "waterfalls" / severity
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"waterfall_{sid}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    logger.info(f"批量瀑布图: {_OUT_DIR / 'waterfalls'}/（{len(_select_subjects(pred_df))} 张）")


def main(run_analysis: bool = True) -> None:
    """v1.0.0 批量导出：indices 表 + 特征重要性（CI 分解）+ 批量瀑布图。

    Args:
        run_analysis: True 时调 ``analyze.main("v1.0.0")`` 输出与 v0.1.0 同口径的
            CI 反解特征重要性；False 表示编排方（export --scheme v1.0.0）已在步骤 1 输出。
    """
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    composite = _load_model_pkg()
    X, feature_names, y = _load_scheme()
    logger.info(f"v1.0.0 特征: {X.shape}, feature_names: {len(feature_names)}D")

    _indices_table()
    if run_analysis:
        from commands.export.analyze import main as analyze_main

        analyze_main("v1.0.0")  # CI 反解特征重要性（与 v0.1.0 完全同口径）
    _waterfalls(composite, X, y)
    logger.info(f"\nv1.0.0 导出完成: {_OUT_DIR}/")


if __name__ == "__main__":
    main()
