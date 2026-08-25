"""v1.0.0 边界 Ensemble 模型包 — 落盘协议 + 训练阶段可读产物。

从 ensemble_boundary 拆出（等价重构）：模型包保存（save_boundary_model）与
训练阶段可读产物落盘（_dump_artifacts + JSON 序列化 + 特征重要性/示例瀑布图）。
composite 分量构建在 ensemble_boundary_components，特征构建在 ensemble_boundary_features。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from modeling.ensemble_boundary_components import (
    _PARAMS_JSON,
    _SCHEME,
    build_composite_components,
)
from modeling.ensemble_boundary_features import (
    _RANDOM_STATE,
    _TGT10,
    _TGT20,
    _THR10,
    _THR20,
    fit_boundary_classifiers,
)
from utils.logger import logger

# ── 训练阶段落盘产物（图表/JSON） ──
_CI_FEATURE_NAMES = ("height_dm", "mean_curv_pw", "normal_angle_pw", "normal_vector_cos_pw", "ci10_normal", "ci20_mild")
_GROUP_NAMES = ["Normal Angle", "Morph", "Clinical", "Curvature", "Height", "Roughness", "Other"]
_FIXED_ORDER = ("Normal Angle", "Curvature", "Height", "Roughness", "Morph", "Clinical", "Other")
_EXAMPLE_COBB = 25.0  # 示例瀑布图选 subject 的目标 Cobb 角
_IMPORTANCE_COLORS = {
    "Normal Angle": "#4A72A0",
    "Curvature": "#5A8C5A",
    "Roughness": "#C88A4A",
    "Height": "#D49A4A",
    "Morph": "#6B5B8A",
    "Clinical": "#999999",
}
_MODEL_ID = "v1.0.0"


def _scaler_to_json(scaler: StandardScaler) -> dict:
    """StandardScaler → JSON 可序列化 dict（mean/scale）。"""
    return {
        "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
    }


def _dump_artifacts(
    out_dir: Path,
    composite: dict,
    ai8_params: dict,
    ai8_lr: dict,
    ridge_params: dict,
    ridge_lr: dict,
    alpha: float,
    beta: float,
    perclass_alpha: tuple[float, ...],
    classifiers: dict,
    sids: list[str],
    y: np.ndarray,
    X_boundary: np.ndarray,
) -> None:
    """训练阶段落盘模型目录：4 个可读 JSON + 特征重要性 CSV/图 + 示例瀑布图。

    与 joblib 包同源（同一次拟合），供人工核查、论文制表与 export 批量复用，
    避免在 API/export 运行时临时重拟合。

    Args:
        out_dir: 模型保存目录（与 joblib 同目录）。
        composite: :func:`_build_composite_components` 返回的自包含分量。
        ai8_params/ridge_params: AI 公式参数（cols/coefs）。
        ai8_lr/ridge_lr: AI-LR 全量拟合参数（intercept/coef）。
        alpha/beta/perclass_alpha: blend 配置。
        classifiers: P(y>10)/P(y>20) 全量拟合分类器。
        sids: subject 列表（示例瀑布图标 subject id）。
        y: 真实 Cobb 角（示例瀑布图选代表 subject）。
        X_boundary: 边界分类器特征（前 len(feature_names) 列为模型输入）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = composite["feature_names"]

    # 1. asymmetry_formulas.json — 5 指数公式 + scaler + AI OLS 权重
    _write_json(
        out_dir / "asymmetry_formulas.json",
        {
            "scaler": _scaler_to_json(composite["asymmetry_scaler"]),
            "cols": composite["asymmetry_cols"],
            "formulas": composite["asymmetry_formulas"],
            "ai": composite["asymmetry_ai"],
        },
    )

    # 2. ci_formula_params.json — 4 CI 公式 + ci10/ci20 目标参数（scaler 转 JSON）
    ci_out = {"ci_formula_params": composite["ci_formula_params"]}
    for key in ("ci10_params", "ci20_params"):
        if key in composite:
            ci_out[key] = {**composite[key], "scaler": _scaler_to_json(composite[key]["scaler"])}
    _write_json(out_dir / "ci_formula_params.json", ci_out)

    # 3. ai_formulas.json — Lasso-8 / Ridge-267 公式 + LR
    _write_json(
        out_dir / "ai_formulas.json",
        {
            "ai8": {"cols": ai8_params["cols"], "coefs": ai8_params["coefs"], "lr": ai8_lr},
            "ridge": {"cols": ridge_params["cols"], "coefs": ridge_params["coefs"], "lr": ridge_lr},
        },
    )

    # 4. ensemble_config.json — α/β/钳制阈值/边界分类器系数
    clf_json: dict = {}
    for thr in ("p10", "p20"):
        pkg = classifiers[thr]
        clf_json[thr] = {
            "scaler": _scaler_to_json(pkg["scaler"]),
            "coef": np.asarray(pkg["clf"].coef_[0]).tolist(),
            "intercept": float(pkg["clf"].intercept_[0]),
        }
    _write_json(
        out_dir / "ensemble_config.json",
        {
            "model_id": _MODEL_ID,
            "kind": "v1.0.0_ridge_boundary_ensemble",
            "alpha_base": alpha,
            "perclass_alpha": list(perclass_alpha),
            "beta": beta,
            "clamps": {"thr20": _THR20, "thr10": _THR10, "tgt20": _TGT20, "tgt10": _TGT10},
            "boundary_classifiers": clf_json,
            "n_samples": len(y),
        },
    )

    # 5. feature_importance.csv + top15 图（permutation importance，HistGBRT 无 feature_importances_）
    model = composite["model"]
    estimator = model._reg if hasattr(model, "_reg") else model  # noqa: SLF001  # 包装类内部回归器
    X_model = np.asarray(X_boundary[:, : len(feature_names)], dtype=np.float64)
    if X_model.shape[0] == len(y):
        X_s = composite["scaler"].transform(X_model)
        importance = _permutation_importance(estimator, X_s, np.asarray(y, dtype=np.float64))
        _dump_feature_importance(out_dir, feature_names, importance)

        # 6. example_waterfall.png — 训练集代表 subject 的单 case SHAP 瀑布图
        _dump_example_waterfall(out_dir, composite, estimator, feature_names, sids, y, X_model)

    logger.info(f"训练阶段产物已落盘: {out_dir}（4 JSON + 特征重要性 + 示例瀑布图）")


def _jsonable(obj: object) -> object:
    """递归转换 numpy 标量/数组 → Python 原生（JSON 可序列化）。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _write_json(path: Path, payload: dict) -> None:
    """以缩进格式写 JSON（递归转换 ndarray → list）。"""
    import json

    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _permutation_importance(estimator: object, X_s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Permutation importance（neg_MAE，10 repeats，固定 seed）。

    HistGradientBoostingRegressor 无 feature_importances_ 属性，用 sklearn
    permutation_importance（与 commands/export/analyze.py 同口径）替代。
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        estimator,
        X_s,
        y,
        n_repeats=10,
        random_state=_RANDOM_STATE,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    return result.importances_mean


def _dump_feature_importance(out_dir: Path, feature_names: list[str], importance: np.ndarray) -> None:
    """特征重要性 CSV（对齐 v0.1.0 feature_importance_decomposed.csv 结构）+ Top15 图。"""
    from features.utils.ci_decompose import _assign_group
    from features.utils.ci_display import feature_display_name

    sorted_idx = np.argsort(-importance)
    total = float(importance.sum()) or 1.0
    rows = []
    for rank, idx in enumerate(sorted_idx, 1):
        name = feature_names[idx]
        rows.append(
            {
                "rank": rank,
                "feature": name,
                "importance": round(float(importance[idx]), 6),
                "share_pct": round(float(importance[idx]) / total * 100, 2),
                "group": _assign_group(name),
                "subgroup": "",
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "feature_importance.csv", index=False)

    # Top15 图（复用渲染层，编排在训练脚本内完成）
    top = rows[:15][::-1]
    vals = np.asarray([r["importance"] for r in top])
    labels = [feature_display_name(r["feature"]) for r in top]
    groups = [r["group"] for r in top]
    colors = [_IMPORTANCE_COLORS.get(g, "#999999") for g in groups]
    from matplotlib import pyplot as plt

    from visualization.feature_importance_panels import render_top15_barh

    fig, ax = plt.subplots(figsize=(7, 5))
    render_top15_barh(ax, vals, labels, colors, groups)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("  feature_importance.csv + feature_importance.png（Top15）")


def _dump_example_waterfall(
    out_dir: Path,
    composite: dict,
    estimator: object,
    feature_names: list[str],
    sids: list[str],
    y: np.ndarray,
    X_model: np.ndarray,
) -> None:
    """示例单 case 瀑布图（SHAP 分解 composite 分量，与 api _render_waterfall 同构）。"""
    import shap

    from features.utils.ci_decompose import decompose_waterfall
    from modeling.metrics import SEVERITY_LABELS
    from visualization.waterfall_panels import render_waterfall

    explainer = shap.TreeExplainer(estimator)
    X_s = composite["scaler"].transform(X_model)
    idx = int(np.argmin(np.abs(np.asarray(y) - _EXAMPLE_COBB)))
    shap_row = np.asarray(explainer.shap_values(X_s[idx : idx + 1])[0]).reshape(-1)
    expected_val = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    # decompose_waterfall 内部用 SHAP 可加性取真实模型输出（expected + sum(shap)）
    contrib, pred_val = decompose_waterfall(shap_row, expected_val, feature_names, _CI_FEATURE_NAMES, _GROUP_NAMES)
    true_cobb = float(np.asarray(y)[idx])
    severity = SEVERITY_LABELS[int(np.clip(np.digitize(true_cobb, (10.0, 20.0, 40.0)), 0, 3))]

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    render_waterfall(
        ax,
        contrib,
        expected_val,
        true_cobb,
        pred_val,
        sids[idx],
        severity,
        label="Prediction",
        fixed_order=_FIXED_ORDER,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "example_waterfall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  example_waterfall.png（subject {sids[idx]}, true={true_cobb:.1f}°）")


def save_boundary_model(
    sids: list[str],
    y: np.ndarray,
    ai8: np.ndarray,
    ai8_params: dict,
    ai_ridge: np.ndarray,
    ridge_params: dict,
    X_boundary: np.ndarray,
    alpha: float,
    beta: float,
    perclass_alpha: tuple[float, ...],
    out_dir: Path,
    params_json: str = _PARAMS_JSON,
) -> Path:
    """保存边界 Ensemble 模型包（joblib，供 predict 单 subject 复用）。

    包含：自包含 CompositeV7 分量（内嵌模型/scaler/校准 bias）、Lasso-8/Ridge-AI
    特征参数（cols/coefs）、两个 AI-LR 全量拟合、边界分类器、per-class α、β、钳制阈值。

    Args:
        sids: subject 列表。
        y: 真实 Cobb 角。
        ai8/ai_ridge: AI 特征向量（全量拟合 LR 用）。
        ai8_params/ridge_params: AI 公式参数（cols/coefs）。
        X_boundary: 边界分类器特征（30D + ai8 + ridge）。
        alpha/beta/perclass_alpha: blend 配置。
        out_dir: 保存目录。
        params_json: composite_v7 重训产物（best_params + calibration_bias）。
    """
    import joblib
    import shap

    from modeling.training.save_model import _to_ci_formulas

    composite = build_composite_components(_SCHEME, params_json)
    lr8 = LinearRegression().fit(ai8.reshape(-1, 1), y)
    lr_ridge = LinearRegression().fit(ai_ridge.reshape(-1, 1), y)
    classifiers = fit_boundary_classifiers(X_boundary, y)
    # 预测/分析运行时的派生数据一并落盘（瀑布图基线、数据集均值、统一 CI 公式）
    estimator = composite["model"]._reg if hasattr(composite["model"], "_reg") else composite["model"]  # noqa: SLF001  # 包装类内部回归器
    c7_expected_value = float(np.asarray(shap.TreeExplainer(estimator).expected_value).reshape(-1)[0])
    ci_formulas = _to_ci_formulas(composite.get("ci_formula_params") or {}, composite)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "boundary_ensemble_ridge.joblib"
    joblib.dump(
        {
            "kind": "v1.0.0_ridge_boundary_ensemble",
            **composite,  # composite 分量展平到顶层（与 v0.1.0 结构对齐，预测代码统一读顶层）
            "ai8_formula_params": ai8_params,
            "ai8_lr": {"intercept": float(lr8.intercept_), "coef": float(lr8.coef_[0])},
            "ridge_formula_params": ridge_params,
            "ridge_lr": {"intercept": float(lr_ridge.intercept_), "coef": float(lr_ridge.coef_[0])},
            "boundary_classifiers": classifiers,
            "alpha_base": alpha,
            "perclass_alpha": list(perclass_alpha),
            "beta": beta,
            "thr20": _THR20,
            "thr10": _THR10,
            "tgt20": _TGT20,
            "tgt10": _TGT10,
            "n_samples": len(y),
            "subjects": sids,
            "y_mean": float(y.mean()),
            "c7_expected_value": c7_expected_value,
            "ci_formulas": ci_formulas,
        },
        path,
    )
    logger.info(f"边界 Ensemble(ridge) 模型包已保存: {path}")

    # 训练阶段可读产物：JSON 公式 + 特征重要性 + 示例瀑布图（与 joblib 同源）
    _dump_artifacts(
        out_dir,
        composite,
        ai8_params,
        {"intercept": float(lr8.intercept_), "coef": float(lr8.coef_[0])},
        ridge_params,
        {"intercept": float(lr_ridge.intercept_), "coef": float(lr_ridge.coef_[0])},
        alpha,
        beta,
        perclass_alpha,
        classifiers,
        sids,
        y,
        X_boundary,
    )
    return path
