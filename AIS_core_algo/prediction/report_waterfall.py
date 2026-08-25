"""单 subject 特征贡献瀑布图（SHAP 分解 + 多分量加权合并）。

从 report.py 拆出（P2）：瀑布图依赖 shap/waterfall_panels，与热力图渲染解耦；
report.py 只保留 8 张报告图（4 热力图 + landmark + 背部光照 + 莫尔条纹）渲染。
"""

from __future__ import annotations

import numpy as np

from prediction.feature_pipeline import _calibrate, _linear_combo, _prepare_feature_df
from utils.constants import SEVERITY_BINS
from utils.logger import logger

# 瀑布图残差容差（cobb 与中间预测值之差小于此值视为已闭合，不画边界调整条）
_MIN_ADJUSTMENT = 1e-9


def _render_waterfall(
    feature_df: object,
    model_pkg: dict,
    out_dir,
    subject_id: str,
    severity: str,
    cobb: float,
) -> None:
    """单 subject 特征贡献瀑布图（多模型特征贡献加权合并，无 GT 虚线）。

    ensemble 各分量分别分解后按混合权重线性合并，特征贡献真正归到类别：
      - C7（HistGBRT）: SHAP 分解，CI 特征（height_dm/ci10/ci20）按生成公式
        反解回基础 region 特征再归组
      - AI8/AI（LR）: 线性分解（scale × coef_j × col_j）归组
      - Ridge（LR）: 同上
    组合基线 = 各分量基线×权重之和；唯一无法归因特征的是 C7 per-class 校准
    （作用于预测值而非特征），保留独立条。最终柱对齐真实预测 cobb，完全闭合。
    参考 commands/export/charts_waterfall.py 的批量瀑布图逻辑。
    """
    # CI 公式加载 + 反解统一在 features.utils.ci_decompose（训练时保存、运行时读取）
    from features.utils.ci_decompose import (
        _assign_group,
        decompose_ci_importance,
        linear_contrib_to_groups,
        load_ci_formulas_from_package,
    )
    from visualization.waterfall_panels import render_waterfall

    kind = model_pkg.get("kind")
    # 模型参数统一在顶层（v0.1.0/v1.0.0 结构一致，无 composite 子包）
    feature_names = model_pkg.get("feature_names")
    model = model_pkg.get("model")
    estimator = getattr(model, "_reg", model)
    try:
        import shap  # 延迟导入：shap 缺失或模型不支持时优雅跳过

        explainer = shap.TreeExplainer(estimator)
    except Exception:
        logger.warning("瀑布图：模型不支持 SHAP 分解，跳过")
        return

    df = _prepare_feature_df(feature_df, model_pkg)  # CI 特征需合成后才在 feature_names 里
    X = df[feature_names].values.astype(np.float64)
    X_s = model_pkg["scaler"].transform(X)
    shap_row = np.asarray(explainer.shap_values(X_s[:1])[0]).reshape(-1)
    # 优先用训练时保存的基线（口径稳定）；旧模型包无此字段时现场算
    expected_val = float(model_pkg.get("c7_expected_value") or np.asarray(explainer.expected_value).reshape(-1)[0])
    c7_raw = expected_val + float(shap_row.sum())

    # C7 贡献：非 CI 特征直接归组；CI 特征用 analyze 现成口径反解回基础特征再归组
    group_names = ["Normal Angle", "Morph", "Clinical", "Curvature", "Height", "Roughness", "Other"]
    fixed_order = ("Normal Angle", "Curvature", "Height", "Roughness", "Morph", "Clinical", "Other")
    ci_formulas = load_ci_formulas_from_package(model_pkg)
    ci_imp: dict[str, float] = {}
    contrib_c7: dict[str, float] = {g: 0.0 for g in group_names}
    other_c7 = 0.0
    for i, name in enumerate(feature_names):
        if name in ci_formulas:
            ci_imp[name] = float(shap_row[i])
            continue
        group = _assign_group(name)
        if group in contrib_c7:
            contrib_c7[group] += shap_row[i]
        else:
            other_c7 += shap_row[i]
    decomposed = decompose_ci_importance(ci_imp, ci_formulas)
    for feat, share in decomposed.items():
        group = _assign_group(feat)
        if group in contrib_c7:
            contrib_c7[group] += share
        else:
            other_c7 += share
    if abs(other_c7) > _MIN_ADJUSTMENT:
        contrib_c7["Other"] = contrib_c7.get("Other", 0.0) + other_c7
    contrib_c7 = {k: v for k, v in contrib_c7.items() if abs(v) > _MIN_ADJUSTMENT}

    # ── 各分量分解 + 混合权重（从 C7 raw 到真实 cobb）──────────────────
    bias = model_pkg.get("calibration_bias") or (model_pkg.get("ensemble") or {}).get("calibration_bias") or {}
    c7_cal = float(_calibrate(np.array([c7_raw]), bias)[0])

    def _linear_baseline(
        cols: list[str], coefs: list[float], lr: dict, means: dict | None, formula_intercept: float = 0.0
    ) -> float:
        """线性分量基线 = LR 截距 + LR 系数 × mean(combo)。

        combo = formula_intercept + Σ coef_j×col_j；mean(combo) 用训练集列均值（means）
        计算，等于分量在训练集上的平均输出（SHAP expected-value 口径）。
        means 为 None（旧模型包未保存均值）时退化为 formula_intercept 处基线。
        """
        combo_mean = formula_intercept
        if means is not None:
            combo_mean += sum(c * means.get(col, 0.0) for c, col in zip(coefs, cols, strict=True))
        return lr["intercept"] + lr["coef"] * combo_mean

    if kind == "ridge_boundary_ensemble":
        # v1.0.0：C7 + AI8（per-class α）+ Ridge（β）
        ai8 = _linear_combo(feature_df, model_pkg["ai8_formula_params"])
        ai8_pred = float(model_pkg["ai8_lr"]["intercept"] + model_pkg["ai8_lr"]["coef"] * ai8[0])
        b0 = model_pkg["alpha_base"] * c7_cal + (1.0 - model_pkg["alpha_base"]) * ai8_pred
        pc = int(np.digitize(b0, SEVERITY_BINS[1:-1]))
        alpha = float(np.asarray(model_pkg["perclass_alpha"])[pc])
        beta = model_pkg["beta"]
        w_c7, w_ai8, w_ridge = beta * alpha, beta * (1.0 - alpha), 1.0 - beta
        ai8_means = model_pkg["ai8_formula_params"].get("mean")
        ridge_means = model_pkg["ridge_formula_params"].get("mean")
        contrib_ai8 = linear_contrib_to_groups(
            model_pkg["ai8_formula_params"]["cols"],
            model_pkg["ai8_formula_params"]["coefs"],
            df,
            group_names,
            model_pkg["ai8_lr"]["coef"],
            feature_means=ai8_means,
        )
        contrib_ridge = linear_contrib_to_groups(
            model_pkg["ridge_formula_params"]["cols"],
            model_pkg["ridge_formula_params"]["coefs"],
            df,
            group_names,
            model_pkg["ridge_lr"]["coef"],
            feature_means=ridge_means,
        )
        base = (
            w_c7 * expected_val
            + w_ai8
            * _linear_baseline(
                model_pkg["ai8_formula_params"]["cols"],
                model_pkg["ai8_formula_params"]["coefs"],
                model_pkg["ai8_lr"],
                ai8_means,
            )
            + w_ridge
            * _linear_baseline(
                model_pkg["ridge_formula_params"]["cols"],
                model_pkg["ridge_formula_params"]["coefs"],
                model_pkg["ridge_lr"],
                ridge_means,
            )
        )
        contrib_extra = {g: w_ai8 * v for g, v in contrib_ai8.items()}
        for g, v in contrib_ridge.items():
            contrib_extra[g] = contrib_extra.get(g, 0.0) + w_ridge * v
    elif model_pkg.get("ensemble"):
        # v0.1.0：α×C7 + (1−α)×AI-LR（AI 公式含 intercept，并入基线）
        ensemble = model_pkg["ensemble"]
        w_c7, w_ai = ensemble["alpha"], 1.0 - ensemble["alpha"]
        ai_formula = ensemble["ai_formula"]
        ai_means = ai_formula.get("mean")
        contrib_ai = linear_contrib_to_groups(
            ai_formula["feats"],
            ai_formula["coefs"],
            df,
            group_names,
            ensemble["ai_lr_coef"],
            feature_means=ai_means,
        )
        base = w_c7 * expected_val + w_ai * _linear_baseline(
            ai_formula["feats"],
            ai_formula["coefs"],
            {"intercept": ensemble["ai_lr_intercept"], "coef": ensemble["ai_lr_coef"]},
            ai_means,
            float(ai_formula["intercept"]),
        )
        contrib_extra = {g: w_ai * v for g, v in contrib_ai.items()}
    else:
        # 单模型：仅 C7
        w_c7, base, contrib_extra = 1.0, expected_val, {}

    # C7 偏置修正（per-class 常数，作用在预测值上，无法归因特征，独立条）+ 边界钳制残差
    calibration = w_c7 * (c7_cal - c7_raw)
    merged: dict[str, float] = {}
    for g, v in contrib_c7.items():
        merged[g] = merged.get(g, 0.0) + w_c7 * v
    for g, v in contrib_extra.items():
        merged[g] = merged.get(g, 0.0) + v
    residual = cobb - (base + calibration + sum(merged.values()))
    if abs(calibration) > _MIN_ADJUSTMENT:
        merged["Calibration"] = calibration
    if abs(residual) > _MIN_ADJUSTMENT:
        merged["Boundary Clamp"] = residual
    extra_order = tuple(k for k in ("Calibration", "Boundary Clamp") if k in merged)
    fixed_order = fixed_order + extra_order
    # 预测/推理场景无 ground truth：不画 GT 虚线（render_waterfall true_cobb=None）

    report = out_dir / "report"
    report.mkdir(parents=True, exist_ok=True)
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    from visualization._render_utils import save_img
    from visualization._style import ACADEMIC_STYLE

    # 统一 academic 样式（rc_context 局部生效）
    with mpl.rc_context(ACADEMIC_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        render_waterfall(
            ax,
            merged,
            base,  # 组合基线（各分量基线×权重）
            None,  # 预测场景无 GT
            cobb,  # 最终柱 = 真实预测（与 prediction.json 一致）
            subject_id,
            severity,
            label="Prediction",
            fixed_order=fixed_order,
        )
        fig.tight_layout()
        path = report / "waterfall.png"
        save_img(fig, str(path), dpi=150)
        logger.info(f"报告图已保存: {path}")
