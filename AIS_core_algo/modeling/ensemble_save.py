"""CompositeV7 ensemble 模型保存 — 加权全量重训 + AI 分量 + 派生数据落盘。

从 ensemble.py 拆出（等价重构）：模型保存协议独立成模块；
训练闭环（train_ensemble / reproduce_manuscript_ensemble）在 ensemble_train，
AI 特征层（build_ai_feature / _fit_ai_linear）在 ensemble。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from modeling.contracts import FeatureSet
from modeling.ensemble import _AI_FORMULA, _FEATURES_2700D, _fit_ai_linear
from utils.logger import logger


def save_composite_model(
    feature_set: FeatureSet,
    scheme_name: str,
    model_name: str = "HistGBRT",
    alpha: float = 0.6,
    weight_components: list | None = None,
    calibration_bias: dict | None = None,
    best_params: dict | None = None,
    sids: list[str] | None = None,
    region_csv: str = _FEATURES_2700D,
) -> Path:
    """全量重训 CompositeV7 加权模型 + AI 分量，保存 ensemble joblib（单 subject 预测用）。

    模型包含：
    - CompositeV7 分量：加权 HistGBRT（weight_components 过采样）+ per-class 校准 bias
    - AI 分量：ai_formula + LR 全量拟合（intercept + coef×AI）
    predict 单 subject 用 ``alpha×C7 + (1-alpha)×AI_pred`` 加权集成。

    Args:
        feature_set: 训练数据（X=X_basic + y + feature_names + X_raw_blocks['region']）。
        scheme_name: 特征方案名（保存目录）。
        model_name:  CompositeV7 分量模型名。
        alpha: CompositeV7 权重（0.6 = manuscript）。
        weight_components: composite_v7 的样本权重组件列表。
        calibration_bias: per-class 偏差校正（MarginTrainer OOF 训练 result.details 传入）。
        best_params: 训练超参（可 None 用模型默认）。

    Returns:
        保存的 .joblib 路径。
    """
    import joblib
    import shap
    from sklearn.preprocessing import StandardScaler

    from modeling.models import REGISTRY as MODEL_REGISTRY
    from modeling.training.save_model import (
        _fit_asymmetry_formulas,
        _fit_ci_formula_params,
        _fit_ci_targets,
        _to_ci_formulas,
        MODELS_DIR,
    )

    X = np.asarray(feature_set.X, dtype=np.float64)
    y = np.asarray(feature_set.y, dtype=np.float64)
    feature_names = feature_set.feature_names or []

    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    # CompositeV7 分量：加权全量重训（transform_target=False，与 composite_v7 方案一致）
    sample_weight = np.ones(len(y))
    for wc in weight_components or []:
        sample_weight *= wc.compute(y)
    model_cls = MODEL_REGISTRY[model_name]
    model_hp = {
        k: v for k, v in (best_params or {}).items()
        if k in model_cls().get_param_space()
    }
    model = model_cls(params=model_hp)
    model.external_weight = sample_weight
    model.fit(X_s, y)

    # AI 分量：ai_formula + LR 全量拟合（region 特征从训练 CSV 按 subject 顺序加载）
    formula = json.loads(Path(_AI_FORMULA).read_text(encoding="utf-8"))
    ai_params = _fit_ai_linear(sids or [], formula, region_csv=region_csv)

    # 预测/分析运行时的派生数据一并落盘（C7 瀑布图基线、数据集均值、subject 列表、统一 CI 公式）
    ci_targets = _fit_ci_targets(feature_set, feature_names)
    ci_formula_params = _fit_ci_formula_params(feature_set)
    estimator = model._reg if hasattr(model, "_reg") else model  # noqa: SLF001  # 包装类内部回归器
    c7_expected_value = float(np.asarray(shap.TreeExplainer(estimator).expected_value).reshape(-1)[0])

    out_dir = MODELS_DIR / scheme_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "CompositeV7.joblib"
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "transform_target": False,
            "feature_names": feature_names,
            "ensemble": {
                "alpha": alpha,
                "calibration_bias": calibration_bias or {},
                **ai_params,
            },
            "scheme_name": scheme_name,
            "model_name": model_name,
            "best_params": best_params or {},
            "n_samples": len(y),
            "subjects": list(sids or []),
            "y_mean": float(y.mean()),
            "c7_expected_value": c7_expected_value,
            **ci_targets,
            "ci_formula_params": ci_formula_params,
            "ci_formulas": _to_ci_formulas(ci_formula_params, ci_targets),
            **_fit_asymmetry_formulas(),
        },
        path,
    )
    logger.info(f"CompositeV7 ensemble 模型已保存: {path}")
    return path
