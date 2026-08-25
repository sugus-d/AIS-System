"""v1.0.0 边界 Ensemble CompositeV7 分量构建。

从 ensemble_boundary_artifacts 拆出（等价重构）：自包含 CompositeV7 分量
（加权模型 + scaler + feature_names + 校准 bias）与 manual 方案重拟合的
CI 合成参数/不对称指数公式。落盘协议在 ensemble_boundary_artifacts。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from features.selectors.schemes import SELECTION_REGISTRY
from modeling.ensemble_boundary_features import _COBB_10, _COBB_20
from utils.paths import MODELS_DIR

_REGION_CSV = "results/extraction/features_extraction/v1.0.0/region_asymmetry.csv"
_SCHEME = "v1.0.0"
# composite_v7 重训产物（best_params + calibration_bias，供单 subject 预测包内嵌 composite 模型）
# 与 CompositeV7.joblib 同目录（v1.0.0 模型产物）；2026-08-16 从 archive/manual_roi_search 提升
_PARAMS_JSON = str(MODELS_DIR / "v1.0.0" / "composite_c7_params.json")


def _fit_manual_ci_params(feature_names: list[str]) -> dict:
    """拟合 manual 方案的 CI 合成参数（4 CI 公式 + ci10/ci20）。

    save_model._fit_ci_formula_params 硬编码 v0.1.0 的 CI 组名（mean_curv_dm）和
    v0.1.0 region CSV；manual 方案选了 mean_curv_pw，且 region 源为 manual。
    复用 features.synthesis 合成器（与 v0.1.0 同逻辑单点）：CI 公式 alpha=0.5、
    ci10/ci20 去高相关 0.95（manual 历史口径）。

    Returns:
        {"ci_formula_params": {...}, "ci10_params": {...}, "ci20_params": {...}}。
    """
    from features.synthesis import CiFormulaSynthesizer, CiTargetSynthesizer

    region_df = pd.read_csv(_REGION_CSV).dropna(subset=["max_cobb"])
    y = region_df["max_cobb"].values.astype(float)
    region_cols = [c for c in region_df.columns if c not in ("subject_id", "max_cobb")]

    # 1. 4 个 CI 公式特征（按方案实际选中的 dm/pw 组）
    ci_groups = [
        n for n in feature_names if n in ("height_dm", "mean_curv_pw", "normal_angle_pw", "normal_vector_cos_pw")
    ]
    formula_groups = [(group.rsplit("_", 1)[0], group.rsplit("_", 1)[1]) for group in ci_groups]
    ci_formula = CiFormulaSynthesizer(groups=formula_groups).fit(region_df, y).to_params()

    # 2. ci10_normal / ci20_mild 目标参数（Logistic，0.95 去高相关）
    all_cols: list[str] = []
    for measure in ("normal_angle", "normal_vector_cos", "height", "mean_curv", "gauss_curv", "roughness"):
        for method in ("dm", "pw"):
            suffix = f"_{measure}__pw" if method == "pw" else f"_{measure}"
            all_cols.extend(c for c in region_cols if c.endswith(suffix))
    Xr_all = region_df[all_cols].values.astype(float)
    ci_targets: dict = {}
    if "ci10_normal" in feature_names:
        yb10 = (y < _COBB_10).astype(float)
        ci_targets["ci10_params"] = CiTargetSynthesizer().fit(all_cols, Xr_all, yb10, C=0.1, thr=0.05, corr_threshold=0.95).to_params()
    if "ci20_mild" in feature_names:
        yb20 = (y > _COBB_20).astype(float)
        ci_targets["ci20_params"] = CiTargetSynthesizer().fit(all_cols, Xr_all, yb20, C=0.2, thr=0.1, corr_threshold=0.95).to_params()
    return {"ci_formula_params": ci_formula, **ci_targets}


def _fit_manual_asymmetry_formulas() -> dict:
    """拟合 manual 方案的 5 不对称指数公式（复用 features.synthesis.AsymmetrySynthesizer）。

    save_model._fit_asymmetry_formulas 用 v0.1.0 region CSV；manual 方案用 manual
    region CSV 重拟合（curvature/height/nai/ri 各用 Lasso→Ridge 选特征拟合）。
    AI = OLS 组合（论文表3 固定权重，与 v0.1.0 一致）。

    Returns:
        {"asymmetry_scaler", "asymmetry_cols", "asymmetry_formulas", "asymmetry_ai"}。
    """
    from features.synthesis import AsymmetrySynthesizer

    region_df = pd.read_csv(_REGION_CSV).dropna(subset=["max_cobb"])
    return AsymmetrySynthesizer().fit(region_df).to_params()


def build_composite_components(scheme_name: str, params_json: str, model_name: str = "HistGBRT") -> dict:
    """构建自包含 CompositeV7 分量（模型/scaler/feature_names/校准 bias）。

    与 modeling.ensemble.save_composite_model 同口径（margin 权重 + best_params），
    供边界 Ensemble 单 subject 预测包内嵌。best_params/calibration_bias 来自
    composite_v7 重训（models/v1.0.0/composite_c7_params.json）。

    Args:
        scheme_name: 特征方案名。
        params_json: best_params + calibration_bias 的 JSON 路径。
        model_name: CompositeV7 分量模型名。

    Returns:
        {"model": ..., "scaler": ..., "feature_names": [...], "calibration_bias": {...},
         "transform_target": False}。
    """
    import json

    from modeling.models import REGISTRY as MODEL_REGISTRY
    from modeling.training.weights import DecayWeight, InvFreqWeight, MarginBoostWeight

    params = json.loads(Path(params_json).read_text(encoding="utf-8"))
    best_params = params["best_params"]
    calibration_bias = params["calibration_bias"]

    scheme_data = SELECTION_REGISTRY[scheme_name].load()
    X = np.asarray(scheme_data["X_basic"], dtype=np.float64)
    y = np.asarray(scheme_data["y"], dtype=np.float64)
    feature_names = scheme_data.get("feature_names") or []
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    # 全新权重对象（避免共享 scheme 权重被 searcher 污染），与 composite_v7 方案配置一致
    weight_components = [
        InvFreqWeight(max_ratio=3.0, normalize=False),
        MarginBoostWeight(normalize=False),
        DecayWeight(clinical=10, class_weight=2.0, dist_k=0.1, normalize=False, threshold=False),
    ]
    sample_weight = np.ones(len(y))
    for wc in weight_components:
        sample_weight *= wc.compute(y)
    model_cls = MODEL_REGISTRY[model_name]
    model_hp = {k: v for k, v in best_params.items() if k in model_cls().get_param_space()}
    model = model_cls(params=model_hp)
    model.external_weight = sample_weight
    model.fit(X_s, y)

    # CI 合成参数 + 5 不对称指数公式——按 manual 方案重拟合，
    # 供 api _prepare_feature_df / _compute_indices 单 subject 复现
    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "calibration_bias": calibration_bias,
        "transform_target": False,
        **_fit_manual_ci_params(feature_names),
        **_fit_manual_asymmetry_formulas(),
    }
