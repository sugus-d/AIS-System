"""训练模型持久化 — 全量重训最终模型并保存 joblib。

背景：Trainer 的 per-fold CV 只产出 metrics/predictions JSON，不保存可复用的
模型对象，导致缺少单 subject 推理能力。本模块在训练完成后，用全量数据 +
CV 最优超参重训最终模型，连同标准化 / target 变换 / 特征名一起 joblib 持久化，
供 :mod:`api.predict` 加载预测。

保存内容（joblib dict）：
- ``model``: 全量重训的 sklearn 估计器
- ``scaler``: 拟合在全量筛选特征上的 StandardScaler
- ``selector``: per-fold 特征选择器（固定方案下恒等，动态方案用于复现筛选）
- ``transform_target``: 是否对 target 做 piecewise log 变换
- ``feature_names``: 特征方案选出的列名（固定方案；None=per-fold 动态）
- ``scheme_name`` / ``model_name`` / ``best_params``: 元信息
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from modeling._shared import transform_target
from modeling.contracts import FeatureSet, TrainingConfig
from modeling.model_package import load_model_package
from modeling.models import REGISTRY as MODEL_REGISTRY
from modeling.training.feature_selector import PerFoldFeatureSelector
from utils.logger import logger

# 模型权重根目录（与 prediction 评估结果分开存放）
MODELS_DIR = Path("results/modeling/models")

# ci10/ci20 合成超参（与 features/selectors/scheme_morph_region_ci_35d 一致）
_COBB_LIGHT = 10
_COBB_CLINICAL = 20
_CI_MEASURES = ["normal_angle", "normal_vector_cos", "height", "mean_curv", "gauss_curv", "roughness"]
_REGION_CSV = Path("results/extraction/features_extraction/v0.1.0/region_asymmetry.csv")

# 4 个 CI 公式特征（v0.1.0 方案的 ci_final，训练时由 _build_one_ci 拟合）
_CI_FORMULA_GROUPS = [
    ("height", "dm"),
    ("mean_curv", "dm"),
    ("normal_angle", "pw"),
    ("normal_vector_cos", "pw"),
]


def _fit_ci_target(
    region_cols: list[str],
    Xr: np.ndarray,
    target: np.ndarray,
    C: float,
    thr: float,
) -> dict:
    """拟合单目标 CI 特征（ci10_normal/ci20_mild）的合成参数。

    复用 features.synthesis.CiTargetSynthesizer（与 _loaders_ci::_build_ci_for_target
    同逻辑的单点实现），返回可复现参数（保存后由 predict 合成）。
    """
    from features.synthesis import CiTargetSynthesizer

    return CiTargetSynthesizer().fit(region_cols, Xr, target, C, thr).to_params()


def _fit_ci_targets(feature_set: FeatureSet, feature_names: list[str] | None) -> dict:
    """若方案含 ci10_normal/ci20_mild，从训练 region 特征拟合合成参数。

    注意用 save_trained_model 的 feature_names 参数判断（feature_set.feature_names
    在 _train_one 里为空，不代表方案不含 CI）。
    """
    import pandas as pd

    needed = [n for n in ("ci10_normal", "ci20_mild") if n in (feature_names or [])]
    if not needed:
        return {}

    region_df = pd.read_csv(_REGION_CSV).dropna(subset=["max_cobb"])
    y = np.asarray(feature_set.y, dtype=np.float64)
    region_cols = [c for c in region_df.columns if c not in ("subject_id", "max_cobb")]
    all_cols: list[str] = []
    for measure in _CI_MEASURES:
        for method in ("dm", "pw"):
            suffix = f"_{measure}__pw" if method == "pw" else f"_{measure}"
            all_cols.extend(c for c in region_cols if c.endswith(suffix))
    Xr_all = region_df[all_cols].values.astype(float)

    out: dict = {}
    if "ci10_normal" in needed:
        yb10 = (y < _COBB_LIGHT).astype(float)
        out["ci10_params"] = _fit_ci_target(all_cols, Xr_all, yb10, C=0.1, thr=0.05)
    if "ci20_mild" in needed:
        yb20 = (y > _COBB_CLINICAL).astype(float)
        out["ci20_params"] = _fit_ci_target(all_cols, Xr_all, yb20, C=0.2, thr=0.1)
    return out


def _fit_ci_formula_params(feature_set: FeatureSet) -> dict:
    """拟合 4 个 CI 公式特征的合成参数（供 predict 单 subject 复现）。

    训练特征方案 v0.1.0 的 4 个 CI（height_dm/mean_curv_dm/
    normal_angle_pw/normal_vector_cos_pw）由 features.synthesis.CiFormulaSynthesizer
    用 **全 subject 标准化**（全局 mean/std）+ Lasso 拟合。alpha 用 0.5
    （预测复现历史口径；与训练方案 CI_ALPHA 的差异是已知现状，统一需重训）。

    Args:
        feature_set: 训练数据（保留签名，region 特征直接读训练 CSV）。

    Returns:
        {group: {"columns": 选中的 region 列, "mean": 全局均值,
                 "std": 全局标准差, "coef": Lasso 非零系数}}。
    """
    import pandas as pd

    from features.synthesis import CiFormulaSynthesizer

    region_df = pd.read_csv(_REGION_CSV).dropna(subset=["max_cobb"])
    y = region_df["max_cobb"].values.astype(float)
    return CiFormulaSynthesizer(groups=_CI_FORMULA_GROUPS).fit(region_df, y).to_params()


def _fit_asymmetry_formulas() -> dict:
    """拟合 5 个不对称指数的合成公式（复用 features.synthesis.AsymmetrySynthesizer）。

    论文表3 的 5 指数（Curvature/Height/NAI/RI + AI OLS 组合）由合成器用
    Lasso/Ridge 从 region 特征拟合（与 tables._compute_indices 同逻辑的单点实现），
    训练时保存参数，predict 对单 subject 复用公式计算。
    """
    import pandas as pd

    from features.synthesis import AsymmetrySynthesizer

    region_df = pd.read_csv(_REGION_CSV).dropna(subset=["max_cobb"])
    return AsymmetrySynthesizer().fit(region_df).to_params()


def _to_ci_formulas(ci_formula_params: dict, ci_targets: dict) -> dict:
    """统一 CI 公式格式 {ci_name: {"columns": [...], "coefs": [...]}}，训练时随模型包保存。

    从 ci_formula_params（4 个复合指数）+ ci10/ci20_params（单目标 Lasso）转换，
    供特征反解（features.utils.ci_decompose.decompose_ci_importance）直接使用，
    运行时无需读 CSV / 现场拟合。命名按各方案 feature_names 实际出现的 CI 名。
    """
    formulas: dict = {}
    for group, p in (ci_formula_params or {}).items():
        formulas[group] = {"columns": list(p["columns"]), "coefs": [float(c) for c in p["coef"]]}
    for key, name in (("ci10_params", "ci10_normal"), ("ci20_params", "ci20_mild")):
        p = ci_targets.get(key) or {}
        if not p:
            continue
        columns = p["columns"]
        nz = np.asarray(p["nz"], dtype=int)
        formulas[name] = {"columns": [columns[i] for i in nz], "coefs": [float(c) for c in p["coef"]]}
    return formulas


def save_trained_model(
    feature_set: FeatureSet,
    best_params: dict,
    model_name: str,
    scheme_name: str,
    config: TrainingConfig,
    feature_names: list[str] | None = None,
) -> Path:
    """用全量数据 + 最优超参重训最终模型并 joblib 保存。

    Args:
        feature_set: 训练数据（含全量 X / X_raw_blocks / y）。
        best_params: CV 搜索得到的最优超参。
        model_name: 模型名（MODEL_REGISTRY 的 key）。
        scheme_name: 特征方案名（保存目录用）。
        config: 训练配置（transform_target 等）。
        feature_names: 特征方案选出的列名（固定方案），None 表示 per-fold 动态。

    Returns:
        保存的 .joblib 文件路径。
    """
    X_raw = feature_set.X_raw_blocks or {"basic": feature_set.X}
    y = np.asarray(feature_set.y, dtype=np.float64)

    # 全量特征筛选（basic 全保留；morph/region 用 CV 学习的索引/CI 变换）
    selector = PerFoldFeatureSelector()
    X_sel = selector.fit_transform(X_raw, y, feature_set.region_column_names)

    # 标准化 + target 变换
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_sel)
    yt = transform_target(y) if config.transform_target else y.copy()

    # 全量重训最终模型（只传模型超参，过滤 sample_weight 等非模型参数）
    model_cls = MODEL_REGISTRY[model_name]
    model_hp = {
        k: v for k, v in (best_params or {}).items()
        if k in model_cls().get_param_space()
    }
    model = model_cls(params=model_hp)
    model.fit(X_s, yt)

    out_dir = MODELS_DIR / scheme_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model_name}.joblib"

    ci_targets = _fit_ci_targets(feature_set, feature_names)
    ci_formula_params = _fit_ci_formula_params(feature_set)
    # 瀑布图基线（C7 SHAP expected）：不支持 SHAP 的模型留 None，predict 现场算
    c7_expected_value = None
    try:
        import shap

        estimator = model._reg if hasattr(model, "_reg") else model  # noqa: SLF001  # 包装类内部回归器
        c7_expected_value = float(np.asarray(shap.TreeExplainer(estimator).expected_value).reshape(-1)[0])
    except Exception:  # noqa: BLE001  # 非 SHAP 模型（LR 等）跳过
        pass

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "selector": selector,
            "transform_target": config.transform_target,
            "feature_names": feature_names,
            "scheme_name": scheme_name,
            "model_name": model_name,
            "best_params": best_params,
            "n_samples": len(y),
            "y_mean": float(y.mean()),
            "c7_expected_value": c7_expected_value,
            **ci_targets,  # ci10/ci20 合成参数（如适用）
            "ci_formula_params": ci_formula_params,  # 4 CI 公式参数（predict 复现训练 CI）
            "ci_formulas": _to_ci_formulas(ci_formula_params, ci_targets),  # 统一 CI 公式（反解用）
            **_fit_asymmetry_formulas(),  # 5 个不对称指数公式（论文表3）
        },
        path,
    )
    # 落盘后 round-trip 自检：用共享加载器复读（含结构校验），保证训练写的包能被预测/导出直接消费
    load_model_package(str(path), use_cache=False)
    logger.info(f"最终模型已保存: {path}")
    return path
