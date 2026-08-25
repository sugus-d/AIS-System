"""特征装配与模型推理 — UV 参数化 → 全量特征 → CI 合成 → 模型预测 → 指数。

CI 合成必须用模型包保存的训练参数（`ci_formula_params` + `ci10/ci20_params`，
全 subject 标准化 + Lasso），不能用 `compute_ci` 的单行标准化（系统性偏差 ~3°）。
指数公式由训练时（save_model）拟合后存入模型包，运行时只做应用，不重拟合。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

from parameterization.pipeline import run_pipeline
from utils.constants import classify_cobb, SEVERITY_BINS

# 不对称指数对外全称（内部缩写 nai/ri/ai 仅存在于训练公式）
_INDEX_FULL_NAMES = {"nai": "normal_angle_index", "ri": "roughness_index", "ai": "asymmetric_index"}


def _run_parameterization(
    subject_id: str,
    roi_path: Path,
    landmarks_path: Path,
    out_dir: Path,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """UV 参数化，返回 (cut_mesh, uv_coords)。"""
    run_pipeline(
        subject_id,
        output_dir=str(out_dir / "param"),
        mesh_path=str(roi_path),
        landmarks_path=str(landmarks_path),
    )
    param_dir = out_dir / "param" / subject_id
    cut_mesh = o3d.io.read_triangle_mesh(str(param_dir / "mesh_cut.ply"))
    uv = np.load(param_dir / "uv_coords.npy")
    return cut_mesh, uv


def _compute_ci_target(feature_df: pd.DataFrame, params: dict) -> float:
    """用保存的合成参数计算单目标 CI 特征（ci10_normal / ci20_mild）。

    复用 features.synthesis.CiTargetSynthesizer（拟合/应用单点，缺列对齐含在内）。
    """
    from features.synthesis import CiTargetSynthesizer

    return float(CiTargetSynthesizer.from_params(params).transform(feature_df)[0])


def _add_ci_features(feature_df: pd.DataFrame, model_pkg: dict) -> pd.DataFrame:
    """合成特征方案所需的 CI 特征（formulas 复合指数 + ci10/ci20 单目标）。

    训练特征方案 v0.1.0 含 6 个合成 CI 特征（不在 extract_all
    的 raw 输出里）：4 个来自复合指数公式（results_compressed.csv），2 个
    来自训练时拟合的单目标 Logistic（参数已随模型保存）。合成逻辑复用
    features.synthesis（训练/预测同源单点），模型包参数即合成器序列化。
    """
    from features.synthesis import CiFormulaSynthesizer

    feature_names = model_pkg.get("feature_names") or []
    df = feature_df.copy()

    # 4 个 CI 公式特征（height_dm/mean_curv_dm/normal_angle_pw/normal_vector_cos_pw）：
    # 用训练时保存的拟合参数合成（全 subject 标准化 + Lasso，不能用 compute_ci
    # 的单行标准化，会系统性偏差 ~3°）。
    ci_params = model_pkg.get("ci_formula_params")
    if ci_params:
        ci_df = CiFormulaSynthesizer.from_params(ci_params).transform(df)
        for group in ci_df.columns:
            if group in feature_names:
                df[group] = ci_df[group]

    # 单目标 CI（训练时拟合，参数随模型保存）
    if "ci10_normal" in feature_names and "ci10_params" in model_pkg:
        df["ci10_normal"] = _compute_ci_target(df, model_pkg["ci10_params"])
    if "ci20_mild" in feature_names and "ci20_params" in model_pkg:
        df["ci20_mild"] = _compute_ci_target(df, model_pkg["ci20_params"])
    return df


def _prepare_feature_df(feature_df: pd.DataFrame, model_pkg: dict) -> pd.DataFrame:
    """合成 CI 特征 + 清洗 Gender（对齐训练分布），返回可喂模型的特征 DataFrame。"""
    df = _add_ci_features(feature_df, model_pkg)
    # Gender 在训练 basic.csv 中无区分度（全 0），字符串值转数值对齐训练分布
    if "Gender" in df.columns:
        try:
            df["Gender"] = df["Gender"].astype(float)
        except (ValueError, TypeError):
            df["Gender"] = 0.0
    return df


def _prepare_model_input(feature_df: pd.DataFrame, model_pkg: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """合成 CI 特征 + 选列 + 缺列检查 + scaler 变换，返回 (df, X, X_s)。

    df 供按列名二次取特征（瀑布图/指数）；X 为选列原始值（边界分类器输入）；
    X_s 为 scaler 变换后直接喂模型。
    """
    feature_names = model_pkg.get("feature_names")
    if not feature_names:
        raise ValueError("模型包 feature_names 为空（per-fold 动态方案暂不支持单模型预测）")
    df = _prepare_feature_df(feature_df, model_pkg)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"特征缺失 {len(missing)} 个: {missing[:5]}...（模型与特征提取不匹配）")
    X = df[feature_names].values.astype(np.float64)
    return df, X, model_pkg["scaler"].transform(X)


def _calibrate(preds: np.ndarray, bias: dict) -> np.ndarray:
    """Per-class 偏差校正（复用 modeling._shared.apply_calibration 单点）。

    bias 键可能为 int（v0.1.0 原样保存）或 str（JSON 序列化后，如 v1.0.0），
    两者都兼容——之前只查 int 导致旧模型包校准静默失效。
    """
    from modeling._shared import apply_calibration

    return apply_calibration(preds, bias)


def _predict_ensemble(feature_df: pd.DataFrame, model_pkg: dict) -> dict:
    """Ensemble 预测：0.6×CompositeV7(校准) + 0.4×AI-LR。

    CompositeV7 分量：按 feature_names 选列 → scaler → 加权 HistGBRT → per-class 校准。
    AI 分量：ai_formula 对单行 region 特征求值 → LR（intercept + coef×AI）。
    """
    from modeling.ensemble import build_ai_feature

    ensemble = model_pkg["ensemble"]
    _, _, X_s = _prepare_model_input(feature_df, model_pkg)
    c7 = np.asarray(model_pkg["model"].predict(X_s)).reshape(-1)
    c7 = _calibrate(c7, ensemble.get("calibration_bias") or {})
    ai = build_ai_feature(feature_df, ensemble["ai_formula"])
    ai_pred = ensemble["ai_lr_intercept"] + ensemble["ai_lr_coef"] * np.asarray(ai).reshape(-1)
    ens = ensemble["alpha"] * c7 + (1.0 - ensemble["alpha"]) * ai_pred
    cobb = float(np.clip(ens[0], 0.0, 90.0))
    return {"cobb": cobb, "severity": classify_cobb(cobb), "c7_pred": float(c7[0]), "ai_pred": float(ai_pred[0])}


def _linear_combo(feature_df: pd.DataFrame, params: dict) -> np.ndarray:
    """特征线性组合求值（AI 公式）：cols × coefs。

    复用 features.synthesis.eval_linear_formula（与 build_ai_feature 单点）。
    """
    from features.synthesis import eval_linear_formula

    return eval_linear_formula(feature_df, params["cols"], params["coefs"])


def _boundary_prob(clf_pkg: dict, X_boundary: np.ndarray) -> float:
    """边界分类器 P(y>thr) 概率（clf_pkg 含 scaler + clf，输入 30D+ai8+ridge）。"""
    X_s = clf_pkg["scaler"].transform(X_boundary)
    return float(clf_pkg["clf"].predict_proba(X_s)[0, 1])


def _predict_boundary_ensemble(feature_df: pd.DataFrame, model_pkg: dict) -> dict:
    """per-class α + Ridge-AI 边界 Ensemble 单 subject 预测。

    自包含边界 Ensemble 包（ridge_boundary_ensemble，v1.0.0）四步：
    1. CompositeV7 分量（模型 + 校准 bias，展平在顶层）→ c7
    2. Lasso-8 / Ridge-AI 公式 + LR → ai8_pred / ridge_pred
    3. per-class α blend + β 加权 → blend
    4. P(y>10)/P(y>20) 分类器边界钳制
    """
    _, X, X_s = _prepare_model_input(feature_df, model_pkg)
    c7 = np.asarray(model_pkg["model"].predict(X_s)).reshape(-1)
    c7 = _calibrate(c7, model_pkg.get("calibration_bias") or {})

    ai8 = _linear_combo(feature_df, model_pkg["ai8_formula_params"])
    ai8_pred = model_pkg["ai8_lr"]["intercept"] + model_pkg["ai8_lr"]["coef"] * ai8
    ai_ridge = _linear_combo(feature_df, model_pkg["ridge_formula_params"])
    ridge_pred = model_pkg["ridge_lr"]["intercept"] + model_pkg["ridge_lr"]["coef"] * ai_ridge

    alpha_base = model_pkg["alpha_base"]
    b0 = alpha_base * c7 + (1.0 - alpha_base) * ai8_pred
    pc = np.digitize(b0, SEVERITY_BINS[1:-1])  # [10, 20, 40]
    alphas = np.asarray(model_pkg["perclass_alpha"])[pc]
    pbase = alphas * c7 + (1.0 - alphas) * ai8_pred
    blend = model_pkg["beta"] * pbase + (1.0 - model_pkg["beta"]) * ridge_pred

    # 边界分类器输入 = 30D + ai8 + ridge（与训练 X_boundary 同构）
    X_boundary = np.column_stack([X, ai8, ai_ridge])
    p10 = _boundary_prob(model_pkg["boundary_classifiers"]["p10"], X_boundary)
    p20 = _boundary_prob(model_pkg["boundary_classifiers"]["p20"], X_boundary)

    cobb = float(np.clip(blend[0], 0.0, 90.0))
    if cobb < SEVERITY_BINS[2] and p20 > model_pkg["thr20"]:
        cobb = max(cobb, model_pkg["tgt20"])
    if SEVERITY_BINS[1] <= cobb < SEVERITY_BINS[2] and p10 < model_pkg["thr10"]:
        cobb = min(cobb, model_pkg["tgt10"])
    return {
        "cobb": cobb,
        "severity": classify_cobb(cobb),
        "c7_pred": float(c7[0]),
        "ai8_pred": float(ai8_pred[0]),
        "ridge_pred": float(ridge_pred[0]),
        "p20": float(p20),
    }


def _predict(feature_df: pd.DataFrame, model_pkg: dict) -> dict:
    """按 feature_names 选列 → scaler → model → inv_transform → cobb。"""
    if model_pkg.get("kind") == "ridge_boundary_ensemble":
        return _predict_boundary_ensemble(feature_df, model_pkg)
    if model_pkg.get("ensemble"):
        return _predict_ensemble(feature_df, model_pkg)
    from modeling._shared import inv_transform

    _, _, X_s = _prepare_model_input(feature_df, model_pkg)
    raw = np.asarray(model_pkg["model"].predict(X_s)).reshape(-1)
    cobb = inv_transform(raw) if model_pkg.get("transform_target") else raw
    cobb = float(np.clip(cobb[0], 0.0, 90.0))
    return {"cobb": cobb, "severity": classify_cobb(cobb)}


def _compute_indices(feature_df: pd.DataFrame, model_pkg: dict) -> dict[str, float]:
    """用模型包保存的 5 不对称指数公式计算（复用 features.synthesis 合成器）。

    公式由 save_model 训练时用 AsymmetrySynthesizer 拟合后存入模型包
    （与论文表3 tables._compute_indices 同逻辑单点），这里对单 subject
    应用公式。参数（asymmetry_*）在模型包顶层（v0.1.0/v1.0.0 结构统一）。
    """
    from features.synthesis import AsymmetrySynthesizer

    values = AsymmetrySynthesizer.from_params(model_pkg).transform(feature_df)
    out = {key: float(value[0]) for key, value in values.items()}
    # 对外键名展开全称（nai→normal_angle_index 等，对齐论文表3）
    return {_INDEX_FULL_NAMES.get(key, key): value for key, value in out.items()}
