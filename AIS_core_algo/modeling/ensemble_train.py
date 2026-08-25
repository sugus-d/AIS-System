"""manuscript ensemble 训练闭环 — 重建/训练 0.6×CompositeV7 + 0.4×AI-LR 并落盘。

从 ensemble.py 拆出（等价重构）：训练编排（reproduce_manuscript_ensemble /
train_ensemble）独立成模块；AI 特征与预测层（build_ai_feature /
fit_ai_linear_oof / _load_ai_feature）在 modeling.ensemble，
模型保存（save_composite_model）在 modeling.ensemble_save。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.contracts import TrainingConfig
from modeling.ensemble import (
    _COMPOSITE_V7_PRED,
    _FEATURES_2700D,
    _load_ai_feature,
    _SCHEME,
    build_ensemble_preds,
    fit_ai_linear_oof,
)
from modeling.ensemble_save import save_composite_model
from modeling.training.result_paths import metrics_json_path, save_results
from utils.logger import logger
from utils.paths import FEATURES_EXTRACTION_DIR


def reproduce_manuscript_ensemble(
    alpha: float = 0.6,
    out_scheme: str = _SCHEME,
    label: str = "ai60-lroof",
) -> Path:
    """用现有 composite_v7 预测 + ai_formula 重建 manuscript ensemble 并落盘。

    纯计算秒级完成，验证当前数据/代码能否复现 MF1=0.724 / MAE=4.53°。

    Args:
        alpha: CompositeV7 权重（0.6 = manuscript）。
        out_scheme: 落盘结果的特征方案目录。
        label: 结果目录签名后缀。

    Returns:
        落盘的 metrics.json 路径。
    """
    c7_df = pd.read_csv(_COMPOSITE_V7_PRED)
    sids = c7_df["subject_id"].tolist()
    ai, y, _ = _load_ai_feature(sids)
    ai_pred = fit_ai_linear_oof(ai, y)
    ensemble_preds = build_ensemble_preds(c7_df["max_cobb_pred"].values.astype(float), ai_pred, alpha)

    cfg = TrainingConfig(
        models=["Ensemble"],
        hp_searcher_params={"n_iter": 5},
        trainer=None,
        transform_target=False,
    )
    metrics = save_results(
        "Ensemble", ensemble_preds, cfg, "ensemble", label,
        y, sids, feat_scheme_name=out_scheme,
    )
    mae = float(np.abs(ensemble_preds - y).mean())
    logger.info(
        f"ensemble 重建: MF1={metrics['macro_f1']:.4f} MAE={mae:.2f}° "
        f"(目标 MF1=0.724 MAE=4.53°)"
    )
    return metrics_json_path(out_scheme, "ensemble", label, "Ensemble")


def train_ensemble(
    scheme_name: str = _SCHEME,
    model_name: str = "HistGBRT",
    alpha: float = 0.6,
    hp_n_iter: int = 5,
    force_retrain: bool = False,
    label: str = "ai60-lroof",
) -> Path:
    """完整闭环：CompositeV7（训练或复用现有结果）→ AI-LR OOF → 加权集成 → 落盘。

    1. CompositeV7 OOF 预测：优先复用现有结果（find_existing_metrics），
       否则用 composite_v7 训练方案 + MarginTrainer 重训（耗时）。
    2. AI 特征（ai_formula）→ AI-LR OOF。
    3. 加权集成 → save_results 落盘（与阶段 1 同格式）。

    Args:
        scheme_name: 特征方案名。
        model_name:  CompositeV7 的模型名。
        alpha: CompositeV7 权重。
        hp_n_iter: composite_v7 训练 HP 搜索次数（重训时生效）。
        force_retrain: 强制重训 CompositeV7（忽略已有结果）。
        label: ensemble 落盘签名后缀。

    Returns:
        落盘的 metrics.json 路径。
    """
    from modeling.training.result_paths import (
        extra_para_signature,
        find_existing_metrics,
    )
    from modeling.training.schemes import get_scheme

    train_cfg = get_scheme("composite_v7")
    train_cfg.hp_searcher_params["n_iter"] = hp_n_iter
    extra = extra_para_signature(train_cfg, label="")
    # 候选复用签名：优先当前配置签名，其次历史 composite_v7 结果
    # （hp100 + composite_v7_stability，manuscript 主分量）
    candidates = [extra, "wc3-calmargin-hp100-composite_v7_stability"]

    # 1. CompositeV7 OOF 预测（复用或训练）
    from features.selectors.schemes import SELECTION_REGISTRY
    from modeling.contracts import FeatureSet

    scheme = SELECTION_REGISTRY[scheme_name]
    scheme_data = scheme.load()
    feature_set = FeatureSet(
        name=scheme_name,
        y=scheme_data["y"].astype(float),
        X=scheme_data["X_basic"],
        feature_names=scheme_data.get("feature_names") or [],
    )
    y_full = np.asarray(feature_set.y, dtype=np.float64)

    # 数据源目录跟随方案（v0.1.0=算法 ROI，v1.0.0=人工 ROI）
    region_sources = [p for p in scheme.source_files if p.endswith("region_asymmetry.csv")]
    if region_sources:
        source_dir = Path(region_sources[0]).parent
        basic_csv = source_dir / "basic.csv"
        ai_region_csv = region_sources[0]
    else:
        basic_csv = FEATURES_EXTRACTION_DIR / "v0.1.0" / "basic.csv"
        ai_region_csv = _FEATURES_2700D

    existing = None
    for cand in candidates:
        existing = find_existing_metrics(scheme_name, "composite_v7", cand, model_name)
        if existing is not None:
            extra = cand
            break
    if existing is not None and not force_retrain:
        c7_dir = metrics_json_path(scheme_name, "composite_v7", extra, model_name).parent
        c7_df = pd.read_csv(c7_dir / "predictions.csv")
        sids = c7_df["subject_id"].tolist()
        y = c7_df["max_cobb_true"].values.astype(float)
        c7_pred = c7_df["max_cobb_pred"].values.astype(float)
        # 优先读训练时存档的校准 bias（与 MarginTrainer 口径一致）；旧产物无存档时
        # 退化为 _bias_from_oof 近似并警告（该近似与训练校准有偏差，会导致预测漂移）
        saved_cfg = c7_dir / "config.json"
        calibration_bias = None
        if saved_cfg.exists():
            saved_bias = json.loads(saved_cfg.read_text(encoding="utf-8")).get("calibration", {}).get("bias") or {}
            calibration_bias = {int(k): v for k, v in saved_bias.items()}  # JSON 键为 str，转回 int
        if not calibration_bias:
            calibration_bias = _bias_from_oof(y, c7_pred, [0, 10, 20, 40, np.inf])
            logger.warning(f"复用 {model_name} 无校准存档，用 _bias_from_oof 近似（可能与训练口径有偏差）")
        best_params = None
        logger.info(f"复用 CompositeV7 结果: {model_name} (MF1={existing['macro_f1']:.4f}, 校准={calibration_bias})")
    else:
        from modeling.training.trainer_margin import MarginTrainer

        sids = pd.read_csv(basic_csv).dropna(subset=["max_cobb"])["subject_id"].tolist()
        y = y_full
        train_cfg.models = [model_name]
        result = MarginTrainer(train_cfg).train(feature_set)[0]
        c7_pred = np.array([float(x) for x in result.predictions])
        calibration_bias = (result.details or {}).get("calibration", {}).get("bias")
        best_params = result.best_params
        save_results(
            model_name, c7_pred, train_cfg, "composite_v7", extra, y, sids,
            feat_scheme_name=scheme_name, best_params=best_params, details=result.details,
        )
        logger.info(f"CompositeV7 训练完成: {model_name}")

    # 保存 CompositeV7 单 subject 预测模型（加权模型 + 校准 bias + AI-LR 分量）
    save_composite_model(
        feature_set, scheme_name, model_name, alpha,
        train_cfg.weight_components, calibration_bias or {}, best_params, sids,
        region_csv=ai_region_csv,
    )

    # 2. AI 特征 + AI-LR OOF（按 CompositeV7 subject 顺序对齐）
    ai, y_aligned, aligned_sids = _load_ai_feature(sids, region_csv=ai_region_csv)
    if not np.array_equal(np.asarray(aligned_sids), np.asarray(sids)):
        raise ValueError("AI 特征 subject 顺序与 CompositeV7 不一致")
    ai_pred = fit_ai_linear_oof(ai, y_aligned)

    # 3. 集成 + 落盘
    ensemble_preds = build_ensemble_preds(c7_pred, ai_pred, alpha)
    cfg = TrainingConfig(
        models=["Ensemble"],
        hp_searcher_params={"n_iter": hp_n_iter},
        trainer=None,
        transform_target=False,
    )
    metrics = save_results(
        "Ensemble", ensemble_preds, cfg, "ensemble", label,
        y_aligned, sids, feat_scheme_name=scheme_name,
    )
    mae = float(np.abs(ensemble_preds - y_aligned).mean())
    logger.info(
        f"ensemble 闭环: MF1={metrics['macro_f1']:.4f} MAE={mae:.2f}° "
        f"(α={alpha}, 目标 MF1=0.724 MAE=4.53°)"
    )
    return metrics_json_path(scheme_name, "ensemble", label, "Ensemble")


def _bias_from_oof(y_true: np.ndarray, y_pred: np.ndarray, severity_bins: list) -> dict:
    """从 OOF 预测计算 per-class median 偏差（MarginTrainer 校准口径的近似）。

    复用已有 CompositeV7 OOF predictions.csv 时，无 result.details 的校准 bias，
    用此函数从 (true, pred) 重算（与 MarginTrainer 的 per-class median 一致）。
    """
    bias: dict[int, float] = {}
    for c in range(4):
        lo, hi = severity_bins[c], severity_bins[c + 1]
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 0:
            bias[c] = float(np.median(y_pred[mask] - y_true[mask]))
    return bias
