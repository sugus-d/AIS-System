"""训练结果路径与落盘 — Trainer 路径的共享壳层。

commands/pipeline.py::_run_train 与 modeling/train.py::_run_weighted 都走 Trainer
（新架构），结果目录签名、已存在跳过、metrics 落盘在此统一，
避免两入口各实现一份路径规则。

旧 cross_validate 路径（modeling/train.py 默认）不落盘到 results/modeling/prediction/，
仍写 all_results.json（旧评估入口），不受本模块影响。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modeling.contracts import TRAINING_RESULTS_DIR as RESULTS_DIR


def extra_para_signature(cfg: object, label: str = "") -> str:
    """结果目录签名：加权-校准-hp[-标签]。"""
    weight_components = getattr(cfg, "weight_components", None)
    w_name = f"wc{len(weight_components)}" if weight_components else "none"
    trainer_tag = getattr(cfg, "trainer", None) or "none"
    hp_n_iter = getattr(cfg, "hp_searcher_params", {}).get("n_iter", 40)
    suffix = f"-{label}" if label else ""
    return f"{w_name}-cal{trainer_tag}-hp{hp_n_iter}{suffix}"


def scheme_results_path(scheme_name: str) -> Path:
    """某特征方案的结果根目录（RESULTS_DIR / scheme_name）。"""
    return RESULTS_DIR / scheme_name


def metrics_json_path(
    feat_scheme_name: str,
    train_scheme: str,
    extra_para: str,
    model_name: str,
) -> Path:
    """模型结果 metrics.json 的落盘路径。"""
    return RESULTS_DIR / feat_scheme_name / f"{train_scheme}-{extra_para}" / model_name / "metrics.json"


def find_existing_metrics(
    feat_scheme_name: str,
    train_scheme: str,
    extra_para: str,
    model_name: str,
) -> dict | None:
    """结果已存在则读回 metrics.json，否则返回 None。"""
    path = metrics_json_path(feat_scheme_name, train_scheme, extra_para, model_name)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def save_results(
    model_name: str,
    preds: np.ndarray,
    cfg: object,
    train_scheme: str,
    extra_para: str,
    y_true: np.ndarray,
    subjects: np.ndarray | None = None,
    *,
    feat_scheme_name: str = "unknown",
    best_params: dict | None = None,
    details: dict | None = None,
) -> dict:
    """写 predictions.csv + metrics.json + config.json，返回 metrics。

    训练参数（best_params / calibration）随 config.json 存档，供复用分支
    （如 train_ensemble 复用 C7 OOF）读取，避免运行时重算与训练口径不一致
    （_bias_from_oof 曾因与 MarginTrainer 校准口径不同导致预测漂移）。

    Args:
        feat_scheme_name: 特征方案名，决定结果目录层级。
        best_params: HP 搜索结果（Trainer 的 result.best_params）。
        details: 训练详情（含 calibration.bias，Trainer 的 result.details）。
    """
    from modeling.metrics import compute_4class_metrics, SEVERITY_BINS, SEVERITY_LABELS

    out_dir = metrics_json_path(feat_scheme_name, train_scheme, extra_para, model_name).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_4class_metrics(y_true, preds)
    y_class = np.digitize(y_true, SEVERITY_BINS[1:-1])
    p_class = np.digitize(preds, SEVERITY_BINS[1:-1])
    pdf = pd.DataFrame(
        {
            "subject_id": subjects if subjects is not None else range(len(y_true)),
            "max_cobb_true": y_true,
            "max_cobb_pred": np.round(preds, 2),
            "class_true": [SEVERITY_LABELS[c] for c in y_class],
            "class_pred": [SEVERITY_LABELS[c] for c in p_class],
        }
    )
    pdf.to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    config = {
        "model": model_name,
        "train_scheme": train_scheme,
        "feature_scheme": feat_scheme_name,
        "extra_para": extra_para,
        "params": {
            "hp_n_iter": getattr(cfg, "hp_searcher_params", {}).get("n_iter", 40),
            "trainer": getattr(cfg, "trainer", None),
            "transform_target": getattr(cfg, "transform_target", True),
        },
        # 训练参数存档：复用分支读取，保证模型与评估口径一致
        "best_params": best_params,
        "calibration": (details or {}).get("calibration", {}),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    return metrics
