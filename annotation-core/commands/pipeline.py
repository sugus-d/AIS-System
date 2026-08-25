"""管道编排器 — 按配置执行各步骤。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG = {
    "steps": [
        # roi 已从默认管线移除 — 仅用于预标注实验，不适用于生产流程
        # 生产路径：commands/batch_process_all.py → 人工修正
        # 原 roi 步骤为 bfs 算法，角度阈值 15
        {"name": "feature_eng", "scheme": "morph_region_ci_40d"},
        {"name": "train", "model": "HistGBRT", "scheme": "baseline",
         "params": {"cv": 5, "thorough": 40}},
    ],
}


def load_config(path: str | Path | None = None) -> dict:
    """加载 YAML 配置，缺省返回默认配置。"""
    if path is None:
        return dict(DEFAULT_CONFIG)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with open(p) as f:
        return yaml.safe_load(f)


_DEFAULT_FEATURE_SCHEME = "morph_region_ci_40d"  # fallback only when no param given


def run(
    config_path: str | None = None,
    steps: list[str] | None = None,
    overrides: dict | None = None,
) -> dict:
    cfg = load_config(config_path)
    results: dict = {}
    step_configs = cfg["steps"]
    if steps:
        step_configs = [s for s in step_configs if s["name"] in steps]

    for step in step_configs:
        name = step["name"]
        merged = dict(step)
        if overrides and name in overrides:
            merged.update(overrides[name])
        results[name] = _run_step(name, merged)
    return results


def _run_step(name: str, step_cfg: dict) -> object:
    params = step_cfg.get("params", {}) or {}
    if name == "roi":
        return _run_roi(step_cfg["algo"], params)
    if name == "feature_eng":
        return _run_feature_eng(step_cfg["scheme"], params)
    if name == "train":
        models = step_cfg.get("model")
        if isinstance(models, str):
            models = [models]
        return _run_train(models or ["Ridge"], step_cfg.get("scheme"), params)
    raise ValueError(f"未知步骤: {name}")


def _run_roi(algo: str, params: dict) -> str:
    from mesh.roi.registry import get

    selector = get(algo)
    return f"ROI: {selector.name} ({selector.description})"


def _run_feature_eng(scheme: str, params: dict) -> str:
    from features.selectors import get_selector

    sel = get_selector(scheme)
    return f"Feature: {sel.name} ({sel.n_features}D)"




def _load_true_labels() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv("results/extraction/features_extraction/back_v1/basic.csv").dropna(subset=["max_cobb"])
    return df["subject_id"].values, df["max_cobb"].values.astype(float)


def _run_train(models: list[str], scheme: str | None, params: dict) -> dict:
    import warnings

    warnings.filterwarnings("ignore")
    from features.selectors.schemes import SELECTION_REGISTRY as SCHEME_REGISTRY
    from modeling.contracts import FeatureSet
    from modeling.training.result_paths import extra_para_signature, find_existing_metrics, save_results
    from modeling.training.schemes import get_scheme
    from modeling.training.trainer import Trainer as BaseTrainer
    from modeling.training.trainer_margin import MarginTrainer
    from utils.logger import logger

    train_scheme = scheme or "baseline"
    cfg = get_scheme(train_scheme)

    for key in (
        "data_splitter",
        "trainer",
        "hp_searcher",
        "transform_target",
        "weight_components",
        "search_weight_components",
        "search_data_splitter",
    ):
        if key in params:
            setattr(cfg, key, params[key])
    if "hp_n_iter" in params:
        cfg.hp_searcher_params["n_iter"] = int(params["hp_n_iter"])
    if "score_metric" in params:
        cfg.hp_searcher_params["score_metric"] = str(params["score_metric"])
    if "cv" in params:
        cfg.data_splitter_params["n_splits"] = int(params["cv"])

    # 路径签名
    extra_para = extra_para_signature(cfg, label=params.get("_label", ""))

    feat_scheme_name = params.get("feature_scheme", _DEFAULT_FEATURE_SCHEME)
    scheme_data = SCHEME_REGISTRY[feat_scheme_name].load()
    feature_set = FeatureSet(
        name=feat_scheme_name,
        y=scheme_data["y"],
        X=scheme_data.get("X_basic"),
        feature_names=scheme_data.get("feature_names", []) or [],
    )

    subjects, y_true = _load_true_labels()

    out: dict[str, dict] = {}
    for model_name in models:
        # 跳过已有结果
        m = find_existing_metrics(feat_scheme_name, train_scheme, extra_para, model_name)
        if m is not None:
            logger.info(f"  {model_name}: 结果已存在，跳过")
            out[model_name] = {
                "rmse": m.get("rmse", 0),
                "r": m.get("r", 0),
                "f1": m.get("f1_20", 0),
                "macro_f1": m["macro_f1"],
            }
            continue
        cfg.models = [model_name]
        tr = MarginTrainer(cfg) if cfg.trainer == "margin" else BaseTrainer(cfg)
        results = tr.train(feature_set)
        for r in results:
            preds = np.array([float(x) for x in r.predictions])
            metrics = save_results(
                r.model_name,
                preds,
                cfg,
                train_scheme,
                extra_para,
                y_true,
                subjects,
                feat_scheme_name=feat_scheme_name,
            )
            out[r.model_name] = {
                "rmse": round(r.metrics.get("rmse", 0), 4),
                "r": round(r.metrics.get("r", 0), 4),
                "f1": round(r.metrics.get("f1", 0), 4),
                "macro_f1": round(metrics["macro_f1"], 4),
            }
            logger.info(
                f"{r.model_name}  >>  "
                f"Macro-F1={metrics['macro_f1']:.4f}  "
                f"N={metrics['per_class']['Normal']['f1']:.3f}  "
                f"Se={metrics['per_class']['Severe']['f1']:.3f}  "
                f"RMSE={r.metrics.get('rmse', 0):.2f}"
            )
    return out
