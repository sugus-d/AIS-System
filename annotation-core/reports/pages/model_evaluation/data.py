"""模型评估数据加载。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st

from modeling.training.result_paths import RESULTS_DIR, scheme_results_path

OUT_DIR = RESULTS_DIR


def _best_result_file(algo: str) -> Path | None:
    """返回模型的最佳结果文件（按 F1 取最高值）。"""
    best, best_f1 = None, -1
    for rnd in [1, 2, 3]:
        p = OUT_DIR / f"round{rnd}_{algo}.json"
        if not p.exists():
            continue
        try:
            with open(p) as fp:
                d = json.load(fp)
            if d and d.get("f1", 0) > best_f1:
                best, best_f1 = p, d["f1"]
        except (json.JSONDecodeError, KeyError):
            continue
    return best


@st.cache_data
def load_best_results() -> list[dict]:
    """加载所有模型的最佳结果。"""
    from modeling.models import REGISTRY

    results = []
    for name in REGISTRY:
        if name.startswith("Ensemble"):
            continue
        fp = _best_result_file(name)
        if fp is None:
            continue
        with open(fp) as f:
            d = json.load(f)
        if d and "f1" in d:
            d["_source"] = fp.stem
            results.append(d)

    he = OUT_DIR / "round3_HuberEpsilon.json"
    if he.exists():
        with open(he) as f:
            d = json.load(f)
        d["algo"] = "HuberEpsilon"
        d["_source"] = "round3_HuberEpsilon"
        results.append(d)

    results.sort(key=lambda r: r.get("f1", 0), reverse=True)
    return results


@st.cache_data
def load_scheme_results(filename: str) -> list[dict]:
    fp = OUT_DIR / filename
    if not fp.exists():
        return []
    with open(fp) as f:
        data = json.load(f)
    data = [r for r in data if "f1" in r] if isinstance(data, list) else [data] if "f1" in data else []
    data.sort(key=lambda r: r.get("f1", 0), reverse=True)
    return data


@st.cache_data
def load_ensemble_results() -> list[dict]:
    fp = OUT_DIR / "ensemble_results.json"
    return json.loads(fp.read_text()) if fp.exists() else []


@st.cache_data
def load_y_true() -> np.ndarray:
    try:
        from features.loaders import load_scheme_b_features
        return load_scheme_b_features()["max_cobb"].values.astype(float)
    except Exception:
        return np.array([])


@st.cache_data
def load_latest_schemes(base_path: str | Path | None = None) -> dict[str, list[dict]]:
    """加载指定目录下所有方案的最新结果（直接从 metrics.json 读 4 分类指标）。

    Args:
        base_path: 方案根目录，默认 results/modeling/prediction/train_back_v1
    """
    base = Path(base_path) if base_path is not None else scheme_results_path("train_back_v1")
    if not base.exists():
        return {}

    schemes: dict[str, list[dict]] = {}
    for scheme_dir in sorted(base.iterdir()):
        if not scheme_dir.is_dir():
            continue
        scheme_name = scheme_dir.name
        models = []
        for model_dir in sorted(scheme_dir.iterdir()):
            mf = model_dir / "metrics.json"
            if not mf.exists():
                continue
            with open(mf) as f:
                metrics = json.load(f)
            entry = {
                "model": model_dir.name,
                "scheme": scheme_name,
                "macro_f1": metrics.get("macro_f1", 0),
                "rmse": metrics.get("rmse", 0),
                "total_accuracy": metrics.get("total_accuracy", 0),
                "per_class": metrics.get("per_class", {}),
                "confusion_matrix": metrics.get("confusion_matrix", []),
            }
            models.append(entry)
        if models:
            schemes[scheme_name] = models
    return schemes
