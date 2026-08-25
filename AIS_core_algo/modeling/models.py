"""回归模型注册表 —— 数据驱动工厂。

原先 18 个近同构包装文件（modeling/models/ 子包）合并为单一模块：
每个模型由 _MODEL_SPECS 中的一条 spec 描述（估计器构造器、param_space、
fit 行为标志），由 _make_model_class 生成独立的模型类，行为与旧实现逐位一致。

用法:
    from modeling.models import REGISTRY, get_model, list_models
    model = get_model("Ridge")
    model.fit(X_train, y_train, sample_weight=sw)
    preds = model.predict(X_test)
"""

from __future__ import annotations

from typing import Any

from .models_base import (
    _build_bagging_en,
    _builder,
    _oversample,  # noqa: F401 — re-exported for external callers
    BaseModel,
)
from .models_ensemble import EnsembleMean, EnsembleStack, EnsembleWeighted
from .models_spec import _make_model_class, _SpecModel  # noqa: F401 — re-exported for external callers

# 每条 spec 描述一个模型：builder 构造估计器，param_space 供超参数搜索，
# 标志位控制 fit 行为（accepts_sample_weight / oversample / lazy_estimator /
# random_search / n_iter），fit 为 None 时走 _SpecModel 通用路径。
_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "BaggingEN": {
        "builder": _build_bagging_en,
        "param_space": {"n_estimators": [20, 50, 100]},
        "accepts_sample_weight": False,
        "oversample": True,
    },
    "CatBoost": {
        "builder": None,
        "param_space": {"learning_rate": [0.01, 0.05, 0.1], "depth": [4, 6, 8], "iterations": [100, 300]},
        "lazy_estimator": True,
        "fit": "_fit_catboost",
        "random_search": True,
    },
    "DecisionTree": {
        "builder": _builder("sklearn.tree", "DecisionTreeRegressor", random_state=42),
        "param_space": {"max_depth": [3, 5, 7, 10], "min_samples_split": [2, 5, 10]},
    },
    "ElasticNet": {
        "builder": _builder("sklearn.linear_model", "ElasticNet", random_state=42, max_iter=5000),
        "param_space": {
            "alpha": [0.0005, 0.002, 0.005, 0.02, 0.05, 0.2, 0.5, 2, 5],
            "l1_ratio": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        },
        "random_search": True,
        "n_iter": 30,
    },
    "ExtraTrees": {
        "builder": _builder("sklearn.ensemble", "ExtraTreesRegressor", random_state=42),
        "param_space": {"n_estimators": [100, 300], "max_depth": [3, 5, 7]},
        "accepts_sample_weight": False,
        "random_search": True,
    },
    "GBRT": {
        "builder": _builder("sklearn.ensemble", "GradientBoostingRegressor", random_state=42),
        "param_space": {"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        "accepts_sample_weight": False,
        "random_search": True,
    },
    "HistGBRT": {
        "builder": _builder("sklearn.ensemble", "HistGradientBoostingRegressor", random_state=42, max_iter=500),
        "param_space": {"max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2]},
        "accepts_sample_weight": False,
        "oversample": True,
        "random_search": True,
        "n_iter": 6,
    },
    "Huber": {
        "builder": _builder("sklearn.linear_model", "HuberRegressor", max_iter=100000, tol=1e-4),
        "param_space": {"epsilon": [1.0, 1.15, 1.35, 1.5, 1.75, 2.0]},
        "accepts_sample_weight": False,
        "fit": "_fit_huber",
    },
    "KNN": {
        "builder": _builder("sklearn.neighbors", "KNeighborsRegressor"),
        "param_space": {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]},
        "oversample": True,
    },
    "LightGBM": {
        "builder": _builder("lightgbm", "LGBMRegressor", random_state=42, n_jobs=1, verbose=-1),
        "param_space": {"learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5], "num_leaves": [7, 15, 31]},
        "random_search": True,
        "n_iter": 6,
    },
    "MLP": {
        "builder": _builder("sklearn.neural_network", "MLPRegressor", random_state=42, max_iter=2000),
        "param_space": {"hidden_layer_sizes": [(50,), (100,), (50, 25)], "alpha": [0.0001, 0.001, 0.01]},
        "oversample": True,
    },
    "RF": {
        "builder": _builder("sklearn.ensemble", "RandomForestRegressor", random_state=42),
        "param_space": {"n_estimators": [100, 300, 500], "max_depth": [3, 5, 7, 10]},
        "random_search": True,
    },
    "Ridge": {
        "builder": _builder("sklearn.linear_model", "Ridge", random_state=42, max_iter=5000),
        "param_space": {"alpha": [0.01, 0.1, 1, 10, 100]},
    },
    "SGD": {
        "builder": _builder("sklearn.linear_model", "SGDRegressor", loss="huber", max_iter=5000, random_state=42),
        "param_space": {"alpha": [0.0001, 0.001, 0.01], "learning_rate": ["invscaling", "adaptive"]},
        "accepts_sample_weight": False,
    },
    "SVR": {
        "builder": _builder("sklearn.svm", "SVR", max_iter=5000),
        "param_space": {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1]},
        "oversample": True,
    },
    "XGBoost": {
        "builder": None,
        "param_space": {"learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7], "n_estimators": [100, 300]},
        "lazy_estimator": True,
        "fit": "_fit_xgboost",
        "random_search": True,
        "n_iter": 6,
    },
}


def _register_specs() -> None:
    """按 spec 生成模型类并注册到 REGISTRY（顺序即注册顺序）。

    动态创建的类必须绑定到本模块命名空间并设置 ``__module__``，
    否则 joblib/pickle 序列化模型实例时找不到类（PicklingError）。
    """
    for name, spec in _MODEL_SPECS.items():
        model_cls = _make_model_class(name, {**spec, "name": name})
        model_cls.__module__ = __name__
        globals()[name] = model_cls
        REGISTRY[name] = model_cls


REGISTRY: dict[str, type] = {}
_register_specs()
# Ensemble（非 BaseModel，独立的 fit/predict 接口），保持在 BaseModel 之后注册
REGISTRY["EnsembleMean"] = EnsembleMean
REGISTRY["EnsembleWeighted"] = EnsembleWeighted
REGISTRY["EnsembleStack"] = EnsembleStack
REGISTRY["EnsembleStackEN"] = lambda: EnsembleStack(meta="en")


def get_model(name: str) -> BaseModel:
    """按名称获取模型实例（大小写敏感）。

    Args:
        name: 模型名，如 "Ridge"、"LightGBM"。

    Returns:
        对应模型的实例。
    """
    if name not in REGISTRY:
        raise KeyError(f"未知模型: {name}，可选: {list(REGISTRY.keys())}")
    return REGISTRY[name]()


def list_models() -> list[str]:
    """列出所有已注册的模型名。"""
    return list(REGISTRY.keys())
