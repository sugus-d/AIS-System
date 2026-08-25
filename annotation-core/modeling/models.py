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

import importlib
import time
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from utils.logger import logger

# ── 基类与样本加权辅助 ─────────────────────────────────


class BaseModel:
    """所有模型的抽象基类。

    Attributes:
        external_weight: 外部样本权重（由 WeightComponent 设置）。设置后 _build_weight 返回此值。
        params: 传递给 sklearn 构造器的参数字典。
    """

    def __init__(self, params: dict | None = None) -> None:
        self.external_weight: np.ndarray | None = None
        if params is not None:
            self.params = {**params}
        else:
            self.params = {}

    def _build_weight(self, y: np.ndarray) -> np.ndarray:
        """返回样本权重。权重由 WeightComponent 外部计算后通过 external_weight 传入。"""
        if self.external_weight is not None:
            return self.external_weight
        return np.ones(len(y))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """训练模型，子类必须实现。

        Args:
            X: 训练特征，形状 (n_samples, n_features)。
            y: 目标值，形状 (n_samples,)。
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测，子类必须实现。

        Args:
            X: 特征，形状 (n_samples, n_features)。

        Returns:
            预测值，形状 (n_samples,)。
        """
        raise NotImplementedError

    def get_param_space(self) -> dict:
        """返回超参数搜索网格。

        Returns:
            GridSearchCV 兼容的参数字典，空 dict 表示无默认搜索空间。
        """
        return {}


def _oversample(
    X: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """对权重 > 1.0 的样本做重复采样以模拟加权训练。

    每个权重 > 1.0 的样本复制 int(round(w[i])) 份，
    权重 <= 1.0 的样本保留原样。

    Args:
        X: 训练特征。
        y: 目标值。
        w: 样本权重数组，形状 (n_samples,)。

    Returns:
        (增强后的 X, 增强后的 y)。
    """
    if w is None:
        return X, y

    high_mask = w > 1.0
    if not high_mask.any():
        return X, y

    extra_indices: list[int] = []
    for i, flag in enumerate(high_mask):
        if flag:
            repeat = int(round(w[i]))
            # 重复 repeat 份，但原始样本已存在，所以加 repeat - 1 份
            extra_indices.extend([i] * (repeat - 1))

    if not extra_indices:
        return X, y

    X_aug = np.vstack([X, X[extra_indices]])
    y_aug = np.concatenate([y, y[extra_indices]])
    return X_aug, y_aug


# ── 数据驱动工厂 ─────────────────────────────────────


def _builder(module: str, class_name: str, **fixed: Any) -> Callable[[dict], Any]:  # noqa: ANN401 — 构造器参数类型由估计器决定
    """生成惰性导入的估计器构造器。

    Args:
        module: 估计器所在模块（首次构造时才 import）。
        class_name: 估计器类名。
        fixed: 固定 kwargs，可被用户 params 覆盖。

    Returns:
        build(params) -> 估计器实例。
    """

    def build(params: dict) -> Any:  # noqa: ANN401 — 返回类型由估计器决定
        est_module = importlib.import_module(module)
        return getattr(est_module, class_name)(**fixed, **params)

    return build


def _build_bagging_en(params: dict) -> Any:  # noqa: ANN401 — 返回类型由估计器决定
    """BaggingEN 专用构造器：base 估计器是固定默认的 ElasticNet。"""
    from sklearn.ensemble import BaggingRegressor
    from sklearn.linear_model import ElasticNet

    return BaggingRegressor(
        estimator=ElasticNet(random_state=42, max_iter=5000),
        random_state=42,
        **params,
    )


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


class _SpecModel(BaseModel):
    """由 spec 数据驱动生成的模型包装器。

    每个注册模型是通过 _make_model_class 以本类为基类动态生成的独立类，
    _spec 类属性携带该模型的全部配置。
    """

    _spec: ClassVar[dict[str, Any]]

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        if not self._spec.get("lazy_estimator", False):
            self._reg = self._spec["builder"](self.params)

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> None:
        custom_fit = self._spec.get("fit")
        if custom_fit is not None:
            getattr(self, custom_fit)(X, y, sample_weight)
            return
        sw = sample_weight if (self._spec.get("accepts_sample_weight", True) and sample_weight is not None) \
            else self._build_weight(y)
        if self._spec.get("oversample", False):
            X_aug, y_aug = _oversample(X, y, sw)
            self._reg.fit(X_aug, y_aug)
        else:
            self._reg.fit(X, y, sample_weight=sw)

    def _fit_huber(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        """Huber 专用 fit：收敛失败时放宽 tol 后重试。"""
        w = sample_weight if sample_weight is not None else self._build_weight(y)
        try:
            self._reg.fit(X, y, sample_weight=w)
        except ValueError as exc:
            if "convergence failed" not in str(exc):
                raise
            self._reg.set_params(max_iter=500000, tol=1e-3)
            self._reg.fit(X, y, sample_weight=w)

    def _fit_xgboost(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        """XGBoost 专用 fit：惰性 import + 训练日志。"""
        t0 = time.time()
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost 未安装: pip install xgboost") from None
        sw = sample_weight if sample_weight is not None else self._build_weight(y)
        est_params: dict = {"random_state": 42, "verbosity": 0, "n_jobs": 1}
        est_params.update(self.params)
        logger.info(
            f"  XGBoost 开始训练: X={X.shape}, "
            f"n_est={est_params.get('n_estimators','?')}, lr={est_params.get('learning_rate','?')}, "
            f"max_depth={est_params.get('max_depth','?')}, sample_weight={'有' if sw is not None else '无'}"
        )
        self._reg = xgb.XGBRegressor(**est_params)
        self._reg.fit(X, y, sample_weight=sw)
        logger.info(f"  XGBoost 完成: {time.time()-t0:.1f}s")

    def _fit_catboost(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        """CatBoost 专用 fit：惰性 import + 训练日志。"""
        t0 = time.time()
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError("catboost 未安装: pip install catboost") from None
        sw = sample_weight if sample_weight is not None else self._build_weight(y)
        est_params: dict = {"random_seed": 42, "verbose": False, "thread_count": 1}
        est_params.update(self.params)
        logger.info(
            f"  CatBoost 开始训练: X={X.shape}, "
            f"iter={est_params.get('iterations','?')}, lr={est_params.get('learning_rate','?')}, "
            f"depth={est_params.get('depth','?')}, sample_weight={'有' if sw is not None else '无'}"
        )
        self._reg = CatBoostRegressor(**est_params)
        self._reg.fit(X, y, sample_weight=sw)
        logger.info(f"  CatBoost 完成: {time.time()-t0:.1f}s")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_reg"):
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        return self._reg.predict(X)

    def get_param_space(self) -> dict:
        return {key: list(values) for key, values in self._spec["param_space"].items()}


def _make_model_class(name: str, spec: dict[str, Any]) -> type:
    """由 spec 生成独立模型类。

    Args:
        name: 模型名（REGISTRY 的 key，也是类属性 name）。
        spec: 该模型的配置 dict。

    Returns:
        可实例化的模型类（_SpecModel 子类）。
    """
    return type(name, (_SpecModel,), {"name": name, "_spec": spec})


# ── 注册表 ──────────────────────────────────────────


class EnsembleMean:
    """等权平均 ensemble。"""

    name = "EnsembleMean"

    def fit(self, preds_matrix: np.ndarray, _y_true: np.ndarray) -> None:
        n_models = preds_matrix.shape[1]
        self.weights = np.ones(n_models) / n_models

    def predict(self, preds_matrix: np.ndarray) -> np.ndarray:
        return preds_matrix @ self.weights


class EnsembleWeighted:
    """Ridge 学习非负权重后归一化的加权 ensemble。"""

    name = "EnsembleWeighted"

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, preds_matrix: np.ndarray, y_true: np.ndarray) -> None:
        from sklearn.linear_model import Ridge

        meta = Ridge(alpha=self.alpha).fit(preds_matrix, y_true)
        self.weights = np.clip(meta.coef_, 0, None)
        if self.weights.sum() > 0:
            self.weights /= self.weights.sum()
        else:
            n_models = preds_matrix.shape[1]
            self.weights = np.ones(n_models) / n_models

    def predict(self, preds_matrix: np.ndarray) -> np.ndarray:
        return preds_matrix @ self.weights


class EnsembleStack:
    """Stacking ensemble，用 Ridge 或 ElasticNet 做 meta-learner，支持 alpha 搜索。"""

    name = "EnsembleStack"

    def __init__(self, meta: str = "ridge", alpha: float = 1.0) -> None:
        self.meta = meta
        self.alpha = alpha

    def fit(self, preds_matrix: np.ndarray, y_true: np.ndarray) -> None:
        from sklearn.linear_model import ElasticNet, Ridge

        if self.meta == "ridge":
            self._meta = Ridge(alpha=self.alpha)
        else:
            self._meta = ElasticNet(alpha=self.alpha, l1_ratio=0.5)
        self._meta.fit(preds_matrix, y_true)

    def predict(self, preds_matrix: np.ndarray) -> np.ndarray:
        return self._meta.predict(preds_matrix)


def _register_specs() -> None:
    """按 spec 生成模型类并注册到 REGISTRY（顺序即注册顺序）。"""
    for name, spec in _MODEL_SPECS.items():
        REGISTRY[name] = _make_model_class(name, {**spec, "name": name})


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
