"""数据驱动模型生成。"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import numpy as np

from utils.logger import logger

from .models_base import _oversample, BaseModel


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
