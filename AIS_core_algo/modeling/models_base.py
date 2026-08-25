"""回归模型基类与样本加权辅助。"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np


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
