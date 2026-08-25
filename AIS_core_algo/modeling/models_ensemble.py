"""Ensemble 模型。"""

import numpy as np


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
