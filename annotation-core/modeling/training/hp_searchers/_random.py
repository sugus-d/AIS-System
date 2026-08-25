"""随机超参搜索策略。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from modeling.contracts import DataSplitter
from modeling.training.hp_searchers._scores import _fold_scores
from modeling.training.hp_searchers._search_utils import (
    _apply_trial_to_weights,
    _DefaultInnerSplitter,
    _grid_size,
    _merge_weight_space,
    _narrow_grid,
)
from utils.logger import logger


class RandomSearch:
    """随机超参搜索。

    Attributes:
        name: 策略名，固定为 "random"。
    """

    name = "random"

    def search(self, model: object, X: NDArray, y: NDArray,
               splitter: DataSplitter | None = None, **params: object
               ) -> tuple[object, dict]:
        """随机采样搜索最佳超参。

        Args:
            model:    模型实例（用于获取 get_param_space() 和类型）。
            X:        特征矩阵。
            y:        目标值（已在变换空间）。
            splitter: 内层切分策略。None 时使用默认的 3 折 KFold。
            **params: n_iter (int), score_metric (str),
                      ref_params (dict | None), hp_space_overrides (dict | None),
                      weight_components (list | None)。

        Returns:
            (最佳模型（已用全量数据训练）, 最佳参数)。
        """
        n_iter = params.get("n_iter", 40)
        score_metric = params.get("score_metric", "r2")
        ref_params = params.get("ref_params")
        hp_space_overrides = params.get("hp_space_overrides")
        wide = params.get("wide", False)
        weight_components: list | None = params.get("weight_components")

        hp_grid = dict(model.get_param_space())
        if hp_space_overrides:
            hp_grid.update(hp_space_overrides)
        model_hp_keys = set(hp_grid.keys())
        model_key_order = list(hp_grid.keys())
        hp_grid = _merge_weight_space(hp_grid, model_key_order, weight_components, wide=wide)

        if ref_params:
            narrowed = _narrow_grid(ref_params, hp_grid)
            hp_grid = narrowed

        keys = list(hp_grid.keys())
        values = list(hp_grid.values())
        total = _grid_size(hp_grid)
        n_iter = min(total, n_iter)

        if splitter is None:
            inner_splitter: DataSplitter = _DefaultInnerSplitter()
        else:
            inner_splitter = splitter

        rng = np.random.default_rng(42)
        best_score = -np.inf
        best_model: object = None
        best_params: dict = {}

        for it in range(n_iter):
            combo = [list(v)[rng.integers(len(v))] for v in values]
            trial_params = dict(zip(keys, combo, strict=True))
            model_params = {k: v for k, v in trial_params.items() if k in model_hp_keys}
            trial_model = type(model)(model_params)
            _apply_trial_to_weights(weight_components or [], trial_params)
            scores = _fold_scores(trial_model, X, y, inner_splitter, score_metric,
                                  weight_components)
            mean_score = float(np.mean(scores))
            is_best = mean_score > best_score
            if is_best:
                best_score = mean_score
                best_params = trial_params
                best_model = trial_model
            hp_str = ", ".join(f"{k}={v}" for k, v in trial_params.items())
            score_str = ", ".join(f"{s:.4f}" for s in scores)
            logger.info(f"    trial[{it+1}/{n_iter}] HP=[{hp_str}]  scores=[{score_str}] mean={mean_score:.4f}{' ★BEST' if is_best else ''}")

        logger.info(f"  >> 搜索结束: best_params={best_params}")
        if best_model is None:
            best_model = type(model)()
            best_model.fit(X, y)

        return best_model, best_params
