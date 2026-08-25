"""全网格超参搜索策略。"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from modeling.contracts import DataSplitter
from modeling.training.hp_searchers._scores import _fold_scores
from modeling.training.hp_searchers._search_utils import (
    _apply_trial_to_weights,
    _DefaultInnerSplitter,
    _merge_weight_space,
    _narrow_grid,
)


class GridSearch:
    """全网格超参搜索（遍历所有组合）。

    Attributes:
        name: 策略名，固定为 "grid"。
    """

    name = "grid"

    def search(self, model: object, X: NDArray, y: NDArray,
               splitter: DataSplitter | None = None, **params: object
               ) -> tuple[object, dict]:
        """全网格遍历搜索最佳超参。

        Args:
            model:    模型实例（用于获取 get_param_space() 和类型）。
            X:        特征矩阵。
            y:        目标值（已在变换空间）。
            splitter: 内层切分策略。None 时使用默认的 3 折 KFold。
            **params: score_metric (str),
                      ref_params (dict | None), hp_space_overrides (dict | None),
                      sample_weight (NDArray | None)。

        Returns:
            (最佳模型（已用全量数据训练）, 最佳参数)。
        """
        score_metric = params.get("score_metric", "r2")
        ref_params = params.get("ref_params")
        hp_space_overrides = params.get("hp_space_overrides")
        wide = params.get("wide", False)
        weight_components: list | None = params.get("weight_components")
        hp_grid = model.get_param_space()
        if hp_space_overrides:
            hp_grid.update(hp_space_overrides)
        # 外部加权时，class_weight/dist_k 不生效，不注入搜索空间
        model_hp_keys = set(hp_grid.keys())
        model_key_order = list(hp_grid.keys())
        hp_grid = _merge_weight_space(hp_grid, model_key_order, weight_components, wide=wide)

        if ref_params:
            narrowed = _narrow_grid(ref_params, hp_grid)
            hp_grid = narrowed

        keys = list(hp_grid.keys())
        values = list(hp_grid.values())

        if splitter is None:
            inner_splitter: DataSplitter = _DefaultInnerSplitter()
        else:
            inner_splitter = splitter

        best_score = -np.inf
        best_model: object = None
        best_params: dict = {}

        # 用 product 生成所有组合（字典序展开，与手写嵌套等价）
        def _iterate_grid() -> list[dict]:
            return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]

        all_combos = _iterate_grid()
        for trial_params in all_combos:
            model_params = {k: v for k, v in trial_params.items() if k in model_hp_keys}
            trial_model = type(model)(model_params)
            _apply_trial_to_weights(weight_components or [], trial_params)
            scores = _fold_scores(trial_model, X, y, inner_splitter, score_metric,
                                  weight_components)
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = trial_params
                best_model = trial_model

        if best_model is None:
            best_model = type(model)()
            best_model.fit(X, y)

        return best_model, best_params
