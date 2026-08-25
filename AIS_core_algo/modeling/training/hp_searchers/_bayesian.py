"""贝叶斯超参搜索策略（optuna TPE + 连续参数）。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from modeling.contracts import DataSplitter
from modeling.training.hp_searchers._scores import _fold_scores
from modeling.training.hp_searchers._search_utils import (
    _apply_trial_to_weights,
    _DefaultInnerSplitter,
    _grid_size,
    _LOG_SCALE_RATIO,
    _merge_weight_space,
    _MIN_NUMERIC_VALUES,
)


class BayesianSearch:
    """贝叶斯超参搜索（optuna TPE + 连续参数）。

    数值参数自动转为连续 suggest_float/suggest_int，
    非数值参数保持 suggest_categorical。
    """

    name = "bayesian"

    def search(self, model: object, X: NDArray, y: NDArray,
               splitter: DataSplitter | None = None, **params: object
               ) -> tuple[object, dict]:
        n_iter = params.get("n_iter", 40)
        score_metric = params.get("score_metric", "r2")
        hp_space_overrides = params.get("hp_space_overrides")
        weight_components: list | None = params.get("weight_components")

        hp_grid = dict(model.get_param_space())
        if hp_space_overrides:
            hp_grid.update(hp_space_overrides)
        model_hp_keys = set(hp_grid.keys())
        model_key_order = list(hp_grid.keys())
        hp_grid = _merge_weight_space(hp_grid, model_key_order, weight_components)

        keys = list(hp_grid.keys())
        values_list = [hp_grid[k] for k in keys]
        total = _grid_size(hp_grid)
        n_iter = min(total, n_iter)

        if splitter is None:
            inner_splitter: DataSplitter = _DefaultInnerSplitter()
        else:
            inner_splitter = splitter

        import optuna

        from utils.logger import logger

        best_score = -np.inf
        best_model: object = None
        best_params: dict = {}

        def _objective(trial: optuna.Trial) -> float:
            nonlocal best_score, best_model, best_params
            trial_params = {}
            for k, vals in zip(keys, values_list, strict=True):
                vlist = list(vals)
                numeric = [v for v in vlist if isinstance(v, (int, float))]
                if len(numeric) == len(vlist) and len(numeric) >= _MIN_NUMERIC_VALUES:
                    lo, hi = float(min(numeric)), float(max(numeric))
                    if all(isinstance(v, int) for v in numeric):
                        trial_params[k] = trial.suggest_int(k, int(lo), int(hi))
                    elif hi / lo > _LOG_SCALE_RATIO:
                        trial_params[k] = trial.suggest_float(k, lo, hi, log=True)
                    else:
                        trial_params[k] = trial.suggest_float(k, lo, hi)
                else:
                    trial_params[k] = trial.suggest_categorical(k, vlist)

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

            if trial.number % 100 == 0 or is_best:
                logger.info(f"  trial[{trial.number+1}/{n_iter}] mean={mean_score:.4f} best={best_score:.4f} {'★' if is_best else ''}")
            if trial.number % 500 == 0:
                logger.info(f"  [Bayesian] @{trial.number+1}/{n_iter} best_so_far={best_score:.4f}")
            return mean_score

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        logger.info(f"  贝叶斯搜索: {n_iter} trials")
        study.optimize(_objective, n_trials=n_iter, show_progress_bar=False)

        logger.info(f"  >> 搜索结束: best_params={best_params}")
        if best_model is None:
            best_model = type(model)()
            best_model.fit(X, y)

        return best_model, best_params
