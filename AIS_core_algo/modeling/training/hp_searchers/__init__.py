"""超参搜索策略 — 实现 HPSearcher 协议。

内层搜索与评分逻辑源自原 cross_verification（已合并为单路径）。

模块划分：
    _scores       评分工具（_r2_score/_sens_spec_score/_macro_f1_score/_fold_scores）
    _search_utils 网格工具（_inject_weight_params/_grid_size/_narrow_grid/_DefaultInnerSplitter）
    _random/_grid/_bayesian  三个搜索类
"""

from modeling.training.hp_searchers._bayesian import BayesianSearch
from modeling.training.hp_searchers._grid import GridSearch
from modeling.training.hp_searchers._random import RandomSearch
from modeling.training.hp_searchers._scores import (
    _fold_scores,
    _macro_f1_score,
    _r2_score,
    _sens_spec_score,
)
from modeling.training.hp_searchers._search_utils import (
    _DefaultInnerSplitter,
    _grid_size,
    _inject_weight_params,
    _narrow_grid,
)

SEARCHERS: dict[str, type] = {
    "random": RandomSearch,
    "grid": GridSearch,
    "bayesian": BayesianSearch,
}

__all__ = [
    "RandomSearch",
    "GridSearch",
    "BayesianSearch",
    "SEARCHERS",
    "_DefaultInnerSplitter",
    "_inject_weight_params",
    "_narrow_grid",
    "_grid_size",
    "_fold_scores",
    "_r2_score",
    "_sens_spec_score",
    "_macro_f1_score",
]
