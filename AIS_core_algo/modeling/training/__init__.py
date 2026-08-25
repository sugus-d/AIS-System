"""训练组件 — DataSplitter, HPSearcher, Trainer, WeightComponent, 训练方案。"""

from modeling.contracts import DataSplitter, HPSearcher
from modeling.training.data_splitters import (
    KFoldSplitter,
    SPLITTERS,
    StratifiedKFoldSplitter,
)
from modeling.training.hp_searchers import (
    GridSearch,
    RandomSearch,
    SEARCHERS,
)
from modeling.training.schemes import get_scheme, list_schemes, TRAINING_SCHEMES
from modeling.training.trainer import Trainer

__all__ = [
    "DataSplitter",
    "HPSearcher",
    "KFoldSplitter",
    "StratifiedKFoldSplitter",
    "SPLITTERS",
    "RandomSearch",
    "GridSearch",
    "SEARCHERS",
    "Trainer",
    "TRAINING_SCHEMES",
    "get_scheme",
    "list_schemes",
]
