"""数据切分策略 — 实现 DataSplitter 协议。

参考 modeling/_shared.py:_stratify_bins 的分层逻辑和 cross_validate 的 K 折流程。
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold, StratifiedKFold

from modeling._shared import _stratify_bins, CLINICAL
from utils.logger import logger


class KFoldSplitter:
    """普通 K 折切分，支持多轮重复。

    Attributes:
        name: 策略名，固定为 "kfold"。
    """

    name = "kfold"

    def __init__(self, n_splits: int = 5, n_repeats: int = 5, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, y: NDArray
              ) -> Generator[tuple[NDArray, NDArray], None, None]:
        """产生 (train_idx, test_idx) 的迭代器。

        每轮 repeat 产生 n_splits 个切分，seed = random_state + repeat。
        """
        for repeat in range(self.n_repeats):
            seed = self.random_state + repeat
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
            yield from cv.split(np.zeros(len(y)))


class StratifiedKFoldSplitter:
    """分层 K 折切分 — 按目标值分层后切分。

    分层逻辑参考 modeling/_shared.py:_stratify_bins。

    Attributes:
        name: 策略名，固定为 "stratified_kfold"。
    """

    name = "stratified_kfold"

    def __init__(self, n_splits: int = 5, n_repeats: int = 5,
                 random_state: int = 42, clinical: float = CLINICAL) -> None:
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.clinical = clinical

    def split(self, y: NDArray
              ) -> Generator[tuple[NDArray, NDArray], None, None]:
        """产生 (train_idx, test_idx) 的迭代器。

        每轮 repeat 先对 y 做分层 binning，再用 StratifiedKFold 切分。
        某类样本数 < n_splits 时 sklearn 只发 UserWarning 但仍继续（近似分层），
        这里显式退化为普通 KFold 并记日志，避免无效分层。
        """
        for repeat in range(self.n_repeats):
            seed = self.random_state + repeat
            bins = _stratify_bins(y, clinical=self.clinical)
            if np.bincount(bins).min() < self.n_splits:
                logger.warning(
                    f"stratified_kfold: 最少类样本 {np.bincount(bins).min()} < "
                    f"n_splits={self.n_splits}，退化为普通 KFold"
                )
                cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
                yield from cv.split(np.zeros(len(y)))
                continue
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
            yield from cv.split(np.zeros(len(y)), bins)


SPLITTERS: dict[str, type] = {
    "kfold": KFoldSplitter,
    "stratified_kfold": StratifiedKFoldSplitter,
}
