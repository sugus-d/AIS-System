"""网格工具 — 搜索网格的注入、收窄与规模计算，以及默认内层切分。"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold


def _inject_weight_params(hp_grid: dict, wide: bool = False) -> dict:
    """注入权重搜索参数（class_weight, dist_k），如网格中尚无。

    Args:
        hp_grid: 原始搜索网格。
        wide:    是否使用宽松搜索范围。

    Returns:
        注入后的搜索网格（不修改原 dict）。
    """
    injected = dict(hp_grid)
    if "class_weight" not in injected:
        injected["class_weight"] = [3, 5, 8, 12, 15, 20] if wide else [3, 5, 8, 12]
    if "dist_k" not in injected:
        injected["dist_k"] = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0] if wide else [0.05, 0.1, 0.2, 0.5]
    return injected


def _grid_size(grid: dict) -> int:
    """计算网格总组合数。"""
    total = 1
    for v in grid.values():
        total *= len(v)
    return total


def _merge_weight_space(hp_grid: dict, model_key_order: list[str],
                        weight_components: list | None, wide: bool = False) -> dict:
    """注入权重搜索参数并重排 key 顺序（RNG 兼容）。

    权重参数前置、模型参数保持原始顺序、其余追加——RandomSearch 的 RNG 组合
    顺序依赖该 key 顺序，搬移必须保序。

    Args:
        hp_grid: 原始搜索网格（会被注入并重排，返回新 dict）。
        model_key_order: 模型参数原始 key 顺序。
        weight_components: 权重组件列表（含 _hp_keys/_hp_space 时注入其搜索空间）。
        wide: 是否使用宽松权重搜索范围。

    Returns:
        注入并重排后的搜索网格。
    """
    hp_grid = _inject_weight_params(hp_grid, wide=wide)
    if weight_components:
        for wc in weight_components:
            # 首次 fold 保存 _hp_keys 和搜索空间；后续 fold 复用（get_param_space 在 params 被 mutate 后返回空）
            if not wc._hp_keys:  # noqa: SLF001
                ps = wc.get_param_space()
                wc._hp_keys = set(ps.keys())  # noqa: SLF001
                wc._hp_space = {k: list(spec["values"]) for k, spec in ps.items()}  # noqa: SLF001
            for key, values in getattr(wc, '_hp_space', {}).items():
                hp_grid[key] = values
    _WC_K = ["class_weight", "dist_k", "margin_factor", "sigma", "max_ratio"]
    return {k: hp_grid[k] for k in _WC_K if k in hp_grid} | \
           {k: hp_grid[k] for k in model_key_order if k in hp_grid and k not in _WC_K} | \
           {k: hp_grid[k] for k in hp_grid if k not in _WC_K and k not in model_key_order}


def _apply_trial_to_weights(weight_components: list, trial_params: dict) -> None:
    """把 trial 参数应用到权重组件（支持别名键如 cw_20 → class_weight）。

    Args:
        weight_components: 权重组件列表。
        trial_params: 单次搜索的参数组合。
    """
    for wc in weight_components:
        for key, val in trial_params.items():
            if key in wc._hp_keys:  # noqa: SLF001
                # 支持别名键（如 cw_20 → class_weight）
                _KEY_ALIAS = {"cw": "class_weight", "dk": "dist_k"}
                _key = key
                for _sfx in ("_10", "_20", "_40"):
                    if _key.endswith(_sfx):
                        _base = _key[:-len(_sfx)]
                        _key = _KEY_ALIAS.get(_base, _base)
                        break
                setattr(wc, _key, val)


def _narrow_grid(ref_params: dict, hp_grid: dict) -> dict:
    """围绕参考参数收窄搜索范围（70%~130%）。

    Args:
        ref_params: 参考参数（如上一轮最佳）。
        hp_grid:    原始搜索网格。

    Returns:
        收窄后的搜索网格。
    """
    narrowed: dict = {}
    for key, values in hp_grid.items():
        if key not in ref_params or not isinstance(ref_params[key], (int, float)):
            narrowed[key] = values
            continue
        is_int = isinstance(ref_params[key], int)
        best_val = float(ref_params[key])
        lo, hi = best_val * 0.7, best_val * 1.3
        candidates = [v for v in values if isinstance(v, (int, float)) and lo <= v <= hi]
        if len(candidates) >= MIN_NARROW_CANDIDATES:
            narrowed[key] = [int(v) for v in candidates] if is_int else [round(float(v), 4) for v in candidates]
        else:
            vals = np.linspace(lo, hi, 5)
            narrowed[key] = [int(round(v)) for v in vals] if is_int else [round(float(v), 4) for v in vals]
    return narrowed


class _DefaultInnerSplitter:
    """默认内层切分：固定的 3 折 KFold，供 HPSearcher 使用。"""

    name = "inner_kfold_3"

    def __init__(self, n_splits: int = 3, random_state: int = 42) -> None:
        self._cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def split(self, y: NDArray) -> Generator[tuple[NDArray, NDArray], None, None]:
        yield from self._cv.split(np.zeros(len(y)))


MIN_NARROW_CANDIDATES = 2
"""_narrow_grid 中候选参数的保留阈值。"""

_MIN_NUMERIC_VALUES = 2
"""将参数视为连续型所需的最少数值候选数。"""

_LOG_SCALE_RATIO = 10
"""连续参数范围比超过该值时使用对数尺度采样。"""
