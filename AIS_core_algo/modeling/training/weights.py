"""WeightComponent — 独立乘区权重架构。

每个乘区独立 compute → normalize(×N/sum(w)) → 全部相乘。
参数可固定也可搜索（由 HP 搜索管理）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from modeling.metrics import SEVERITY_BINS


class WeightComponent(ABC):
    """权重乘区基类。

    每个乘区独立计算权重，可选归一化（×N/sum(w)），最终与其他乘区乘积。
    所有参数通过 __init__ 传入：固定值直接存，可搜索值以 list 形式存。
    _hp_keys 由 HPSearcher 在合并搜索空间时设置，用于追踪可搜索参数。
    """

    name: str = ""
    _hp_keys: set[str] = set()

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize  # compute 时是否自动归一化

    @abstractmethod
    def _compute_raw(self, y: NDArray) -> NDArray:
        """子类实现：返回原始权重（未归一化）。"""

    def compute(self, y: NDArray) -> NDArray:
        """计算权重，按 self.normalize 决定是否归一化。

        Args:
            y: 目标值数组 (N,)。

        Returns:
            权重数组 (N,)，浮点值。
        """
        w = self._compute_raw(y)
        if self.normalize:
            w = self._normalize(w)
        return w

    def _normalize(self, w: NDArray) -> NDArray:
        """归一化：w → w × N / sum(w)，下界保护。"""
        eps = 1e-8
        w = np.clip(w, eps, None)
        return w * len(w) / max(w.sum(), eps)

    def get_param_space(self) -> dict:
        """返回可搜索参数空间。

        Returns:
            {param_name: {"type": "discrete", "values": [...]}} 格式。
            无可搜索参数时返回空 dict。
        """
        return {}


class DecayWeight(WeightComponent):
    """衰减加权：[class_weight] × exp(-dist_k × |y - clinical|)

    参数为 list 时表示可搜索（由 HP 搜索管理取值范围），
    参数为标量时表示固定。

    threshold=True 时（匹配旧 _build_weight 行为）：
      y ≤ clinical 时乘 class_weight，y > clinical 时乘 1.0
    threshold=False 时（匹配旧 continuous_decay 行为）：
      全部乘 class_weight

    Args:
        clinical:     衰减中心（°）。默认 20。
        class_weight: 权重倍率。默认 5。
        dist_k:       衰减速率。默认 0.1。
        normalize:    是否自动归一化。默认 True。
        threshold:    是否在 clinical 两侧应用不同倍率。默认 False。
    """

    name = "decay"

    def __init__(
        self, clinical: int | list[int] = 20,
        class_weight: float | list[float] = 5.0,
        dist_k: float | list[float] = 0.1,
        normalize: bool = True,
        threshold: bool = False,
        key_map: dict | None = None,
    ) -> None:
        super().__init__(normalize=normalize)
        self.clinical = clinical
        self.class_weight = class_weight
        self.dist_k = dist_k
        self.threshold = threshold
        self._key_map = key_map or {}  # e.g. {"class_weight": "cw_20"}

    def get_param_space(self) -> dict:
        space = {}
        for key in ("class_weight", "dist_k", "clinical"):
            val = getattr(self, key)
            if isinstance(val, list):
                sk = self._key_map.get(key, key)
                space[sk] = {"type": "discrete", "values": list(val)}
        return space

    def _get_value(self, param: object) -> float:
        return float(param[0]) if isinstance(param, list) else float(param)

    def _compute_raw(self, y: NDArray) -> NDArray:
        clinical = self._get_value(self.clinical)
        cw = self._get_value(self.class_weight)
        dk = self._get_value(self.dist_k)
        if self.threshold:
            cw = np.where(y <= clinical, cw, 1.0)
        return cw * np.exp(-dk * np.abs(y - clinical))



class InvFreqWeight(WeightComponent):
    """逆频率加权 — 按 4 个 severity 类分别加权。

    Args:
        max_ratio: 最大权重上限。None 不限制。默认 3.0。
        bins:      severity 分界点。
        normalize: 是否自动归一化。默认 True。
    """

    name = "inv_freq"

    def __init__(self, max_ratio: float | None = 3.0,
                 bins: list[float] | None = None,
                 normalize: bool = True) -> None:
        super().__init__(normalize=normalize)
        self.max_ratio = max_ratio
        self.bins = bins or list(SEVERITY_BINS)

    def _compute_raw(self, y: NDArray) -> NDArray:
        labels = np.digitize(y, self.bins[1:-1])
        counts = np.bincount(labels, minlength=len(self.bins) - 1).astype(float)
        counts = np.maximum(counts, 1)
        max_count = float(counts.max())
        weights = max_count / counts[labels]  # 逐元素除法，与列表推导逐位一致
        mr = float(self.max_ratio[0]) if isinstance(self.max_ratio, list) else (self.max_ratio or 0.0)
        if mr > 1.0:
            weights = np.clip(weights, 1.0, mr)
        return weights

    def get_param_space(self) -> dict:
        val = self.max_ratio
        if isinstance(val, list):
            return {"max_ratio": {"type": "discrete", "values": list(val)}}
        return {}


class MarginBoostWeight(WeightComponent):
    """边界增强加权 — 在指定边界附近加权。

    公式: w = 1 + margin_factor × exp(-min_dist_to_boundary / sigma)

    Args:
        margin_factor: 边界处额外权重倍数。默认 2.0。
        sigma:         距离衰减速度（°）。默认 3.0。
        bins:          severity 分界点，边界取自 bins[1:-1]。
        normalize:     是否自动归一化。默认 True。
    """

    name = "margin_boost"

    def __init__(self, margin_factor: float = 2.0, sigma: float = 3.0,
                 bins: list[float] | None = None,
                 normalize: bool = True) -> None:
        super().__init__(normalize=normalize)
        self.margin_factor = margin_factor
        self.sigma = max(sigma, 0.1) if not isinstance(sigma, list) else sigma
        self.bins = bins or list(SEVERITY_BINS)

    def _compute_raw(self, y: NDArray) -> NDArray:
        boundaries = np.array(self.bins[1:-1])
        dist = np.min(np.abs(y[:, None] - boundaries[None, :]), axis=1)
        mf = float(self.margin_factor[0]) if isinstance(self.margin_factor, list) else float(self.margin_factor)
        sg = float(self.sigma[0]) if isinstance(self.sigma, list) else float(self.sigma)
        return 1.0 + mf * np.exp(-dist / sg)

    def get_param_space(self) -> dict:
        space = {}
        for key in ("margin_factor", "sigma"):
            val = getattr(self, key)
            if isinstance(val, list):
                space[key] = {"type": "discrete", "values": list(val)}
        return space


class PerClassWeight(WeightComponent):
    """每类自定义加权 — 4 个 severity 类分别指定权重。

    Args:
        weights: 4 元组，(Normal, Mild, Moderate, Severe) 权重。
        bins:    severity 分界点。
        normalize: 是否自动归一化。默认 True。
    """

    name = "per_class"

    def __init__(self, weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
                 bins: list[float] | None = None,
                 normalize: bool = True) -> None:
        super().__init__(normalize=normalize)
        self.weights = np.array(weights, dtype=float)
        self.bins = bins or list(SEVERITY_BINS)

    def _compute_raw(self, y: NDArray) -> NDArray:
        labels = np.digitize(y, self.bins[1:-1])
        return self.weights[labels]


def build_weight_components(weighting: str | None, params: dict | None = None) -> list | None:
    """按策略名构建权重乘区列表（CLI --weighting 装配层）。

    Args:
        weighting: 加权策略名："inv_freq" / "per_class" / "severe_boost"；
                   None 或 "uniform" 表示不加权。
        params: 策略参数（透传组件构造，如 per_class 的 weights 元组）。

    Returns:
        权重组件列表；None = 不加权。
    """
    params = params or {}
    if weighting in (None, "uniform"):
        return None
    if weighting == "inv_freq":
        return [InvFreqWeight(**params)]
    if weighting == "per_class":
        weights = tuple(params.get("weights", (1.0, 1.0, 1.0, 8.0)))
        return [PerClassWeight(weights=weights)]
    if weighting == "severe_boost":
        return [PerClassWeight(weights=(1.0, 1.0, 1.0, 8.0))]
    msg = f"未知加权策略: {weighting}，可选: inv_freq/per_class/severe_boost/uniform"
    raise ValueError(msg)


class ConstWeight(WeightComponent):
    """常数权重 — 不改变权重，只注入搜索噪声参数。

    Args:
        normalize: 是否自动归一化。默认 True。
        **noise_params: 搜索噪声参数，如 solver=["svd","cholesky"]。
    """

    name = "noise"

    def __init__(self, normalize: bool = True, **noise_params: object) -> None:
        super().__init__(normalize=normalize)
        self._noise_params = noise_params

    def _compute_raw(self, y: NDArray) -> NDArray:
        return np.ones(len(y), dtype=float)

    def get_param_space(self) -> dict:
        return {
            k: {"type": "discrete", "values": list(v)}
            for k, v in self._noise_params.items()
            if isinstance(v, (list, tuple))
        }
