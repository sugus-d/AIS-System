"""Per-fold 嵌入式特征筛选 — 每折在训练集上选特征 + CI 合成。

自原 ml/cross_verification/ 迁入（逻辑原样，防信息泄漏），作为 Trainer 的可选组件
（TrainingConfig.feature_selector="per_fold" 启用）。

子模块拆分：
  - feature_selector_scoring  : 混合评分 + 去高相关策略
  - feature_selector_select   : 2700D 分组解析 + morph/region 筛选
  - feature_selector_ci       : per-fold CI 计算
"""

from __future__ import annotations

import numpy as np

from modeling.training.feature_selector_ci import _compute_ci_per_fold
from modeling.training.feature_selector_scoring import (  # noqa: F401
    _dedup_by_corr,
    _dedup_keep_first,
    _dedup_replace_better,
    _hybrid_scores,
)
from modeling.training.feature_selector_select import (
    _parse_2700d_groups,
    _select_morph,
    _select_region,
)

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

# 2700D 的 8 种测量类型（列名匹配用）
MEASUREMENTS = [
    "height", "mean_curv", "gauss_curv", "roughness",
    "normal_angle", "normal_vector_cos",
]
DIFF_METHODS = ["dm", "pw"]


# ---------------------------------------------------------------------------
# PerFoldFeatureSelector — Trainer 集成组件
# ---------------------------------------------------------------------------


class PerFoldFeatureSelector:
    """per-fold 嵌入式特征筛选：basic 全保留 + morph top10 + region topN + CI 合成。

    在每折训练集上执行筛选（fit_transform），测试折用已拟合的索引/CI 变换（transform）。
    与旧 cross_verification 行为一致，供 Trainer 在 scaler 之前调用。
    """

    def __init__(self, w_auc: float = 2.0) -> None:
        self.w_auc = w_auc
        self._morph_ix: np.ndarray | None = None
        self._reg_ix: np.ndarray | None = None
        self._ci_fn = None

    def fit_transform(
        self,
        raw_blocks: dict[str, np.ndarray],
        y_tr: np.ndarray,
        region_column_names: list[str] | None = None,
    ) -> np.ndarray:
        """在训练折上筛选并返回拼接后的训练特征。

        Args:
            raw_blocks: 原始特征块 {"basic": (N,5), "morph": (N,~58), "region": (N,2700)}。
            y_tr: 训练折目标（原始空间，未变换）。
            region_column_names: 2700D 列名（启用 CI 合成时必填）。

        Returns:
            (N, n_sel) 拼接特征：basic + morph_sel + region_sel + ci。
        """
        parts: list[np.ndarray] = []
        basic = raw_blocks.get("basic")
        morph = raw_blocks.get("morph")
        region = raw_blocks.get("region")
        if basic is not None:
            parts.append(basic)
        if morph is not None:
            self._morph_ix = _select_morph(morph, y_tr, w_auc=self.w_auc)
            parts.append(morph[:, self._morph_ix])
        if region is not None:
            self._reg_ix = _select_region(region, y_tr, w_auc=self.w_auc)
            parts.append(region[:, self._reg_ix])
        if region is not None and region_column_names:
            groups = _parse_2700d_groups(region_column_names)
            ci_tr, self._ci_fn = _compute_ci_per_fold(region, y_tr, groups, w_auc=self.w_auc)
            if ci_tr.shape[1] > 0:
                parts.append(ci_tr)
        if not parts:
            raise ValueError("raw_blocks 至少需要一个特征块")
        return np.column_stack(parts) if len(parts) > 1 else parts[0]

    def transform(self, raw_blocks: dict[str, np.ndarray]) -> np.ndarray:
        """用已拟合的筛选索引/CI 变换测试折。"""
        parts: list[np.ndarray] = []
        basic = raw_blocks.get("basic")
        morph = raw_blocks.get("morph")
        region = raw_blocks.get("region")
        if basic is not None:
            parts.append(basic)
        if morph is not None and self._morph_ix is not None:
            parts.append(morph[:, self._morph_ix])
        if region is not None and self._reg_ix is not None:
            parts.append(region[:, self._reg_ix])
        if region is not None and self._ci_fn is not None:
            parts.append(self._ci_fn(region))
        if not parts:
            raise ValueError("raw_blocks 至少需要一个特征块")
        return np.column_stack(parts) if len(parts) > 1 else parts[0]
