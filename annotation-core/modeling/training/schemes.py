"""训练方案注册表 — 每种方案定义完整的训练策略配置。

每个方案指定：
  - 目标变换（transform_target）
  - 加权策略（weighting + weighting_params）
  - 后处理校准（calibrate）
  - Trainer 实现（trainer）
  - 数据切分 + HP 搜索（data_splitter, hp_searcher）

⚠️ 与 presets.py 的关系：本表是**完整训练方案**（15 个，pipeline/run.py
与 modeling/train.py --scheme 使用）；presets.py 是 CLI 快捷预设（3 个，仅覆盖
weighting/calibrate/hp 三项，供 --training-preset 使用）。两表都定义了
名为 "baseline" 的入口但**语义不同**（本表 baseline 无加权 + 无目标变换；
presets baseline 是 Ridge/hp20），勿混用。

用法:
    from modeling.training.schemes import get_scheme
    cfg = get_scheme("baseline")
"""

from __future__ import annotations

from modeling.contracts import TrainingConfig
from modeling.training.weights import DecayWeight, InvFreqWeight, MarginBoostWeight


def _tc(
    transform_target: bool = True,
    weighting: str | None = None,
    weighting_params: dict | None = None,
    calibrate: bool = False,
    trainer: str | None = None,
    data_splitter: str = "stratified_kfold",
    hp_searcher: str = "random",
    hp_n_iter: int = 40,
    weight_components: list | None = None,
    search_weight_components: list | None = None,
    search_data_splitter: str | None = None,
) -> TrainingConfig:
    """快捷构造 TrainingConfig。"""
    return TrainingConfig(
        models=["Ridge"],  # 占位，实际由 CLI 覆盖
        transform_target=transform_target,
        weighting=weighting,
        weighting_params=weighting_params or {},
        calibrate=calibrate,
        trainer=trainer,
        data_splitter=data_splitter,
        data_splitter_params={"n_splits": 5, "n_repeats": 1},
        hp_searcher=hp_searcher,
        hp_searcher_params={"n_iter": hp_n_iter, "score_metric": "r2"},
        weight_components=weight_components,
        search_weight_components=search_weight_components,
        search_data_splitter=search_data_splitter,
    )


TRAINING_SCHEMES: dict[str, TrainingConfig] = {
    # ── Level 0: 裸基线 ──
    "baseline": _tc(
        transform_target=False,
        weighting=None,
        calibrate=False,
    ),
    # ── Level 1: 目标变换 ──
    "log48_transform": _tc(
        transform_target=True,
        weighting=None,
        calibrate=False,
    ),
    # ── Level 2: 连续衰减加权 ──
    "continuous_decay": _tc(
        transform_target=False,
        weighting="continuous_decay",
        weighting_params={"class_weight": 5.0, "dist_k": 0.1, "clinical": 20.0},
        calibrate=False,
    ),
    # ── Level 3: inv_freq 离散加权 ──
    "inv_freq": _tc(
        transform_target=False,
        weighting="inv_freq",
        weighting_params={"max_ratio": 5.0},
        calibrate=False,
    ),
    # ── Level 3b: inv_freq + 校准（最佳组合） ──
    "inv_freq_calib": _tc(
        transform_target=True,
        weighting="inv_freq",
        weighting_params={"max_ratio": 3.0},
        calibrate=True,
        trainer="margin",
    ),
    # ── Level 4: 纯边界强调 ──
    "margin_boost": _tc(
        transform_target=True,
        weighting="margin_boost",
        weighting_params={"margin_factor": 2.0, "sigma": 3.0},
        calibrate=False,
    ),
    # ── Level 5: per_class 自定义加权（Normal=6x） ──
    "per_class_normal6": _tc(
        transform_target=True,
        weighting="per_class",
        weighting_params={"weights": {0: 6.0}},
        calibrate=False,
    ),
    # ── Legacy 0.731 复现 ──
    "composite_0731": _tc(
        transform_target=False,
        weighting="composite",
        weighting_params={"strategies": [["inv_freq", {"max_ratio": 3.0}],
                                         ["margin_boost", {}],
                                         ["continuous_decay", {"clinical": 10.0, "class_weight": 2.0, "dist_k": 0.1}]]},
        calibrate=True,
        trainer="margin",
        hp_searcher="random",
        hp_n_iter=100,
        search_data_splitter="kfold",
    ),
    "composite_decay20_search": _tc(
        transform_target=False,
        weighting="composite",
        weighting_params={"strategies": [["inv_freq", {"max_ratio": 3.0}],
                                         ["margin_boost", {}],
                                         ["continuous_decay", {"clinical": 10.0, "class_weight": 2.0, "dist_k": 0.1}]]},
        calibrate=True,
        trainer="margin",
        hp_searcher="random",
        hp_n_iter=100,
        search_data_splitter="kfold",
        search_weight_components=[
            DecayWeight(clinical=20, class_weight=[3, 5, 8, 12],
                        dist_k=[0.05, 0.1, 0.2, 0.5],
                        normalize=False, threshold=True),
        ],
    ),
    "composite_v7": _tc(
        transform_target=False,
        calibrate=True,
        trainer="margin",
        hp_searcher="random",
        hp_n_iter=100,
        search_data_splitter="kfold",
        weight_components=[
            InvFreqWeight(max_ratio=3.0, normalize=False),
            MarginBoostWeight(normalize=False),
            DecayWeight(clinical=10, class_weight=2.0, dist_k=0.1,
                        normalize=False, threshold=False),
        ],
        search_weight_components=[
            DecayWeight(clinical=10, class_weight=[3, 5, 8, 12],
                        dist_k=[0.05, 0.1, 0.2, 0.5],
                        normalize=False, threshold=False),
            DecayWeight(clinical=20, class_weight=[3, 5, 8, 12],
                        dist_k=[0.05, 0.1, 0.2, 0.5],
                        normalize=False, threshold=True),
        ],
    ),
    # ── A: 独立搜索 Decay10/Decay20 的 cw 和 dk ──
    "composite_v7_a": _tc(
        transform_target=False,
        calibrate=True,
        trainer="margin",
        hp_searcher="random",
        hp_n_iter=100,
        search_data_splitter="kfold",
        weight_components=[
            InvFreqWeight(max_ratio=3.0, normalize=False),
            MarginBoostWeight(normalize=False),
            DecayWeight(clinical=10, class_weight=2.0, dist_k=0.1,
                        normalize=False, threshold=False),
        ],
        search_weight_components=[
            DecayWeight(clinical=10, class_weight=[3, 5, 8, 12],
                        dist_k=[0.05, 0.1, 0.2, 0.5],
                        normalize=False, threshold=False),
            DecayWeight(clinical=20, class_weight=[3, 5, 8, 12],
                        dist_k=[0.05, 0.1, 0.2, 0.5],
                        normalize=False, threshold=True,
                        key_map={"class_weight": "cw_20", "dist_k": "dk_20"}),
        ],
    ),
    # ── B: 固定 dk（Bayesian 规律），只搜 cw ──
    "composite_v7_b": _tc(
        transform_target=False,
        calibrate=True,
        trainer="margin",
        hp_searcher="random",
        hp_n_iter=100,
        search_data_splitter="kfold",
        weight_components=[
            InvFreqWeight(max_ratio=3.0, normalize=False),
            MarginBoostWeight(normalize=False),
            DecayWeight(clinical=10, class_weight=2.0, dist_k=0.1,
                        normalize=False, threshold=False),
        ],
        search_weight_components=[
            DecayWeight(clinical=10, class_weight=[3, 5, 8, 12],
                        dist_k=0.35,  # Bayesian 规律：快衰减
                        normalize=False, threshold=False),
            DecayWeight(clinical=20, class_weight=[3, 5, 8, 12],
                        dist_k=0.08,  # Bayesian 规律：慢衰减
                        normalize=False, threshold=True),
        ],
    ),
    # ── C: 固定 dk + 最终训练包含 Decay20 ──
    "composite_v7_c": _tc(
        transform_target=False,
        calibrate=True,
        trainer="margin",
        hp_searcher="random",
        hp_n_iter=100,
        search_data_splitter="kfold",
        weight_components=[
            InvFreqWeight(max_ratio=3.0, normalize=False),
            MarginBoostWeight(normalize=False),
            DecayWeight(clinical=10, class_weight=2.0, dist_k=0.35,
                        normalize=False, threshold=False),
            DecayWeight(clinical=20, class_weight=2.0, dist_k=0.08,
                        normalize=False, threshold=True),
        ],
        search_weight_components=[
            DecayWeight(clinical=10, class_weight=[3, 5, 8, 12],
                        dist_k=0.35,
                        normalize=False, threshold=False),
            DecayWeight(clinical=20, class_weight=[3, 5, 8, 12],
                        dist_k=0.08,
                        normalize=False, threshold=True),
        ],
    ),
    # ── E: 独立Decay+收紧dk+其余同T7 ──
    "composite_v7_e": _tc(
        transform_target=False,
        calibrate=True,
        trainer="margin",
        hp_searcher="random",
        hp_n_iter=100,
        search_data_splitter="kfold",
        weight_components=[
            InvFreqWeight(max_ratio=3.0, normalize=False),
            MarginBoostWeight(normalize=False),
            DecayWeight(clinical=10, class_weight=2.0, dist_k=0.35,
                        normalize=False, threshold=False),
            DecayWeight(clinical=20, class_weight=2.0, dist_k=0.08,
                        normalize=False, threshold=True),
        ],
        search_weight_components=[
            DecayWeight(clinical=10, class_weight=[3, 5, 8, 12],
                        dist_k=[0.25, 0.30, 0.35, 0.40, 0.50],
                        normalize=False, threshold=False),
            DecayWeight(clinical=20, class_weight=[3, 5, 8, 12],
                        dist_k=[0.05, 0.08, 0.10, 0.12, 0.15],
                        normalize=False, threshold=True,
                        key_map={"class_weight": "cw_20", "dist_k": "dk_20"}),
        ],
    ),
    # ── D: 收紧搜索范围（基于 Bayesian 分析） ──
    "composite_v7_d": _tc(
        transform_target=False,
        calibrate=True,
        trainer="margin",
        hp_searcher="random",
        hp_n_iter=200,
        search_data_splitter="kfold",
        weight_components=[
            InvFreqWeight(max_ratio=3.0, normalize=False),
            MarginBoostWeight(normalize=False),
            DecayWeight(clinical=10, class_weight=2.0, dist_k=0.1,
                        normalize=False, threshold=False),
        ],
        search_weight_components=[
            DecayWeight(clinical=10, class_weight=[3, 5, 8, 12],
                        dist_k=[0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
                        normalize=False, threshold=False),
            DecayWeight(clinical=20, class_weight=[3, 5, 8, 12],
                        dist_k=[0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
                        normalize=False, threshold=True),
        ],
    ),
}


def get_scheme(name: str) -> TrainingConfig:
    """按名称获取训练方案配置。"""
    if name not in TRAINING_SCHEMES:
        raise KeyError(f"未知训练方案: {name}，可选: {list(TRAINING_SCHEMES.keys())}")
    # 返回副本防止调用方修改原始配置
    import copy
    return copy.deepcopy(TRAINING_SCHEMES[name])


def list_schemes() -> list[str]:
    """列出所有可用方案。"""
    return list(TRAINING_SCHEMES.keys())
