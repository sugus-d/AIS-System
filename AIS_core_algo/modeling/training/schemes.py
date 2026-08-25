"""训练方案注册表 — 每种方案定义完整的训练策略配置（17 个，version + alias）。

每个方案指定：
  - 目标变换（transform_target）
  - 加权策略（weighting + weighting_params）
  - 后处理校准（calibrate）
  - Trainer 实现（trainer）
  - 数据切分 + HP 搜索（data_splitter, hp_searcher）

版本三档：v1.0.0=现行生产（composite_v7，生产 ensemble 直接来源）、
v0.1.x=活跃不现行（baseline/composite 变体/加权演进）、v0.0.x=历史不用（单点探索）。
2026-08 由 presets.py（TRAINING_PRESETS 快捷预设层）合并而来，--training-preset 已移除；
`get_scheme` 支持 name/version/alias 三查。

用法:
    from modeling.training.schemes import get_scheme
    cfg = get_scheme("baseline")
"""

from __future__ import annotations

from modeling.contracts import TrainingConfig
from modeling.training.weights import DecayWeight, InvFreqWeight, MarginBoostWeight, PerClassWeight


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
    version: str = "v1.0.0",
    alias: str = "",
) -> TrainingConfig:
    """快捷构造 TrainingConfig（version=训练方法版本号, alias=精简语义别名）。"""
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
        version=version,
        alias=alias,
    )


TRAINING_SCHEMES: dict[str, TrainingConfig] = {
    # ══ v0.1.x 活跃但不现行（11） ══
    # ── 基线（pipeline 默认） ──
    "baseline": _tc(
        transform_target=False,
        weighting=None,
        calibrate=False,
        version="v0.1.0", alias="beta",
    ),
    # ══ v1.0.0 现行生产（composite_v7 生产 ensemble 直接来源） ══
    # ── 复合加权（manuscript ensemble 重训） ──
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
        version="v1.0.0", alias="c7",
    ),
    # ── 逆频率加权（原 CLI 预设并入） ──
    "weighted_inv": _tc(
        calibrate=True,
        trainer="margin",
        hp_n_iter=40,
        weight_components=[InvFreqWeight(max_ratio=3.0, normalize=False)],
        version="v0.1.1", alias="winv",
    ),
    # ── Severe 强化（原 CLI 预设并入） ──
    "severe_boost": _tc(
        calibrate=True,
        trainer="margin",
        hp_n_iter=40,
        weight_components=[PerClassWeight(weights=(1.0, 1.0, 1.0, 8.0), normalize=False)],
        version="v0.1.2", alias="sboost",
    ),
    # ══ v0.1.x 活跃但不现行（续） ══
    # ── Legacy 0.731 复现（composite 加权起点） ──
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
        version="v0.1.3", alias="c0731",
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
        version="v0.1.4", alias="cdec20s",
    ),
    # ── composite_v7 变体搜索（Decay 超参） ──
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
        version="v0.1.5", alias="c7a",
    ),
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
        version="v0.1.6", alias="c7b",
    ),
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
        version="v0.1.7", alias="c7c",
    ),
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
        version="v0.1.8", alias="c7e",
    ),
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
        version="v0.1.9", alias="c7d",
    ),
    # ── inv_freq + 校准（weighted_inv 前身） ──
    "inv_freq_calib": _tc(
        transform_target=True,
        weighting="inv_freq",
        weighting_params={"max_ratio": 3.0},
        calibrate=True,
        trainer="margin",
        version="v0.1.10", alias="ifreq_cal",
    ),
    # ══ v0.0.x 历史不用（5） ══
    "log48_transform": _tc(
        transform_target=True,
        weighting=None,
        calibrate=False,
        version="v0.0.1", alias="log48",
    ),
    "continuous_decay": _tc(
        transform_target=False,
        weighting="continuous_decay",
        weighting_params={"class_weight": 5.0, "dist_k": 0.1, "clinical": 20.0},
        calibrate=False,
        version="v0.0.2", alias="cdecay",
    ),
    "inv_freq": _tc(
        transform_target=False,
        weighting="inv_freq",
        weighting_params={"max_ratio": 5.0},
        calibrate=False,
        version="v0.0.3", alias="ifreq",
    ),
    "margin_boost": _tc(
        transform_target=True,
        weighting="margin_boost",
        weighting_params={"margin_factor": 2.0, "sigma": 3.0},
        calibrate=False,
        version="v0.0.4", alias="mboost",
    ),
    "per_class_normal6": _tc(
        transform_target=True,
        weighting="per_class",
        weighting_params={"weights": {0: 6.0}},
        calibrate=False,
        version="v0.0.5", alias="pcn6",
    ),
}


def get_scheme(name: str) -> TrainingConfig:
    """按 name / version / alias 获取训练方案配置。

    精确 key 优先；否则遍历 version/alias 匹配（如 "c7"→composite_v7、
    "v0.1.3"→composite_v7_b）。
    """
    if name in TRAINING_SCHEMES:
        return _deepcopy(TRAINING_SCHEMES[name])
    for scheme in TRAINING_SCHEMES.values():
        if name in (scheme.version, scheme.alias):
            return _deepcopy(scheme)
    raise KeyError(f"未知训练方案: {name}，可选: {list(TRAINING_SCHEMES.keys())}")


def _deepcopy(cfg: TrainingConfig) -> TrainingConfig:
    """返回副本防止调用方修改原始配置。"""
    import copy
    return copy.deepcopy(cfg)


def list_schemes() -> list[str]:
    """列出所有可用方案。"""
    return list(TRAINING_SCHEMES.keys())
