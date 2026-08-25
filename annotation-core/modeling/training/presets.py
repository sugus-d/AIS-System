"""训练预设注册表 — CLI 快捷预设（3 个）。

⚠️ 与 schemes.py 的关系：本表是**快捷层**，仅覆盖 weighting/calibrate/
hp_n_iter 三项，供 `--training-preset` 使用；schemes.py 是完整训练方案
（15 个，供 `--scheme` / pipeline/run.py 使用）。本表 "baseline" 与
schemes.py "baseline" **语义不同**（此处是 Ridge/hp20），勿混用。
"""

from __future__ import annotations

from dataclasses import dataclass

from modeling.contracts import TrainingConfig
from modeling.training.weights import InvFreqWeight, PerClassWeight


def _tc(
    models: list[str] | None = None,
    weight_components: list | None = None,
    data_splitter: str = "kfold",
    hp_searcher: str = "random",
    hp_n_iter: int = 20,
    trainer: str | None = None,
    calibrate: bool = False,
) -> TrainingConfig:
    """构建 TrainingConfig 的快捷函数。"""
    return TrainingConfig(
        models=models or ["Ridge"],
        weight_components=weight_components,
        data_splitter=data_splitter,
        hp_searcher=hp_searcher,
        hp_searcher_params={"n_iter": hp_n_iter, "score_metric": "r2"},
        trainer=trainer,
        calibrate=calibrate,
    )


@dataclass
class TrainingPreset:
    """命名训练管线预设。"""
    name: str
    label: str
    description: str
    config: TrainingConfig


TRAINING_PRESETS: dict[str, TrainingPreset] = {
    "baseline": TrainingPreset(
        name="baseline", label="基线",
        description="Ridge, 无加权, 随机搜索 20 次",
        config=_tc(),
    ),
    "weighted_inv": TrainingPreset(
        name="weighted_inv", label="逆频率",
        description="Ridge + InvFreqWeight(mr=3) + 校准",
        config=_tc(weight_components=[InvFreqWeight(max_ratio=3.0, normalize=False)],
                   hp_n_iter=40, trainer="margin", calibrate=True),
    ),
    "severe_boost": TrainingPreset(
        name="severe_boost", label="Severe 8x",
        description="Ridge + PerClassWeight(Severe=8x) + 校准",
        config=_tc(weight_components=[PerClassWeight(weights=(1.0, 1.0, 1.0, 8.0), normalize=False)],
                   hp_n_iter=40, trainer="margin", calibrate=True),
    ),
}


def get_training_preset(name: str) -> TrainingPreset:
    """按名称获取训练预设。"""
    if name not in TRAINING_PRESETS:
        raise KeyError(f"未知训练预设: {name}，可选: {list(TRAINING_PRESETS.keys())}")
    return TRAINING_PRESETS[name]
