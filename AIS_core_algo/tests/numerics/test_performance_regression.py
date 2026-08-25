"""性能回归测试 — 快速模型保证方案性能在修改中不退化。

与黄金测试（test_*_golden.py）互补：
  - 黄金测试：锁算法输出逐位一致（确定性，容差 0）
  - 本测试：锁方案性能不退化（±1% 相对容差，CV 有随机性故需容差）

配置：Ridge/ElasticNet 固定参数（hp_space_overrides 单值网格）、无加权、5×5 CV。
基线：2026-08-02 在验证过的 HEAD 上生成。有意的性能变化（算法改进/方案调整）
需人工确认新值后更新 BASELINE_MACRO_F1（失败信息会打印当前实测值）。
"""

from __future__ import annotations

import warnings

from tests.numerics.conftest import DATA_DIR

# 基线 Macro-F1: {方案: {模型: 值}} — 2026-08-02 生成（Ridge/ElasticNet 固定参数无加权 5×5）
BASELINE_MACRO_F1: dict[str, dict[str, float]] = {
    "v0.1.0": {"Ridge": 0.6311, "ElasticNet": 0.6254},
    "archived/morph_region_ci_37d": {"Ridge": 0.6202, "ElasticNet": 0.6184},
    "archived/morph_region_ci_36d": {"Ridge": 0.6151, "ElasticNet": 0.6153},
    "archived/morph_region_ci_35d": {"Ridge": 0.5640, "ElasticNet": 0.6143},
    "archived/morph_region_ci_27d": {"Ridge": 0.6502, "ElasticNet": 0.6611},
    "archived/canonical_union_64d": {"Ridge": 0.5204, "ElasticNet": 0.5947},
    "archived/canonical_44d": {"Ridge": 0.5070, "ElasticNet": 0.5335},
}

REL_TOLERANCE = 0.01  # ±1% 相对容差（双向：超差即提示人工确认）

FIXED_PARAMS: dict[str, dict] = {
    "Ridge": {"alpha": [1.0]},
    "ElasticNet": {"alpha": [0.5], "l1_ratio": [0.5]},
}


def test_performance_no_regression(monkeypatch) -> None:
    """全部选择方案 × 快速模型：Macro-F1 与基线相对偏差 ≤ ±1%。"""
    warnings.filterwarnings("ignore")
    monkeypatch.chdir(DATA_DIR / "features")
    from features.selectors.schemes import SELECTION_REGISTRY
    from modeling.contracts import FeatureSet, TrainingConfig
    from modeling.metrics import compute_4class_metrics
    from modeling.training.trainer import Trainer

    for scheme, model_baselines in BASELINE_MACRO_F1.items():
        data = SELECTION_REGISTRY[scheme].load()
        for model_name, baseline in model_baselines.items():
            config = TrainingConfig(
                models=[model_name],
                data_splitter="kfold",
                data_splitter_params={"n_splits": 5, "n_repeats": 5},
                hp_searcher="random",
                hp_searcher_params={"n_iter": 1},
                hp_space_overrides=FIXED_PARAMS[model_name],  # 单值网格=固定参数
                transform_target=True,
            )
            feature_set = FeatureSet(
                name=scheme, y=data["y"], X=data["X_basic"],
                feature_names=[],
            )
            result = Trainer(config).train(feature_set)[0]
            metrics = compute_4class_metrics(data["y"], result.predictions)
            current = metrics["macro_f1"]
            rel_change = abs(current - baseline) / baseline
            assert rel_change <= REL_TOLERANCE, (
                f"{scheme} × {model_name}: Macro-F1 {current:.4f} vs 基线 {baseline:.4f} "
                f"(相对偏差 {rel_change:.2%} > ±{REL_TOLERANCE:.0%})\n"
                "若为有意的性能变化，人工确认后更新 BASELINE_MACRO_F1"
            )
