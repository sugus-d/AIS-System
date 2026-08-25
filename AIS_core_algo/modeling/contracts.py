"""AIS 预测管道 — 数据契约。

训练管线各组件间的共享数据类型与配置。
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any
from typing import Protocol as _Protocol

from numpy.typing import NDArray

from utils.paths import MODELING_PREDICTION_DIR

# 训练结果根目录（契约层别名，指向 utils/paths.MODELING_PREDICTION_DIR）
# 注意：此为训练结果子目录（非 results 根；utils/paths.py 的 RESULTS_DIR 是根目录，两者勿混）
TRAINING_RESULTS_DIR = MODELING_PREDICTION_DIR


@dataclass
class FeatureSet:
    """特征工程后的最终特征集——固定、无筛选、直接可训练。"""
    name: str                    # 方案名: "canonical_47d"
    y: NDArray                   # (N,)
    X: NDArray                   # (N, n_features) 已拼接好的特征矩阵
    feature_names: list[str]     # 全部特征列名

    # 分块边界（可空——不是所有方案都有分块信息）
    block_slices: dict[str, slice] | None = None
    selection_log: dict = field(default_factory=dict)

    # per-fold 筛选原始块（启用 TrainingConfig.feature_selector 时提供）
    X_raw_blocks: dict[str, NDArray] | None = None
    region_column_names: list[str] | None = None


# =====================================================================
# ⑤ Training
# =====================================================================


@dataclass
class TrainingConfig:
    """训练配置——解耦的组件装配。"""
    models: list[str]                    # 模型名列表
    version: str = "v1.0.0"              # 训练方法版本号（v1.0.0 现行 / 0.1.x 活跃 / 0.0.x 历史）
    alias: str = ""                      # 精简语义别名（如 composite_v7→c7）

    # DataSplitter 配置
    data_splitter: str = "stratified_kfold"
    data_splitter_params: dict = field(default_factory=lambda: {"n_splits": 5, "n_repeats": 1})

    # HPSearcher 配置
    hp_searcher: str = "random"
    hp_searcher_params: dict = field(default_factory=lambda: {"n_iter": 25})
    hp_space_overrides: dict | None = None  # 参数范围覆盖

    # 输出
    output_dir: str = str(TRAINING_RESULTS_DIR)

    # 样本加权
    weighting: str | None = None  # 加权策略名: "inv_freq" "severe_boost" 等，None=不加权
    weighting_params: dict = field(default_factory=dict)  # 策略参数
    weight_components: list | None = None  # 主权重乘区（最终训练用）
    search_weight_components: list | None = None  # HP 搜索用（默认=weight_components）
    search_data_splitter: str | None = None  # HP 搜索内层 CV 切分器，默认和 data_splitter 一致

    # 后处理校准
    calibrate: bool = False  # 是否在 CV 后做 per-class 偏差校正

    # 目标变换
    transform_target: bool = True  # 是否对 >48° 做 log 压缩变换

    # Trainer 实现选择
    trainer: str | None = None  # None=原版 Trainer, "margin"=MarginTrainer

    # per-fold 嵌入式特征筛选（防泄漏；None=不筛选，用方案特征）
    feature_selector: str | None = None  # "per_fold"=启用（需 FeatureSet.X_raw_blocks）


@dataclass
class TrainingResult:
    """单模型训练结果。"""
    scheme: str
    model_name: str
    predictions: NDArray          # (N,)
    metrics: dict                 # {"f1": ..., "sens": ..., "spec": ..., "rmse": ..., "r": ...}
    best_params: dict             # 搜索到的最佳参数（最后一折）
    details: dict = field(default_factory=dict)  # 详细日志（非结构化）
    fold_details: list[dict] = field(default_factory=list)  # 每折详情
    session_dir: str = ""         # 结果目录路径
    training_log: str = ""        # 会话日志路径


class DataSplitter(_Protocol):
    """数据切分策略"""
    name: str
    def split(self, y: NDArray
              ) -> Generator[tuple[NDArray, NDArray], None, None]: ...


class HPSearcher(_Protocol):
    """超参搜索策略"""
    name: str
    def search(self, model: object, X: NDArray, y: NDArray,
               splitter: DataSplitter, **params: object) -> tuple[Any, dict]: ...
