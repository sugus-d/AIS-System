"""从 mesh + UV 参数化结果中提取不对称特征。

复用 analysis 层的 compute_asymmetric_index、compute_z_index、compute_region_features
等函数，将它们的输出整理为统一字典。
"""

from __future__ import annotations

import numpy as np

from features.extractors.asymmetry.asymmetric_index import compute_asymmetric_index
from features.extractors.asymmetry.landmark_regions import (
    classify_by_region,
    compute_region_features,
)
from features.extractors.asymmetry.z_index import compute_z_index

SEGMENT_NAMES = ["shoulder", "thoracic", "lumbar", "pelvic"]


def extract_asymmetry(
    *,
    uv_coords: np.ndarray,
    heights: np.ndarray,
    curv_mean: np.ndarray,
    curv_gauss: np.ndarray,
    lambda_m: float = 1.0,
    lambda_g: float = 1.0,
    segment_weights: np.ndarray | None = None,
) -> dict:
    """从 mesh + UV 参数化结果中提取全部不对称特征。

    包含：
      - AI（曲率不对称指数）全局 + 4 节段
      - Z Index（高度不对称）全局 + 4 节段
      - 区域特征（region-based |Δmean|）：height / curv_mean / curv_gauss per region

    Args:
        uv_coords:  (N, 2) UV 参数化坐标。
        heights:    (N,) 顶点高度。
        curv_mean:  (N,) 顶点平均曲率。
        curv_gauss: (N,) 顶点高斯曲率。
        lambda_m:   平均曲率项权重。
        lambda_g:   高斯曲率项权重。
        segment_weights: 各节段权重，默认等权 (4,)。

    Returns:
        dict: 不对称特征：
            - ai_global / ai_<segment>
            - z_global / z_<segment>
            - region_height / region_curv_mean / region_curv_gauss (per region)
    """
    # UV-based 区域分类（使用模板区域多边形）
    labels, sides = classify_by_region(uv_coords)

    # 节段数检查
    n_segments = len(SEGMENT_NAMES)
    if segment_weights is None:
        segment_weights = np.ones(n_segments, dtype=np.float64) / n_segments
    segment_weights = np.asarray(segment_weights, dtype=np.float64)

    # --- AI（曲率不对称）---
    ai_global, ai_segments = compute_asymmetric_index(
        curv_mean,
        curv_gauss,
        labels,
        sides,
        weights=segment_weights,
        lambda_m=lambda_m,
        lambda_g=lambda_g,
    )

    # --- Z Index（高度不对称）---
    z_global, z_segments = compute_z_index(
        heights,
        labels,
        sides,
        weights=segment_weights,
    )

    # --- 区域 |Δmean| 特征（shoulder / thoracic / lumbar，不含 pelvic）---
    region_features, region_names = compute_region_features(
        heights,
        curv_mean,
        curv_gauss,
        labels,
        sides,
    )

    result: dict[str, float] = {
        "ai_global": float(ai_global),
        "z_global": float(z_global),
    }
    for i, name in enumerate(SEGMENT_NAMES):
        result[f"ai_{name}"] = float(ai_segments[i])
        result[f"z_{name}"] = float(z_segments[i])

    for i, rname in enumerate(region_names):
        result[f"region_{rname}"] = float(region_features[i])

    return result
