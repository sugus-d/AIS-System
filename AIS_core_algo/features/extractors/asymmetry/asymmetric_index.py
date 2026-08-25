"""不对称指数计算 — 按区域段聚合顶点级测量，输出左右不对称指标。

供特征提取（region 候选）与热力图渲染共用，N_SEGMENTS 段划分对齐论文口径。
"""

import numpy as np

N_SEGMENTS = 4


def compute_asymmetric_index(
    curvatures_mean: np.ndarray,
    curvatures_gauss: np.ndarray,
    segment_labels: np.ndarray,
    sides: np.ndarray,
    weights: np.ndarray | None = None,
    lambda_m: float = 1.0,
    lambda_g: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Compute the Asymmetric Index (paper §3.4.3, Eq. 10).

        AI = Σ_i  w_i · ( λ_M |κ̄_M,i^L - κ̄_M,i^R|
                         + λ_G |κ̄_G,i^L - κ̄_G,i^R| )

    Args:
        curvatures_mean:  np.ndarray (N,) -- mean curvature per vertex.
        curvatures_gauss: np.ndarray (N,) -- Gaussian curvature per vertex.
        segment_labels:   np.ndarray (N,) -- segment id 0-3 per vertex.
        sides:            np.ndarray (N,) -- 0=left, 1=right per vertex.
        weights:          np.ndarray (4,) -- per-segment weights w_i.
                          Defaults to equal weights (1/4 each).
        lambda_m: Weight for mean curvature term.
        lambda_g: Weight for Gaussian curvature term.

    Returns:
        ai:          float -- global Asymmetric Index.
        ai_segments: np.ndarray (4,) -- per-segment AI values.
    """
    curvatures_mean = np.asarray(curvatures_mean, dtype=np.float64)
    curvatures_gauss = np.asarray(curvatures_gauss, dtype=np.float64)
    segment_labels = np.asarray(segment_labels, dtype=np.int32)
    sides = np.asarray(sides, dtype=np.int32)

    if weights is None:
        weights = np.ones(N_SEGMENTS, dtype=np.float64) / N_SEGMENTS
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != N_SEGMENTS:
        raise ValueError(f"weights must have length {N_SEGMENTS}.")

    # 逐段计算不对称度：先按区域标签筛选，再分左右求均值差异
    ai_segments = np.zeros(N_SEGMENTS, dtype=np.float64)

    for seg in range(N_SEGMENTS):
        seg_mask = segment_labels == seg
        left_mask = seg_mask & (sides == 0)
        right_mask = seg_mask & (sides == 1)

        # 若某段只有一侧有顶点（边界处），跳过不计算
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue

        mean_L = curvatures_mean[left_mask].mean()
        mean_R = curvatures_mean[right_mask].mean()
        gauss_L = curvatures_gauss[left_mask].mean()
        gauss_R = curvatures_gauss[right_mask].mean()

        # 公式 Eq.10：λ_M · |Δκ̄_M| + λ_G · |Δκ̄_G|
        ai_segments[seg] = lambda_m * abs(mean_L - mean_R) + lambda_g * abs(gauss_L - gauss_R)

    # 全局 AI = 各段加权和
    ai = float(np.dot(weights, ai_segments))
    return ai, ai_segments
