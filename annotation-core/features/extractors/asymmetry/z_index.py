import numpy as np

N_SEGMENTS = 4


def compute_z_index(
    heights: np.ndarray,
    segment_labels: np.ndarray,
    sides: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Compute the Z Index (paper §3.4.4, Eq. 11).

        Z = Σ_i  w_i · |z̄_i^L - z̄_i^R|

    where z̄ are regional mean surface heights above the reference plane in
    left / right sub-regions of each anatomical segment.

    Args:
        heights:        np.ndarray (N,) -- signed surface height per vertex.
        segment_labels: np.ndarray (N,) -- segment id 0-3 per vertex.
        sides:          np.ndarray (N,) -- 0=left, 1=right per vertex.
        weights:        np.ndarray (4,) -- per-segment weights w_i.
                        Defaults to equal weights (1/4 each).

    Returns:
        z_index:    float -- global Z Index.
        z_segments: np.ndarray (4,) -- per-segment Z Index values.
    """
    heights = np.asarray(heights, dtype=np.float64)
    segment_labels = np.asarray(segment_labels, dtype=np.int32)
    sides = np.asarray(sides, dtype=np.int32)

    if weights is None:
        weights = np.ones(N_SEGMENTS, dtype=np.float64) / N_SEGMENTS
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != N_SEGMENTS:
        raise ValueError(f"weights must have length {N_SEGMENTS}.")

    # 逐段计算 Z Index：每个区域内取左右两侧表面高度的均值差
    z_segments = np.zeros(N_SEGMENTS, dtype=np.float64)

    for seg in range(N_SEGMENTS):
        seg_mask = segment_labels == seg
        left_mask = seg_mask & (sides == 0)
        right_mask = seg_mask & (sides == 1)

        # 若某段只有单侧有顶点，跳过（边界处可能发生）
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue

        # 公式 Eq.11：|z̄_L - z̄_R|
        z_segments[seg] = abs(heights[left_mask].mean() - heights[right_mask].mean())

    # 全局 Z Index = 各段加权和
    z_index = float(np.dot(weights, z_segments))
    return z_index, z_segments
