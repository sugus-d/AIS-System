"""脊柱中线检测的调试字典构建器。"""

import numpy as np


def build_spine_debug(
    pts: np.ndarray,
    mid_x: float,
    n_bins_total: int,
    rejected_mad: np.ndarray | None = None,
    rejected_residual: np.ndarray | None = None,
    pts_clean: np.ndarray | None = None,
) -> dict:
    """构建脊柱中线调试字典，兼容候选点不足和完整过滤两条路径。

    当候选点不足时仅传入最少参数；完整过滤路径可传入所有参数。

    WHY: 统一的 debug 字典结构便于可视化层统一解析，
    避免不同路径返回不同字段导致前端渲染异常。
    """
    n_raw: int = len(pts)
    if rejected_mad is None:
        rejected_mad = np.zeros(n_raw, dtype=bool) if n_raw else np.zeros(0, dtype=bool)
    if rejected_residual is None:
        rejected_residual = np.zeros(n_raw, dtype=bool) if n_raw else np.zeros(0, dtype=bool)
    if pts_clean is None:
        pts_clean = np.zeros((0, 3))
    clean_mask: np.ndarray = ~(rejected_mad | rejected_residual)
    n_clean: int = int(clean_mask.sum())
    x_range_raw: float = float(pts[:, 0].max() - pts[:, 0].min()) if n_raw > 0 else 0.0
    x_range_clean: float = float(pts_clean[:, 0].max() - pts_clean[:, 0].min()) if n_clean > 0 else 0.0
    return {
        "bin_candidates": pts,
        "rejected_mad": rejected_mad,
        "rejected_residual": rejected_residual,
        "candidates_clean": pts_clean,
        "mid_x": mid_x,
        "n_bins_total": n_bins_total,
        "n_candidates_raw": n_raw,
        "n_candidates_clean": n_clean,
        "x_range_raw": x_range_raw,
        "x_range_clean": x_range_clean,
    }
