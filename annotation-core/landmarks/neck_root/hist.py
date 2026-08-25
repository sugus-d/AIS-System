"""Neck width histogram helpers for neck root detection."""

import numpy as np

from ..constants import NeckRoot


def compute_histogram_mode_width(
    lines: np.ndarray,
    waist_w: float,
) -> tuple[int, list[dict]]:
    """从颈部宽度剖面计算直方图模态宽度。

    在宽度位于 [waist_w*0.2, waist_w*0.7] 区间内构建直方图，取计数最多的
    bin 作为模态区间。该 bin 内最窄宽度作为 W_mode。

    Args:
        lines: 形状为 (M, 4) 的数组，每行 [x_left, x_right, y, width]。
        waist_w: 腰部宽度，用于计算上/下界。

    Returns:
        mode_bin_num: 模态所在的 bin 索引。
        bin_info: 每个直方图 bin 的调试信息列表，每个元素为 dict:
            {x_left, x_right, y, width, is_mode, count, bin_lo, bin_hi}。
            数据不足的 bin 中 x_left/x_right/y/width 为 None。

    Raises:
        ValueError: neck_ws 数据点不足 NeckRoot.MIN_NECK_MASK_POINTS 时。
    """

    # WHY：颈宽的范围大致在腰宽的 20%~70% 之间（WAIST_W_LOWER_RATIO=0.2,
    # WAIST_W_UPPER_RATIO=0.7）。超出这个区间的宽度要么是腰部本身（>0.7），
    # 要么是脖颈上部被头发/衣物干扰（<0.2），排除它们能提高直方图的信噪比。
    waist_w_upperbound = waist_w * NeckRoot.WAIST_W_UPPER_RATIO
    waist_w_lowerbound = waist_w * NeckRoot.WAIST_W_LOWER_RATIO

    neck_mask = (lines[:, 3] >= waist_w_lowerbound) & (lines[:, 3] < waist_w_upperbound)
    if neck_mask.sum() < NeckRoot.MIN_NECK_MASK_POINTS:
        raise ValueError(f"Insufficient neck width data points ({neck_mask.sum()}) for histogram mode calculation")

    # WHY：用幅度为 sqrt(N) 的直方图对筛选后的宽度值做密度估计。
    # 取计数最多的 bin 作为"最典型"的颈宽区间，因为这个 bin 对应了
    # 脖子窄茎上出现次数最多的宽度值，也就是颈宽稳定段。
    lines = lines[neck_mask]
    n_hist_bins = max(NeckRoot.MIN_HIST_BINS, int(np.sqrt(len(lines))))
    hist, bin_edges = np.histogram(lines[:, 3], bins=n_hist_bins)
    mode_bin_num = int(np.argmax(hist))
    # WHY：遍历每个 bin 构建调试信息。对于非空的 bin，取其中宽度最小的行
    # 代表该 bin，因为最窄宽度最能反映该段是窄茎还是过渡区。
    bin_info: list[dict] = []
    for num in range(n_hist_bins):
        is_mode = num == mode_bin_num

        bin_lo = bin_edges[num]
        bin_hi = bin_edges[num + 1]
        bin_mask = (lines[:, 3] >= bin_lo) & (lines[:, 3] <= bin_hi)
        if bin_mask.sum() <= 0:
            bin_info.append(
                {
                    "x_left": None,
                    "x_right": None,
                    "y": None,
                    "width": None,
                    "is_mode": is_mode,
                    "count": int(hist[num]),
                    "bin_lo": bin_lo,
                    "bin_hi": bin_hi,
                }
            )
            continue

        bin_idx = int(np.argmin(lines[bin_mask][:, 3]))
        line = lines[bin_mask][bin_idx]

        bin_info.append(
            {
                "x_left": line[0],
                "x_right": line[1],
                "y": line[2],
                "width": line[3],
                "is_mode": is_mode,
                "count": int(hist[num]),
                "bin_lo": bin_lo,
                "bin_hi": bin_hi,
            }
        )

    return mode_bin_num, bin_info
