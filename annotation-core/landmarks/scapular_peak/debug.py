"""Top-level debug dict builder for scapular peak detection."""


def build_scapular_peak_debug(
    y_lo: float,
    y_hi: float,
    mid_x: float,
    band_size: int,
    left_debug: dict,
    right_debug: dict,
    dy_mm: float,
    corrected: bool,
) -> dict:
    """构建肩胛峰检测的顶层 debug 字典。

    Args:
        y_lo: Y band 下界。
        y_hi: Y band 上界。
        mid_x: 分界中线 X。
        band_size: band 内顶点数。
        left_debug: 左侧检测中间过程字典。
        right_debug: 右侧检测中间过程字典。
        dy_mm: 左右肩胛峰 Y 坐标差（mm）。
        corrected: 是否触发了对称修正。

    Returns:
        dict: 顶层 debug 字典，结构与原 scapular_peak.py 完全一致。
    """
    return {
        "y_lo": y_lo,
        "y_hi": y_hi,
        "mid_x": mid_x,
        "band_size": band_size,
        "left": left_debug,
        "right": right_debug,
        "symmetry": {
            "dy_mm": dy_mm,
            "corrected": corrected,
        },
    }
