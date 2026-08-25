"""Scapular superior angle: most posterior + superior on each scapula."""

import numpy as np

from utils.logger import logger

from ..constants import ScapularPeak


def detect_scapular_peak(
    vertices: np.ndarray,
    y_min: float,
    y_range: float,
    spine_midline: np.ndarray,
    neck_root: np.ndarray,
    curvature: np.ndarray | None = None,
) -> np.ndarray:
    """检测每侧肩胛峰（scapular peak）：在肩胛带选择最向后且较高的点。

    WHY：肩胛峰位于肩胛骨上角，在背部点云中表现为后凸且相对靠上的区域。
    用脊柱中线（而非下半身中点）分侧，结合 Z 值分位过滤和对称性校验，
    在含噪声的点云中稳健定位左右肩胛峰。

    Args:
        vertices: 网格顶点数组 (N, 3)。
        y_min: 顶点 Y 最小值。
        y_range: 顶点 Y 范围。
        spine_midline: 脊柱中线 (M, 3)，来自 fit_spine_midline。
        neck_root: 颈根点 (2, 3) [left, right]，用于解剖合理性校验。

    Returns:
        scapular_peaks: (2, 3) [left, right] 左右肩胛峰坐标。
    """
    # 主 Y 搜索区间 60%~80%：肩胛骨上角位于 T2-T3 水平，
    # 高于此区间靠近脖子（噪声区），低于则靠近肩胛骨体部（区分度下降）
    y_lo = y_min + ScapularPeak.Y_LO_RATIO * y_range
    y_hi = y_min + ScapularPeak.Y_HI_RATIO * y_range
    mask = (vertices[:, 1] >= y_lo) & (vertices[:, 1] <= y_hi)
    band_size = int(mask.sum())
    logger.info(f"Searching scapular peak: y_range={y_range:.1f}, band=[{y_lo:.1f}, {y_hi:.1f}], size={band_size}")
    if band_size < ScapularPeak.MIN_BAND_SIZE:
        # 点数过少说明网格有缺损或体态极端——仅告警，搜索区间不变
        # （原 fallback 用 FALLBACK_Y_*_RATIO 与主区间同值，是 no-op 分支，已删除）
        logger.warning(f"Band size {band_size} < {ScapularPeak.MIN_BAND_SIZE}")

    band_indices: np.ndarray = np.where(mask)[0]
    band = vertices[mask]

    # 用脊柱中线（spine_midline）的 Y 带内 X 中值作为左右分界中心，
    # 比单纯使用顶点 X 中值更鲁棒——脊柱位置不受肩胛骨偏移影响
    spine_in_band = spine_midline[(spine_midline[:, 1] >= y_lo) & (spine_midline[:, 1] <= y_hi)]
    if len(spine_in_band) >= ScapularPeak.MIN_SPINE_BAND_POINTS:
        mid_x = float(np.median(spine_in_band[:, 0]))
    elif len(spine_midline) > 0:
        mid_x = float(np.median(spine_midline[:, 0]))
    else:
        mid_x = float(np.median(band[:, 0]))

    # 验证 mid_x 在 band X 范围内：spine_midline 可能在搜索区外（罕见体态），
    # 此时 fallback 到 band 自身的中值
    band_x_min, band_x_max = float(band[:, 0].min()), float(band[:, 0].max())
    if not (band_x_min < mid_x < band_x_max):
        logger.warning(
            f"mid_x {mid_x:.1f} outside band X range [{band_x_min:.1f}, {band_x_max:.1f}], fallback to band median"
        )
        mid_x = float(np.median(band[:, 0]))

    x_span = float(band[:, 0].max() - band[:, 0].min())
    margin = ScapularPeak.MID_X_MARGIN_RATIO * x_span  # 中线缓冲区，防止分界上的点被两侧都取到

    # 左右分侧检测
    axilla_y_est = y_min + 0.3 * y_range
    left_peak = _detect_one_side(
        band,
        band_indices,
        True,
        mid_x,
        margin,
        neck_root_y=neck_root[0][1],
        neck_root_x=neck_root[0][0],
        curvature=curvature,
        axilla_y_est=axilla_y_est,
        y_range=y_range,
        x_span=x_span,
    )
    right_peak = _detect_one_side(
        band,
        band_indices,
        False,
        mid_x,
        margin,
        neck_root_y=neck_root[1][1],
        neck_root_x=neck_root[1][0],
        curvature=curvature,
        axilla_y_est=axilla_y_est,
        y_range=y_range,
        x_span=x_span,
    )
    results = np.stack([left_peak, right_peak])

    # 对称性校验：两侧 Y 差超过 3% 身高时触发修正。
    # 肩胛骨上角在解剖上高度对称，偏差大说明单侧检测被噪声干扰
    dy_mm = abs(float(results[0][1] - results[1][1]))
    corrected = dy_mm > ScapularPeak.SYMMETRY_DY_RATIO * y_range
    if corrected:
        logger.warning(f"Symmetry correction triggered: dY={dy_mm:.1f}mm > {0.03 * y_range:.1f}mm")
        # 保留 Y 较高（偏上，更靠近脖子）的一侧为参考，
        # 另一侧以参考 Y 为目标重新搜索
        if results[0][1] >= results[1][1]:
            ref_peak, bad_side_name, bad_idx = results[0], "right", 1
        else:
            ref_peak, bad_side_name, bad_idx = results[1], "left", 0
        corrected_peak = _detect_one_side(
            band,
            band_indices,
            bad_side_name == "left",
            mid_x,
            margin,
            target_y=ref_peak[1],
            neck_root_y=neck_root[bad_idx][1],
            neck_root_x=neck_root[bad_idx][0],
            curvature=curvature,
            axilla_y_est=axilla_y_est,
            y_range=y_range,
            x_span=x_span,
        )
        results[bad_idx] = corrected_peak

    return results


def _detect_one_side(
    band: np.ndarray,
    band_indices: np.ndarray,
    is_left: bool,
    mid_x: float,
    margin: float,
    target_y: float | None = None,
    neck_root_y: float | None = None,
    neck_root_x: float | None = None,
    curvature: np.ndarray | None = None,
    axilla_y_est: float | None = None,
    y_range: float | None = None,
    x_span: float | None = None,
) -> np.ndarray:
    """在 band 中检测一侧肩胛峰。

    Args:
        band: Y band 内的顶点 (B, 3)。
        band_indices: band 顶点在原 vertices 中的索引 (B,)。
        is_left: True=左侧, False=右侧。
        mid_x: 分界中线 X。
        margin: 中线缓冲区宽度。
        target_y: 对称修正时指定的目标 Y，在该 Y 附近选取。

    Returns:
        该侧肩胛峰坐标 (3,)。
    """
    side_mask = band[:, 0] < mid_x - margin if is_left else band[:, 0] >= mid_x + margin

    side = band[side_mask]
    side_indices = band_indices[side_mask]
    side_size = len(side)
    z_threshold = 0.0
    high_z_size = 0
    peak: np.ndarray = np.zeros(3)

    if side_size < ScapularPeak.MIN_SIDE_SIZE:
        # 点数太少，取该侧 Z 最大点
        logger.warning(f"{'Left' if is_left else 'Right'} side_size {side_size} < 3, fallback to Z max")
        if side_size > 0:
            peak = side[np.argmax(side[:, 2])]
        elif len(band) > 0:
            peak = band[0]
    else:
        # Z 值分位过滤（top 85%）：肩胛峰在 Z 方向凸起最显著，
        # 但只用 Z 最高点可能选到肋骨而非肩胛骨，保留更多候选供综合评分
        z_threshold = float(np.percentile(side[:, 2], ScapularPeak.Z_PERCENTILE))
        high_z_mask = side[:, 2] >= z_threshold
        high_z = side[high_z_mask]
        high_z_indices = side_indices[high_z_mask]
        high_z_size = len(high_z)

        if high_z_size < ScapularPeak.MIN_HIGH_Z_SIZE:
            # 高 Z 点太少，取该侧 Y 最高点
            logger.warning(f"{'Left' if is_left else 'Right'} high_z_size {high_z_size} < 2, fallback to Y max")
            peak = side[np.argmax(side[:, 1])]
        else:
            if target_y is not None:
                # 对称修正：在所有 high_z 候选点中找 Y 最接近目标值的点
                peak = high_z[np.argmin(np.abs(high_z[:, 1] - target_y))]
            else:
                K = max(ScapularPeak.K_BASE, int(len(side) * ScapularPeak.K_FRACTION))
                k_idx = np.argsort(high_z[:, 2])[::-1][: min(K, high_z_size)]
                candidates = high_z[k_idx]
                if (
                    neck_root_y is not None
                    and curvature is not None
                    and axilla_y_est is not None
                    and y_range is not None
                ):
                    # 组合评分：曲率 + Z 值 + Y 位置 + X 距颈根。
                    # 同时满足四项约束的点才是真正的肩胛峰——用全部 high_z 候选而非 top-K 避免遗漏
                    candidates_all = high_z
                    cand_indices_all = high_z_indices
                    z_min, z_max = float(candidates_all[:, 2].min()), float(candidates_all[:, 2].max())
                    y_mid = (neck_root_y + axilla_y_est) / 2
                    best_score = -1e9
                    best_local_idx = 0
                    for ci, candidate_pt in enumerate(candidates_all):
                        score = 0.0
                        if curvature[cand_indices_all[ci]] > 0:
                            score += 2.0
                        if z_max > z_min:
                            score += (candidate_pt[2] - z_min) / (z_max - z_min) * 1.0
                        score -= abs(candidate_pt[1] - y_mid) / max(y_range, 1.0) * 3.0
                        if neck_root_x is not None and x_span is not None and x_span > 0:
                            score -= abs(candidate_pt[0] - neck_root_x) / x_span * 3.0
                        if score > best_score:
                            best_score = score
                            best_local_idx = ci
                    peak = candidates_all[best_local_idx]
                else:
                    peak = candidates[np.argmax(candidates[:, 1])]

    return peak
