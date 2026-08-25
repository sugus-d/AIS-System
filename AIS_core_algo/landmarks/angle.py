"""轮廓角度计算工具。

包含在轮廓上侧向采样点并计算角度、以及用于精确定位采样点的私有辅助函数。

注意：此模块只提供纯数学/几何工具，不应依赖任何外部全局状态。所有公开函数有类型注解并返回明确的 numpy 结构。
"""

import numpy as np

from utils.logger import logger

from .geometry import segment_circle_intersection

_MIN_CONTOUR_POINTS = 3       # 至少 3 个点才能取前后邻点
_DEGENERATE_DIST_EPSILON = 1e-9  # 向量长度小于该值视为退化（返回默认角）


def compute_lateral_angle_at_point(
    contour: np.ndarray,
    cand_pt: np.ndarray,
    distance: float = 15.0,
) -> tuple[float, float, np.ndarray, np.ndarray, float, float]:
    """在候选点处沿轮廓采样前后点并计算顺时针转角与余弦值。

    说明：该函数分两步执行：
    1. 在候选点沿轮廓前后寻找距离约为 `distance` 的邻点（优先精确内插）
    2. 基于前后邻点与候选点计算顺时针转角与余弦值

    Args:
        contour: 轮廓点数组，形状 (N, 2)。
        cand_pt: 候选点坐标，数组样式（至少包含 x,y）。
        distance: 前后采样的欧氏距离（mm），默认 15.0。

    Returns:
        Tuple 下列顺序：
        - cosine (float): 向量夹角余弦值。
        - clockwise_deg (float): 顺时针方向角度，值域 [0, 360)。
        - pt_before (np.ndarray): 轮廓遍历方向的前一点坐标（2D）。
        - pt_after (np.ndarray): 轮廓遍历方向的后一点坐标（2D）。
        - dist_before (float): 前一点到候选点的实际距离。
        - dist_after (float): 后一点到候选点的实际距离。

    Notes:
        若任一侧实际距离过小，返回默认值 (cosine=1.0, clockwise_deg=0.0)。
    """
    # 第一步：沿轮廓前后寻找邻点（轮廓遍历顺序）
    pt_prev, pt_next = _find_contour_neighbors(contour, cand_pt, distance=distance)

    # 第二步：计算顺时针转角与余弦值
    cosine, clockwise_deg, dist_prev, dist_next = _compute_angle_and_cosine(
        pt_prev, pt_next, cand_pt
    )

    return cosine, clockwise_deg, pt_prev, pt_next, dist_prev, dist_next


def compute_lateral_angle_profile(
    contour: np.ndarray, distance: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """沿轮廓对每个点采样侧向角度并返回角度曲线。

    说明：该函数对轮廓上每个点调用 compute_lateral_angle_at_point，生成沿轮廓的侧向角序列（angle profile），
    常用于寻找角度曲线的局部极值作为候选 landmark。

    Args:
        contour: 轮廓点数组，形状 (N, 2)。
        distance: 计算夹角时两侧采样的距离（mm），默认 10.0。

    Returns:
        Tuple:
        - sampled_pts (np.ndarray): 原轮廓的 numpy 形式数组，形状 (N, 2)。
        - angle_values (np.ndarray): 长度为 N 的角度曲线数组（dtype float64），单位为余弦值。

    Notes:
        返回的 angle_values 为余弦值而非弧度，便于与坐标空间的极值搜索统一。
    """
    contour = np.asarray(contour, dtype=np.float64)
    if len(contour) == 0:
        return contour, np.empty(0, dtype=np.float64)

    logger.debug(
        f"compute_lateral_angle_profile: contour points={len(contour)}, distance={distance:.1f}"
    )

    angle_values = np.asanyarray(
        [
            compute_lateral_angle_at_point(contour, pt, distance=distance)[0]
            for pt in contour
        ],
        dtype=np.float64,
    )

    return contour, angle_values


def _interpolate_point_at_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    neck_pt: np.ndarray,
    distance: float,
) -> np.ndarray | None:
    """在线段 p1-p2 上内插出距离 neck_pt 恰为 distance 的点（若存在）。"""
    p1_xy = np.asarray(p1, dtype=np.float64)[:2]
    p2_xy = np.asarray(p2, dtype=np.float64)[:2]
    t = segment_circle_intersection(
        p1_xy, p2_xy, np.asarray(neck_pt, dtype=np.float64)[:2], distance
    )
    if t is None:
        return None
    return p1_xy + t * (p2_xy - p1_xy)


def _select_side_point(
    contour_xy: np.ndarray,
    dists: np.ndarray,
    neck_pt: np.ndarray,
    distance: float,
    nearest_idx: int,
    side: str,
    snap_rel_tol: float = 0.01,
) -> np.ndarray:
    """沿轮廓在指定一侧选择与 neck_pt 距离约为 distance 的点（优先精确内插）。

    规则：优先选择与目标距离误差最小的邻点；若误差在 snap_rel_tol 内则直接取该点；
    否则尝试在相邻两点上内插；若内插失败则退回到较近的邻点。
    """
    if side == "prev":
        candidate_indices = np.arange(0, nearest_idx, dtype=int)
        fallback_idx = 0
    else:
        candidate_indices = np.arange(nearest_idx + 1, len(contour_xy), dtype=int)
        fallback_idx = len(contour_xy) - 1

    if len(candidate_indices) == 0:
        fallback_pt = contour_xy[fallback_idx, :2].copy()
        return fallback_pt

    target_errors = np.abs(dists[candidate_indices] - distance)
    p1_idx = int(candidate_indices[int(np.argmin(target_errors))])
    p1 = contour_xy[p1_idx, :2].copy()
    p1_dist = float(dists[p1_idx])
    p1_rel_diff = abs(p1_dist - distance) / max(abs(distance), 1e-9)

    if p1_rel_diff <= snap_rel_tol:
        return p1

    neighbor_indices = [
        neighbor_idx
        for neighbor_idx in (p1_idx - 1, p1_idx + 1)
        if 0 <= neighbor_idx < len(contour_xy)
        and neighbor_idx != p1_idx
        and (
            (side == "prev" and neighbor_idx < nearest_idx)
            or (side == "next" and neighbor_idx > nearest_idx)
        )
    ]
    if not neighbor_indices:
        return p1

    p2_idx = min(
        neighbor_indices, key=lambda neighbor_idx: abs(dists[neighbor_idx] - distance)
    )
    p2 = contour_xy[p2_idx, :2].copy()
    p2_dist = float(dists[p2_idx])

    interp_pt = _interpolate_point_at_distance(p1, p2, neck_pt, distance)
    if interp_pt is not None:
        return interp_pt

    fallback_pt = p1 if abs(p1_dist - distance) <= abs(p2_dist - distance) else p2
    return fallback_pt


def _find_contour_neighbors(
    contour: np.ndarray,
    pt: np.ndarray,
    distance: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """沿轮廓取候选点的前后邻点（利用 CW 顺序：index-1 = before, index+1 = after）。

    轮廓已保证为顺时针顺序，前后邻点直接在数组上取相邻 index 即可。
    `distance` 参数保留仅用于兼容调用侧，不再参与搜索。

    Returns:
        (pt_before, pt_after)：轮廓遍历方向的前后点。
    """
    contour_xy = np.asarray(contour, dtype=np.float64)
    n = len(contour_xy)
    if n < _MIN_CONTOUR_POINTS:
        return contour_xy[0, :2].copy() if n else np.zeros(2), contour_xy[
            -1, :2
        ].copy() if n else np.zeros(2)

    neck_xy = np.asarray(pt, dtype=np.float64)
    dists = np.linalg.norm(contour_xy[:, :2] - neck_xy[:2], axis=1)
    idx = int(np.argmin(dists))

    pt_before = contour_xy[(idx - 1) % n, :2].copy()
    pt_after = contour_xy[(idx + 1) % n, :2].copy()
    return pt_before, pt_after


def _compute_angle_and_cosine(
    pt_prev: np.ndarray,
    pt_next: np.ndarray,
    pt: np.ndarray,
) -> tuple[float, float, float, float]:
    """计算顺时针长轴转角与余弦值。

    给定轮廓遍历顺序的前后两点与候选点，构造向量并计算：
    1. 余弦值（向量夹角的余弦，值域 [-1, 1]）
    2. 顺时针转角（从 before→cand_pt 到 after→cand_pt，值域 [0°, 360°)，< 180°=顺时针转，> 180°=逆时针转）

    Args:
        pt_prev: 轮廓遍历方向的前一点坐标（至少包含 x,y）。
        pt_next: 轮廓遍历方向的后一点坐标（至少包含 x,y）。
        pt: 候选点坐标（至少包含 x,y）。

    Returns:
        Tuple 下列顺序：
        - cosine (float): 向量夹角余弦值。
        - clockwise_deg (float): 顺时针方向的角度，单位度，值域 [0, 360)。
        - dist_prev (float): pt_prev 到 pt 的欧氏距离。
        - dist_next (float): pt_next 到 pt 的欧氏距离。

    Notes:
        若任一向量长度小于 1e-9，返回退化值 (cosine=1.0, clockwise=0.0)。
    """
    v_before = pt_prev[:2] - pt[:2]
    v_after = pt_next[:2] - pt[:2]
    dist_before = float(np.linalg.norm(v_before))
    dist_after = float(np.linalg.norm(v_after))

    if dist_before < _DEGENERATE_DIST_EPSILON or dist_after < _DEGENERATE_DIST_EPSILON:
        logger.info(
            f"_compute_angle_and_cosine: degenerate distances "
            f"before={dist_before:.3e}, after={dist_after:.3e} at cand_pt={pt[:2]}"
        )
        return 1.0, 0.0, dist_before, dist_after

    dot = np.dot(v_before, v_after)
    cosine = float(dot / (dist_before * dist_after))

    cross = float(v_before[0] * v_after[1] - v_before[1] * v_after[0])
    signed_deg = float(np.degrees(np.arctan2(cross, dot)))  # [-180, 180]
    clockwise_deg = (360.0 - signed_deg) % 360.0  # [0, 360)

    return cosine, clockwise_deg, dist_before, dist_after
