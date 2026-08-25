"""肩臂转点（shoulder_transition）检测 — 2D 内部处理、符合 spec 的签名与返回。

实现要点：
- 输入轮廓均为 (N,2) 或 (N,3)，内部强制使用 (N,2) 进行计算并返回 (2,2) 的肩臂转点坐标。
- 不再接受 y_range 与 waist_points 参数；axilla_debug 用于提供边界与 has_arm 标记。
"""

import numpy as np

from landmarks.angle import compute_lateral_angle_at_point
from landmarks.geometry import to_2d
from landmarks.signal_ops import smooth_contour

from ..constants import ShoulderTransition

_Y_SPAN_EPSILON = 1e-6  # 颈根-腋窝 Y 跨度视为零的阈值（防止除零）


def detect_shoulder_transition(
    left_contour: np.ndarray,
    right_contour: np.ndarray,
    neck_root: np.ndarray,
    axilla_points: np.ndarray,
    axilla_debug: dict,
) -> np.ndarray:
    """按 spec 搜索左右肩臂转点，返回 (2,2) 的 2D 点。

    Args:
        left_contour, right_contour: (N,2) 或 (N,3) 轮廓
        neck_root: (2,2) 或 (2,3)
        axilla_points: (2,2) 或 (2,3)
        axilla_debug: 包含左右侧的字典，可能含 has_arm 与 arm_boundary_x

    Returns:
        shoulder_transition (2,2)。
    """
    nrL_x = float(np.asarray(neck_root)[0, 0])
    nrR_x = float(np.asarray(neck_root)[1, 0])
    axL_y = float(np.asarray(axilla_points)[0, 1])
    axR_y = float(np.asarray(axilla_points)[1, 1])
    nrL_y = float(np.asarray(neck_root)[0, 1])
    nrR_y = float(np.asarray(neck_root)[1, 1])
    neck_width = abs(nrR_x - nrL_x)

    left2 = to_2d(left_contour)
    right2 = to_2d(right_contour)

    has_left_arm = axilla_debug.get("left", {}).get("has_arm", True)
    has_right_arm = axilla_debug.get("right", {}).get("has_arm", True)

    results = []

    for side_name, contour2, ax_y, has_arm, inner_x, inner_y, side_idx in [
        ("left", left2, axL_y, has_left_arm, nrL_x, nrL_y, 0),
        ("right", right2, axR_y, has_right_arm, nrR_x, nrR_y, 1),
    ]:
        arm_boundary_x = axilla_debug.get(side_name, {}).get("arm_boundary_x")
        if arm_boundary_x is None and len(contour2) >= ShoulderTransition.MIN_SEGMENT_LEN:
            ax_pt_x = float(np.asarray(axilla_points)[side_idx, 0])
            if abs(ax_pt_x) > 1.0:
                if side_name == "left":
                    arm_boundary_x = ax_pt_x - ShoulderTransition.OUTER_MARGIN
                else:
                    arm_boundary_x = ax_pt_x + ShoulderTransition.OUTER_MARGIN
        if arm_boundary_x is None or len(contour2) < ShoulderTransition.MIN_SEGMENT_LEN:
            results.append(np.zeros(2))
            continue

        outer_x = float(arm_boundary_x)

        # smooth only x,y -> 保证 (N,2)
        smoothed = smooth_contour(contour2, sigma=ShoulderTransition.GAUSSIAN_SIGMA, mode="nearest")

        # 在颈根与肩外侧边界框内搜索：X 范围 [inner_x, outer_x] 框定颈→肩过渡区，
        # Y > ax_y 排除腋窝以下的无关节段，减少噪声干扰
        N = len(smoothed)
        lo, hi = min(inner_x, outer_x), max(inner_x, outer_x)
        box_x = (smoothed[:, 0] >= lo) & (smoothed[:, 0] <= hi)
        box_y = smoothed[:, 1] > float(ax_y)
        box_mask = box_x & box_y

        box_indices = np.where(box_mask)[0]
        cand_indices = []
        if len(box_indices) >= ShoulderTransition.MIN_SEGMENT_LEN:
            # 断点容差：轮廓上的候选点如果被大间隔断开（如手臂与躯干间隙），
            # 说明跨越了不同解剖结构，应拆分为独立段
            split_points = np.where(np.diff(box_indices) > ShoulderTransition.GAP_TOLERANCE)[0] + 1
            segments = np.split(box_indices, split_points)
            max_len = max(len(seg) for seg in segments)
            longest_segs = [seg for seg in segments if len(seg) == max_len]
            if len(longest_segs) == 1:
                longest = longest_segs[0]
            else:
                # 等长时选 Y 更接近颈根的段——肩转点更靠近脖子水平
                longest = min(
                    longest_segs,
                    key=lambda s: abs(np.mean(smoothed[s, 1]) - float(np.mean(neck_root[:, 1]))),
                )
            cand_indices = (
                list(longest)
                if len(longest) >= ShoulderTransition.MIN_SEGMENT_LEN
                else list(box_indices)
            )

        # 长轴转角：沿轮廓前后取弧长点，计算切线偏离垂直方向的角度。
        # 肩臂转点处轮廓方向发生显著变化，角度是区分转点的关键特征
        long_axis_angles = np.full(N, 180.0, dtype=np.float64)
        for idx in cand_indices:
            pt = smoothed[idx]
            _cos, clockwise_deg, *_ = compute_lateral_angle_at_point(smoothed, pt, distance=ShoulderTransition.DISTANCE)
            long_axis_angles[idx] = float(clockwise_deg)

        peak_index = -1
        if cand_indices:
            angles_in_cand = long_axis_angles[cand_indices]
            cand_arr = np.array(cand_indices)

            # 有手臂时肩臂交界呈凹角（角度较小），无手臂时外侧上方呈弯折（角度较大），
            # 因此采用相反的角过滤方向
            if has_arm:
                mask = angles_in_cand <= ShoulderTransition.ANGLE_ARM_MAX
            else:
                mask = angles_in_cand >= ShoulderTransition.ANGLE_NOARM_MIN

            filtered_indices = cand_arr[mask]

            if len(filtered_indices) > 0:
                scores = [
                    _score_candidate(smoothed[idx], inner_x, inner_y, ax_y, neck_width, has_arm)
                    for idx in filtered_indices
                ]
                peak_index = int(filtered_indices[np.argmax(scores)])
            else:
                # 角度过滤过严时放弃角度约束，用位置评分兜底——宁可选错位也不漏检
                scores = [
                    _score_candidate(smoothed[idx], inner_x, inner_y, ax_y, neck_width, has_arm) for idx in cand_arr
                ]
                peak_index = int(cand_arr[np.argmax(scores)])

        if peak_index < 0:
            # 全路径回退：上述所有方法都无法定位时，用最外侧点作为肩转点保底
            if cand_indices:
                cand_arr = np.asarray(cand_indices)
                idx_fn = np.argmin if side_name == "left" else np.argmax
                peak_index = int(cand_arr[idx_fn(smoothed[cand_arr, 0])])
            else:
                idx_fn = np.argmin if side_name == "left" else np.argmax
                peak_index = int(idx_fn(smoothed[:, 0]))

        results.append(smoothed[peak_index].copy())

    return np.stack(results)


def _score_candidate(
    cand_pt: np.ndarray,
    nr_x: float,
    nr_y: float,
    ax_y: float,
    neck_width: float,
    has_arm: bool,
) -> float:
    """给候选点评分，综合 x/y 位置合理性。"""
    dx = abs(float(cand_pt[0]) - nr_x)
    dx_ratio = dx / neck_width * 100.0 if neck_width > 0 else 50.0
    y_ratio = (float(cand_pt[1]) - ax_y) / (nr_y - ax_y) if abs(nr_y - ax_y) > _Y_SPAN_EPSILON else 0.85

    # y_score: 每偏离 0.05 扣 1 分，上限 -10
    y_score = -min(abs(y_ratio - 0.85) / 0.05, 10.0)

    if has_arm:
        # 有手臂时肩臂转点可在较宽的 X 范围内（手臂向外伸展），
        # 使用"死区"评分：dx_ratio 在 25%~85% 区域不扣分，避免过度惩罚
        x_score = -min(max(abs(dx_ratio - 55.0) - 30.0, 0.0) / 5.0, 10.0)
        return x_score + y_score
    else:
        # 无手臂时外侧弯折点 X 位置更确定，缩小评分敏感度；
        # 同时 Y 偏高说明更接近肩峰顶点，加分鼓励靠上选择
        x_score = -min(abs(dx_ratio - 55.0) / 5.0, 10.0)
        y_bonus = (y_ratio - 0.85) * 3.0
        return x_score + y_score + y_bonus
