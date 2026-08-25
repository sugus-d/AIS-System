"""Landmark 坐标顺序逻辑校验（不依赖 mesh）—— lifter 校验层。"""

import numpy as np

from ...constants import BILATERAL_LANDMARKS, LANDMARK_NAMES_ZH, LANDMARK_Y_ORDER


def _get_side_value(landmarks: dict, landmark_name: str, side_index: int, axis: int) -> float | None:
    """提取某侧某轴向的值。side_index=0 → L, 1 → R。"""
    pts = landmarks.get(landmark_name, [])
    if len(pts) > side_index and isinstance(pts[side_index], (list, np.ndarray)) and len(pts[side_index]) > axis:
        return float(pts[side_index][axis])
    return None


def _validate_coordinate_order(landmarks: dict) -> list[dict]:
    """校验 landmarks 坐标顺序合理性。

    检查项目：
    1. 双侧对称点 L_x < R_x（左右反了是错误）
    2. 脊柱点 P 在对应 bilateral pair 的 L_x ~ R_x 范围内（超出是警告）
    3. Y 层级：左右侧各自从高到低顺序（异常是警告）

    不依赖 mesh，仅基于 landmarks 数据做逻辑校验。

    Returns:
        list[dict]: issues 列表，每个 issue 包含 type/severity/message/location。
    """
    issues: list[dict] = []
    SPINE_X_PAIRS: list[str | None] = [
        "neck_root",
        "scapular_peaks",
        "axilla",
        "waist",
        "waist_lower",
        None,  # P5（中背）无对应 pair
    ]

    # ── 1. X 顺序：每对 bilateral 的 L_x < R_x ──
    for name in BILATERAL_LANDMARKS:
        lx = _get_side_value(landmarks, name, 0, 0)
        rx = _get_side_value(landmarks, name, 1, 0)
        if lx is not None and rx is not None and lx >= rx:
            zh = LANDMARK_NAMES_ZH.get(name, name)
            issues.append(
                {
                    "type": "x_order",
                    "landmark": name,
                    "severity": "error",
                    "message": f"{zh} L_x({lx:.1f}) ≥ R_x({rx:.1f})，左右侧可能反了，请检查",
                }
            )

    # ── 2. 脊柱点 X 范围：L_x < P_x < R_x ──
    spine = landmarks.get("spine_points", [])
    for i, pair_name in enumerate(SPINE_X_PAIRS):
        if pair_name is None:
            continue
        if i >= len(spine) or not isinstance(spine[i], (list, np.ndarray)):
            continue
        px = float(spine[i][0])
        lx = _get_side_value(landmarks, pair_name, 0, 0)
        rx = _get_side_value(landmarks, pair_name, 1, 0)
        if lx is not None and rx is not None and not (lx < px < rx):
            zh = LANDMARK_NAMES_ZH.get(pair_name, pair_name)
            issues.append(
                {
                    "type": "spine_x_range",
                    "landmark": "spine_points",
                    "index": i,
                    "severity": "warning",
                    "message": f"脊柱 P{i} 的 X({px:.1f}) 不在 {zh} L({lx:.1f}) ~ R({rx:.1f}) 范围内",
                }
            )

    # ── 3. Y 层级：左右侧各自递减 ──
    for side_index, side_label in enumerate(("L", "R")):
        prev_name = None
        prev_y = None
        for curr_name in LANDMARK_Y_ORDER:
            cy = _get_side_value(landmarks, curr_name, side_index, 1)
            if cy is None:
                prev_name, prev_y = None, None
                continue
            if prev_y is not None and prev_y <= cy:
                zh_prev = LANDMARK_NAMES_ZH.get(prev_name, prev_name) if prev_name else ""
                zh_curr = LANDMARK_NAMES_ZH.get(curr_name, curr_name)
                issues.append(
                    {
                        "type": "y_order",
                        "landmark": curr_name,
                        "severity": "warning",
                        "message": f"{side_label}侧 {zh_prev} Y({prev_y:.1f}) 应高于 {zh_curr} Y({cy:.1f})，当前层级异常",
                    }
                )
            prev_name = curr_name
            prev_y = cy

    return issues
