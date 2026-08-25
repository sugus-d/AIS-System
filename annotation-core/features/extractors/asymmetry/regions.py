"""225 个左右对称解剖区域定义 + UV 空间多边形遮罩。

每个区域由一对左右多边形构成：
- 左多边形：只含 L 侧边缘地标 + 脊柱中线地标
- 右多边形：只含 R 侧边缘地标 + 相同脊柱中线地标

区域生成逻辑与 ``landmark_regions/_regions.py`` 的 V2(bilateral) 版本共享
（避免 fork 双份 225 对生成代码），本模块保留公共 API：
``build_region_polygons`` / ``points_in_polygon`` / ``mask_vertices`` / ``_get_pairs``。
"""

from __future__ import annotations

import numpy as np

from features.extractors.asymmetry.landmark_regions._regions import (
    _get_bilateral,
    _points_in_polygon,
    build_candidate_polygons,
)


def _get_pairs() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """获取全部 225 个左右对称区域对（复用 landmark_regions 的 V2 生成器）。"""
    return _get_bilateral()


def build_region_polygons() -> list[dict]:
    """构建全部 225 个候选区域多边形定义。

    Returns:
        list[dict]: 每个 dict 包含 id, name, left_polygon, right_polygon。
        输出与 ``landmark_regions.build_candidate_polygons("bilateral")`` 一致。
    """
    return build_candidate_polygons("bilateral")


def points_in_polygon(pts: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """射线法判断点是否在多边形内部（偶数-奇数规则）。

    实现复用 ``landmark_regions._regions._points_in_polygon``（同算法）。

    Args:
        pts: (N, 2) 查询点。
        polygon: (M, 2) 多边形顶点，CW 或 CCW 均可。

    Returns:
        (N,) bool 数组，在多边形内部（含边界）为 True。
    """
    return _points_in_polygon(pts, polygon)


def mask_vertices(uv: np.ndarray, candidates: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """批量计算所有候选区域的左右顶点遮罩。

    对比逐区域循环调用 ``points_in_polygon``，此处预先收集所有多边形，
    统一完成遮罩计算。

    Args:
        uv: (N, 2) UV 坐标。
        candidates: ``build_region_polygons()`` 的输出。

    Returns:
        (left_masks, right_masks, region_names):
            left_masks[i]: (N,) bool 数组 — 第 i 个区域左侧包含的顶点。
            right_masks[i]: (N,) bool 数组 — 第 i 个区域右侧包含的顶点。
            region_names[i]: 区域名称字符串。
    """
    left_masks: list[np.ndarray] = []
    right_masks: list[np.ndarray] = []
    names: list[str] = []

    for cand in candidates:
        left_masks.append(points_in_polygon(uv, cand["left_polygon"]))
        right_masks.append(points_in_polygon(uv, cand["right_polygon"]))
        names.append(cand["name"])

    return left_masks, right_masks, names
