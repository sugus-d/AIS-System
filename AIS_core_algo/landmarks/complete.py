"""扁平 landmarks 缺失键补全 — 用训练集平均 landmarks 拟合相似变换映射回当前 mesh。

预测/标注链路中算法检测可能缺 `waist_lower` 或部分 spine 点。补全策略：
已检测到的点对（当前 mesh ↔ 训练集平均模板）拟合 Umeyama 相似变换，
缺失键取平均坐标变换回当前 mesh 物理空间，再映射到最近网格顶点。
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from landmarks.constants import FLAT_KEYS
from landmarks.mean_landmarks import MEAN_LANDMARKS
from parameterization.procrustes import compute_procrustes

# 补全所需的最少已知点对数（Procrustes 相似变换 ≥3 对非共线）
_MIN_MATCH_POINTS = 3


def _to_jsonable(value: object) -> object:
    """坐标值统一转 JSON 可序列化类型（自动检测的 landmark 是 ndarray 切片）。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def complete_landmarks_flat(flat: dict, mesh: o3d.geometry.TriangleMesh) -> dict:
    """补全扁平 landmarks 到 18 点：缺失键取训练集平均 → 相似变换 → mesh 最近顶点。

    Args:
        flat: 扁平 landmarks（可能缺 waist_lower / spine 点）。
        mesh: ROI 网格（用于最近顶点映射）。

    Returns:
        完整 18 键扁平 landmarks（值均为 JSON 可序列化 list；缺失键为 mesh 顶点坐标）。
    """
    missing = [key for key in FLAT_KEYS if key not in flat]
    if not missing:
        return {key: _to_jsonable(value) for key, value in flat.items()}

    known = {key: flat[key] for key in FLAT_KEYS if key in flat}
    if len(known) < _MIN_MATCH_POINTS:
        # 已知点过少无法拟合变换，保持原样（调用方会因 landmarks 不完整报错）
        return {key: _to_jsonable(value) for key, value in flat.items()}

    template_pts = np.array([MEAN_LANDMARKS[key] for key in known], dtype=np.float64)
    current_pts = np.array([known[key] for key in known], dtype=np.float64)
    # 模板 → 当前 mesh 的相似变换（Procrustes/Umeyama，2D/3D 通用）
    scale, rotation, translation = compute_procrustes(template_pts, current_pts)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        return {key: _to_jsonable(value) for key, value in flat.items()}
    kd_tree = cKDTree(vertices)
    result = {key: _to_jsonable(value) for key, value in flat.items()}
    for key in missing:
        if key not in MEAN_LANDMARKS:
            continue
        template_pt = np.asarray(MEAN_LANDMARKS[key], dtype=np.float64)
        mapped = scale * rotation @ template_pt + translation
        _, nearest_idx = kd_tree.query(mapped)
        result[key] = vertices[nearest_idx].tolist()
    return result
