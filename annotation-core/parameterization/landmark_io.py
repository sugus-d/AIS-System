"""Landmark I/O for Ground Truth JSON format.

Parses ground_truth.json from the labeling platform / prelabel pipeline and
matches each 3D landmark to its nearest vertex on the mesh.
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from landmarks.constants import SPINE_POINT_COUNT

_MIN_LANDMARK_MATCHES = 4  # 模板匹配地标数下限（不足则无法参数化）

# 左右成对地标名（与 ground_truth.json 的键一致）
_BILATERAL_NAMES = [
    "neck_root",
    "shoulder_transition",
    "scapular_peaks",
    "axilla",
    "waist",
    "waist_lower",
]


def parse_landmarks_json(json_path: str | Path) -> dict[str, np.ndarray]:
    """解析 Ground Truth JSON，返回短名坐标映射。

    JSON 格式（预标注 batch_prelabel / 标注平台 save_landmarks 产出）：
        {"neck_root": {"L": [x,y,z], "R": [x,y,z]}, ...,
         "spine_points": [[x,y,z], ...], "_features": {...}}
    左右成对地标展开为短名（neck_root_L/R），脊柱点展开为 spine_P0..P5。

    Args:
        json_path: ground_truth.json 文件路径。

    Returns:
        短名（如 "neck_root_L"）到 (3,) float64 XYZ 坐标的映射字典。
        仅返回存在的坐标。

    Raises:
        FileNotFoundError: json_path 不存在。
        ValueError: 无法解析任何地标。
    """
    with open(json_path) as f:
        gt = json.load(f)

    landmarks: dict[str, np.ndarray] = {}
    for name in _BILATERAL_NAMES:
        pair = gt.get(name)
        if not isinstance(pair, dict):
            continue
        for side in ("L", "R"):
            pt = pair.get(side)
            if pt is not None:
                landmarks[f"{name}_{side}"] = np.asarray(pt, dtype=np.float64)

    spine = gt.get("spine_points")
    if isinstance(spine, list):
        for i, pt in enumerate(spine):
            if i >= SPINE_POINT_COUNT:
                break
            if pt is not None:
                landmarks[f"spine_P{i}"] = np.asarray(pt, dtype=np.float64)

    if not landmarks:
        raise ValueError(f"No landmarks parsed from {json_path}")

    return landmarks


def find_landmark_vertices(
    mesh: o3d.geometry.TriangleMesh,
    landmarks: dict[str, np.ndarray],
    template: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """将地标匹配到最近的网格顶点，返回约束数组。

    对于同时存在于 JSON 和模板中的每个地标名称，
    找到距离 3D 地标位置最近的网格顶点，并记录其目标 UV 坐标。

    Args:
        mesh:     Open3D 三角网格（去衣后表面）。
        landmarks: parse_landmarks_json 的输出。
        template: parameterization.template 中的 TEMPLATE_LANDMARKS。

    Returns:
        (k, y):
            k: (M,) int64 顶点索引数组。
            y: (M, 2) float64 目标 (u, v) 坐标数组。
            M 为匹配到的地标数量（通常为 18）。

    Raises:
        ValueError: 匹配到的地标少于 4 个。
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    k_list: list[int] = []
    y_list: list[tuple[float, float]] = []

    for name, target_uv in template.items():
        if name not in landmarks:
            continue
        pt = landmarks[name][:3]  # (x, y, z)
        # Euclidean distance to all vertices, find nearest
        dists = np.linalg.norm(vertices - pt, axis=1)
        nearest = int(np.argmin(dists))
        k_list.append(nearest)
        y_list.append(target_uv)

    if len(k_list) < _MIN_LANDMARK_MATCHES:
        raise ValueError(
            f"Only {len(k_list)} landmarks matched (need >= 4). Check that the JSON landmarks and template names align."
        )

    k = np.array(k_list, dtype=np.int64)
    y = np.array(y_list, dtype=np.float64)
    return k, y
