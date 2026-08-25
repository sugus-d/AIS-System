"""Landmark I/O for Ground Truth JSON format.

Parses ground_truth.json from the labeling platform / prelabel pipeline and
matches each 3D landmark to its nearest vertex on the mesh.
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d

from landmarks.constants import FLAT_KEYS

_MIN_LANDMARK_MATCHES = 4  # 模板匹配地标数下限（不足则无法参数化）


def parse_landmarks_json(json_path: str | Path) -> dict[str, np.ndarray]:
    """解析扁平 18 键 Ground Truth JSON，返回语义名坐标映射。

    JSON 格式（全链路统一契约，FLAT_KEYS 单源）：
        {"neck_root_L": [x,y,z], "neck_root_R": [x,y,z], ...,
         "thoracic_spine_point": [x,y,z], "_features": {...}}

    Args:
        json_path: ground_truth.json 文件路径。

    Returns:
        语义键（FLAT_KEYS，如 "neck_root_L" / "thoracic_spine_point"）到
        (3,) float64 XYZ 坐标的映射字典。仅返回存在的坐标。

    Raises:
        FileNotFoundError: json_path 不存在。
        ValueError: 无法解析任何地标。
    """
    with open(json_path) as f:
        gt = json.load(f)

    landmarks: dict[str, np.ndarray] = {}
    for key in FLAT_KEYS:
        pt = gt.get(key)
        if pt is not None:
            landmarks[key] = np.asarray(pt, dtype=np.float64)

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
