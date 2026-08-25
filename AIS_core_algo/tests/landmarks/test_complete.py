"""complete_landmarks_flat 回归测试 — 自动检测 ndarray 值转 JSON list。

回归背景：extract_landmarks 输出 ndarray 切片值，complete 缺键为空时原样
透传导致 json.dumps 抛 TypeError（predict auto 端到端炸）。修复：所有返回
路径统一转 JSON 可序列化类型。
"""

import json

import numpy as np
import open3d as o3d

from landmarks.complete import complete_landmarks_flat
from landmarks.constants import FLAT_KEYS


def _box_mesh() -> o3d.geometry.TriangleMesh:
    """10³ 立方体网格（补全目标顶点）。"""
    mesh = o3d.geometry.TriangleMesh.create_box(width=10, height=10, depth=10)
    mesh.compute_vertex_normals()
    return mesh


def _ndarray_flat() -> dict:
    """模拟 extract_landmarks 输出：18 键 ndarray 切片值。"""
    rng = np.random.default_rng(0)
    return {key: rng.uniform(0, 10, size=3) for key in FLAT_KEYS}


def _assert_jsonable_18(result: dict) -> None:
    """断言 18 键齐全、值均为 list、可 json.dumps。"""
    assert set(result) == set(FLAT_KEYS)
    assert all(isinstance(result[key], list) for key in FLAT_KEYS)
    json.dumps(result)


def test_json_serializable_when_complete() -> None:
    """18 键齐全（ndarray 值）→ 返回全 list，可 json.dumps。"""
    result = complete_landmarks_flat(_ndarray_flat(), _box_mesh())
    _assert_jsonable_18(result)


def test_json_serializable_with_missing_key() -> None:
    """缺 2 键 → 均值补全路径，已知键与补全键均为 list。"""
    flat = _ndarray_flat()
    del flat["waist_lower_L"]
    del flat["waist_lower_R"]
    result = complete_landmarks_flat(flat, _box_mesh())
    _assert_jsonable_18(result)


def test_few_known_points_passthrough_jsonable() -> None:
    """已知点 <3 无法拟合变换 → 原样透传但值转 list。"""
    flat = {key: np.array([1.0, 2.0, 3.0]) for key in ["neck_root_L", "neck_root_R"]}
    result = complete_landmarks_flat(flat, _box_mesh())
    assert all(isinstance(result[key], list) for key in result)
    json.dumps(result)
