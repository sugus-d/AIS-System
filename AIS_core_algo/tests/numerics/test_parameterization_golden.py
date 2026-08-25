"""M5 参数化黄金测试 — 地标匹配 + 调和参数化 + 测地边界。

输入：固定 ROI 网格（roi_S0006.ply）+ extract_landmarks 的真实地标。
geodesic_boundary 要求 k 与 TEMPLATE_LANDMARKS 全量对齐，缺失地标用确定性
合法顶点补齐（与 _generate_golden.py 相同策略）。
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from tests.numerics.conftest import assert_golden, DATA_DIR

ROI_PLY = DATA_DIR / "mesh" / "roi_S0006.ply"

GOLDEN = {
    "param_k": ("(18,)", "233830.0000000000", "5e77cb5f850d3cbedac965a5c2530b95"),
    "param_y": ("(18, 2)", "-10.0000000000", "38c4a7a91de7ef598175dc38f3f138f0"),
    "param_uv": ("(25495, 2)", "-23020.0088981293", "e89283fe475f2f2d18d1e6c13ca031de"),
    "param_boundary_v": ("(226,)", "252759.0000000000", "3979dd1dced33029daeda6aafe702e24"),
    "param_boundary_f": ("(1935, 3)", "-955441.4501117410", "4828bc8f495306fc351ac76820d29038"),
}


def _flatten_landmarks(lms: dict) -> dict:
    """extract_landmarks 已输出扁平语义键；过滤出纯 landmark 键（FLAT_KEYS）。"""
    from landmarks.constants import FLAT_KEYS

    return {key: lms[key] for key in FLAT_KEYS if key in lms}


@pytest.mark.slow
def test_parameterization_golden() -> None:
    """地标匹配 + 调和参数化 + 测地边界与黄金值逐位一致。"""
    from landmarks.extract import extract_landmarks
    from parameterization.geodesic_cut import geodesic_boundary
    from parameterization.harmonic import harmonic_parameterize
    from parameterization.landmark_io import find_landmark_vertices
    from parameterization.template import TEMPLATE_LANDMARKS

    if not ROI_PLY.exists():
        pytest.skip(f"ROI mesh 缺失（敏感数据不随仓库分发，本地放置后运行）: {ROI_PLY}")
    roi_mesh = o3d.io.read_triangle_mesh(str(ROI_PLY))
    lms = extract_landmarks(roi_mesh)
    flat = _flatten_landmarks(lms)

    k, y_uv = find_landmark_vertices(roi_mesh, flat, TEMPLATE_LANDMARKS)
    assert_golden("param_k", k, *GOLDEN["param_k"])
    assert_golden("param_y", y_uv, *GOLDEN["param_y"])

    _uv_mesh, uv = harmonic_parameterize(roi_mesh, k, y_uv)
    assert_golden("param_uv", uv, *GOLDEN["param_uv"])

    simple = roi_mesh.simplify_quadric_decimation(target_number_of_triangles=3000)
    V = np.asarray(simple.vertices, dtype=np.float64)
    F = np.asarray(simple.triangles)
    k_simple, y_simple = find_landmark_vertices(simple, flat, TEMPLATE_LANDMARKS)
    matched_names = [name for name in TEMPLATE_LANDMARKS if name in flat]
    k_map = dict(zip(matched_names, k_simple, strict=False))
    y_map = dict(zip(matched_names, y_simple, strict=False))
    k_full, y_full = [], []
    for offset, (name, uv_target) in enumerate(TEMPLATE_LANDMARKS.items()):
        if name in k_map:
            k_full.append(k_map[name])
            y_full.append(y_map[name])
        else:
            k_full.append(int(np.argmax(V[:, 1] - offset * 1e-6)))
            y_full.append(np.asarray(uv_target, dtype=float))
    outer_names = [name for name in TEMPLATE_LANDMARKS if not name.endswith("_spine_point")]
    boundary_v, boundary_f = geodesic_boundary(V, F, np.array(k_full), np.array(y_full), outer_names)
    assert_golden("param_boundary_v", boundary_v, *GOLDEN["param_boundary_v"])
    assert_golden("param_boundary_f", boundary_f, *GOLDEN["param_boundary_f"])
