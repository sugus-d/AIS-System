"""M6 utils 数值地基黄金测试 — 合成轮廓/真实网格的纯函数。

全部为确定性输入，毫秒级。
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.numerics.conftest import assert_golden, DATA_DIR, RNG_SEED

GOLDEN = {
    "u_resample": ("(65, 2)", "4067.9596518863", "b744b7d35f40f97943a0709e3711c21e"),
    "u_lower": ("(201, 2)", "15634.9719463444", "3443dcc23fe7fa9f4672774e01d1a927"),
    "u_smooth": ("(200, 2)", "12669.4141305808", "59a8e449ef38c3e469b52b98e449b347"),
    "u_derivative": ("(200,)", "-0.1186170701", "4b82e8d7d8fc3cb8add5f7d1652eda3a"),
    "u_ccw": ("(1,)", "1.0000000000", "55a54008ad1ba589aa210d2629c1df41"),
    "u_angle_sampled": ("(200, 2)", "12670.2959009873", "4700f3dc67f73914ecd4a4dcfdc290c2"),
    "u_angle_values": ("(200,)", "-195.4653376479", "de34200e1e0beb77dd309640e8f3b3ac"),
    "u_radius": ("(1,)", "8.2268336701", "6b329792def300b55b5d0b6b434b4105"),
    "u_lift": ("(2, 3)", "-909.2504310608", "2cc242b159d9b227e11a13ff122a13e8"),
}


def _contour() -> np.ndarray:
    """确定性合成轮廓：半圆 + 固定种子噪声。"""
    rng = np.random.default_rng(RNG_SEED)
    theta = np.linspace(0, np.pi, 200)
    return np.column_stack([100 * np.cos(theta), 100 * np.sin(theta)]) + rng.normal(0, 0.1, size=(200, 2))


def test_contour_signal_ops_golden() -> None:
    """contour 重采样/下边界、平滑、导数与黄金值一致。"""
    from landmarks.contour import extract_lower_boundary_per_integer_x, resample_polyline_uniform
    from landmarks.signal_ops import compute_derivatives_from_xy, smooth_contour

    contour = _contour()
    assert_golden("u_resample", resample_polyline_uniform(contour, step=5.0), *GOLDEN["u_resample"])
    assert_golden("u_lower", extract_lower_boundary_per_integer_x(contour), *GOLDEN["u_lower"])
    smoothed = smooth_contour(contour)
    assert_golden("u_smooth", smoothed, *GOLDEN["u_smooth"])
    derivative = compute_derivatives_from_xy(smoothed[:, 0], smoothed[:, 1])
    assert_golden("u_derivative", derivative, *GOLDEN["u_derivative"])


def test_geometry_angle_golden() -> None:
    """轮廓方向、角度曲线与黄金值一致。"""
    from landmarks.angle import compute_lateral_angle_profile
    from landmarks.geometry import is_contour_ccw

    contour = _contour()
    assert_golden("u_ccw", np.array([is_contour_ccw(contour)]), *GOLDEN["u_ccw"])
    sampled_pts, angle_values = compute_lateral_angle_profile(contour, 10.0)
    assert_golden("u_angle_sampled", sampled_pts, *GOLDEN["u_angle_sampled"])
    assert_golden("u_angle_values", angle_values, *GOLDEN["u_angle_values"])


def test_mesh_utils_golden() -> None:
    """顶点半径估计与 2D→3D 提升（固定 ROI 网格）与黄金值一致。"""
    import open3d as o3d

    from utils.mesh import estimate_vertex_radius, lift_2d_to_vertex

    roi_ply = DATA_DIR / "mesh" / "roi_S0006.ply"
    if not roi_ply.exists():
        pytest.skip(f"真实扫描 mesh 缺失（敏感数据不随仓库分发，本地放置后运行）: {roi_ply}")
    roi_mesh = o3d.io.read_triangle_mesh(str(roi_ply))
    vertices = np.asarray(roi_mesh.vertices, dtype=np.float64)
    radius = estimate_vertex_radius(vertices, nb_neighbors=8)
    assert_golden("u_radius", radius, *GOLDEN["u_radius"])
    pts2d = np.array([[0.0, 0.0], [100.0, -100.0]])
    assert_golden("u_lift", lift_2d_to_vertex(vertices, pts2d), *GOLDEN["u_lift"])
