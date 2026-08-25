"""Tests for lateral_profile: contour extraction and splitting."""

import glob
import os
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d
import pytest

matplotlib.use("Agg")

from landmarks.lateral_profile import (
    _extract_body_contour,
    _split_contours,
    compute_width_profile,
    extract_split_contours,
)
from mesh.preprocess import preprocess_to_vertices
from utils.mesh import load_mesh_by_project

# ── helpers ────────────────────────────────────────────────────────────────


def shoelace_area(pts: np.ndarray) -> float:
    """有符号面积：>0 CCW，<0 CW。"""
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def is_cw(pts: np.ndarray) -> bool:
    return pts.shape[0] >= 3 and shoelace_area(pts) < 0


def _make_cw_contour(
    left_x: float = -50.0,
    right_x: float = 50.0,
    n_right: int = 15,
    n_left: int = 15,
    n_top: int = 3,
    y_top: float = 100.0,
    y_bot: float = -100.0,
) -> np.ndarray:
    """CW 轮廓：top → right side → bottom → left side → top。

    返回的轮廓 is_left[0] != is_left[-1]（无 wrap 问题）。
    """
    pts = []
    for i in range(n_top):
        frac = i / max(n_top - 1, 1)
        pts.append([0.0, y_top - frac * 5.0])
    for y in np.linspace(y_top - 5.0, y_bot, n_right):
        pts.append([right_x, y])
    for y in np.linspace(y_bot, y_top - 5.0, n_left):
        pts.append([left_x, y])
    arr = np.array(pts, dtype=np.float64)
    if not is_cw(arr):
        arr = arr[::-1]
    return arr


# ── Fixtures ───────────────────────────────────────────────────────────────

MESH_ID = "17-10745"
# 测试已按领域分组（tests/landmarks/），上溯两级到项目根
_ROOT = Path(__file__).resolve().parents[2]
_MESH_DIR = os.path.normpath(os.path.join(_ROOT, "data", "mesh", MESH_ID))
_HAS_REAL_MESH = len(glob.glob(os.path.join(_MESH_DIR, "STD_fuse_mesh*.ply"))) > 0


@pytest.fixture
def fake_mesh() -> o3d.geometry.TriangleMesh:
    """简易 4 顶点 mesh（直接构造，不写磁盘）。"""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


@pytest.fixture(scope="session")
def real_vertices() -> np.ndarray | None:
    """S0069 预处理后的顶点数组（session 级缓存）。"""
    if not _HAS_REAL_MESH:
        return None
    return preprocess_to_vertices(load_mesh_by_project(MESH_ID))


# ── _extract_body_contour 单元测试 ────────────────────────────────────────


class TestExtractBodyContour:
    def test_empty_input(self):
        result = _extract_body_contour(np.empty((0, 3)))
        assert result.size == 0

    def test_single_vertex(self):
        result = _extract_body_contour(np.array([[1.0, 2.0, 3.0]]))
        assert result.size > 0
        assert result.ndim == 2
        assert result.shape[1] == 2

    def test_two_vertices(self):
        result = _extract_body_contour(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))
        assert result.size > 0

    def test_few_points_triggers_fallback(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=float)
        result = _extract_body_contour(pts)
        assert result.shape[0] >= 3
        assert result.shape[1] == 2

    def test_output_is_cw(self):
        pts = np.random.rand(100, 3) * 200 - 100
        result = _extract_body_contour(pts)
        if result.shape[0] >= 3:
            assert is_cw(result), f"not CW (area={shoelace_area(result):.1f})"

    def test_starts_at_topmost(self):
        pts = np.random.rand(100, 3) * 200 - 100
        result = _extract_body_contour(pts)
        if result.shape[0] >= 3:
            top_idx = int(np.argmax(result[:, 1]))
            assert top_idx == 0

    def test_max_points_parameter_accepted(self):
        n_pts = 200
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        x, y = np.cos(theta) * 100, np.sin(theta) * 200 + 100
        vertices = np.column_stack([x, y, np.zeros(n_pts)])
        for mp in [None, 10, 100, 500]:
            result = _extract_body_contour(vertices, max_points=mp)
            assert isinstance(result, np.ndarray)
            assert result.shape[1] == 2

    def test_alphashape_fallback_on_collinear(self):
        pts = np.array([[i, 0.0, 0.0] for i in range(10)], dtype=float)
        result = _extract_body_contour(pts)
        assert result.shape[0] == 10
        assert result.shape[1] == 2

    # ── 真实 mesh 验证 ──

    @pytest.mark.skipif(not _HAS_REAL_MESH, reason="real mesh not available")
    def test_real_contour_is_cw(self, real_vertices: np.ndarray | None) -> None:
        """真实 mesh 输出的完整轮廓必须是顺时针。"""
        contour = _extract_body_contour(real_vertices)
        assert contour.shape[0] >= 10
        assert is_cw(contour), f"real contour not CW (area={shoelace_area(contour):.1f})"

    @pytest.mark.skipif(not _HAS_REAL_MESH, reason="real mesh not available")
    def test_real_contour_starts_at_top(self, real_vertices: np.ndarray | None) -> None:
        """真实 mesh 轮廓起始点为最高 Y。"""
        contour = _extract_body_contour(real_vertices)
        top_idx = int(np.argmax(contour[:, 1]))
        assert top_idx == 0


# ── _split_contours 单元测试 ──────────────────────────────────────────────


class TestSplitContours:
    def test_empty_input(self):
        left, right = _split_contours(np.empty((0, 2)))
        assert left.size == 0
        assert right.size == 0

    def test_none_input(self):
        left, right = _split_contours(None)
        assert left.size == 0
        assert right.size == 0

    def test_single_point(self):
        contour = np.array([[10.0, 20.0]])
        left, right = _split_contours(contour)
        assert left.shape[0] + right.shape[0] == 1

    def test_two_points(self):
        contour = np.array([[10.0, 0.0], [-10.0, 0.0]])
        left, right = _split_contours(contour)
        assert left.shape[0] + right.shape[0] == 2
        assert np.median(left[:, 0]) < np.median(right[:, 0])

    def test_no_point_loss_on_no_wrap(self):
        contour = _make_cw_contour()
        left, right = _split_contours(contour)
        assert len(left) + len(right) == len(contour)

    def test_both_sides_clockwise(self):
        """concat(left, right) 可通过 cyclic shift 还原到原轮廓。"""
        contour = _make_cw_contour()
        left, right = _split_contours(contour)
        combined = np.concatenate([left, right], axis=0)
        for shift in range(len(contour)):
            if np.allclose(combined, np.roll(contour, -shift, axis=0)):
                break
        else:
            pytest.fail("(left ++ right) 不能通过 cyclic shift 还原到原轮廓")

    def test_x_separation(self):
        contour = _make_cw_contour(left_x=-60, right_x=60)
        left, right = _split_contours(contour)
        assert np.median(left[:, 0]) < np.median(right[:, 0])

    def test_asymmetric_separation(self):
        contour = _make_cw_contour(left_x=-30, right_x=80)
        left, right = _split_contours(contour)
        assert np.median(left[:, 0]) < np.median(right[:, 0])

    def test_both_nonempty(self):
        contour = _make_cw_contour()
        left, right = _split_contours(contour)
        assert len(left) > 0
        assert len(right) > 0

    def test_fallback_median_on_insufficient_transitions(self):
        contour = np.array([[i, 100 - i] for i in range(1, 50)], dtype=float)
        left, right = _split_contours(contour)
        assert len(left) > 0
        assert len(right) > 0

    # ── 真实 mesh 验证 ──

    @pytest.mark.skipif(not _HAS_REAL_MESH, reason="real mesh not available")
    def test_real_no_point_loss(self, real_vertices: np.ndarray | None) -> None:
        """真实轮廓上无点丢失（需先拿完整轮廓）。"""
        contour = _extract_body_contour(real_vertices)
        left, right = _split_contours(contour)
        assert len(left) > 0
        assert len(right) > 0
        # 点数至少保留（若有 wrap 可能丢少量，但不低于 95%）
        ratio = (len(left) + len(right)) / len(contour)
        assert ratio >= 0.95, f"point loss: {ratio:.0%} retained"

    @pytest.mark.skipif(not _HAS_REAL_MESH, reason="real mesh not available")
    def test_real_x_separation(self, real_vertices: np.ndarray | None) -> None:
        """真实轮廓上左中位 < 右中位。"""
        contour = _extract_body_contour(real_vertices)
        left, right = _split_contours(contour)
        assert np.median(left[:, 0]) < np.median(right[:, 0])

    @pytest.mark.skipif(not _HAS_REAL_MESH, reason="real mesh not available")
    def test_real_cyclic_shift_match(self, real_vertices: np.ndarray | None) -> None:
        """concat(left, right) 在真实轮廓上也能 cyclic shift 匹配。"""
        contour = _extract_body_contour(real_vertices)
        left, right = _split_contours(contour)
        combined = np.concatenate([left, right], axis=0)
        for shift in range(len(contour)):
            if np.allclose(combined, np.roll(contour, -shift, axis=0)):
                break
        else:
            pytest.fail("real contour: (left ++ right) 不能通过 cyclic shift 还原")


# ── compute_width_profile 单元测试 ────────────────────────────────────────


class TestComputeWidthProfile:
    def test_empty_input(self):
        w, y = compute_width_profile(np.empty((0, 2)), np.empty((0, 2)))
        assert w.size == 0
        assert y.size == 0

    def test_insufficient_points(self):
        left = np.array([[0.0, 0.0], [1.0, 0.0]])
        right = np.array([[2.0, 0.0], [3.0, 0.0]])
        w, y = compute_width_profile(left, right)
        assert w.size == 0

    def test_non_overlapping_y_range(self):
        left = np.array([[-10.0, 0.0], [-5.0, 1.0], [0.0, 2.0]])
        right = np.array([[10.0, 5.0], [15.0, 6.0], [20.0, 7.0]])
        w, y = compute_width_profile(left, right)
        assert w.size == 0

    def test_symmetric_width(self):
        y_vals = np.linspace(0, 100, 50)
        left = np.column_stack([np.full_like(y_vals, -40), y_vals])
        right = np.column_stack([np.full_like(y_vals, 60), y_vals])
        w, y_cen = compute_width_profile(left, right)
        assert len(w) == 150
        assert np.allclose(w, 100.0, atol=1.0)
        assert y_cen[0] < y_cen[-1]

    def test_varying_width(self):
        y = np.linspace(-100, 100, 100)
        left = np.column_stack([-(30 + 0.1 * y), y])
        right = np.column_stack([30 - 0.1 * y, y])
        w, _ = compute_width_profile(left, right)
        assert np.all(w > 0)


# ── extract_split_contours 集成测试 ───────────────────────────────────────


class TestExtractSplitContours:
    def test_empty_vertices(self):
        left, right = extract_split_contours(np.empty((0, 3)))
        assert left.size == 0
        assert right.size == 0

    def test_output_ndim(self):
        pts = np.random.rand(50, 3) * 200 - 100
        left, right = extract_split_contours(pts)
        if left.size:
            assert left.ndim == 2
            assert left.shape[1] == 2
        if right.size:
            assert right.ndim == 2
            assert right.shape[1] == 2

    def test_few_vertices_still_produces_output(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]], dtype=float)
        left, right = extract_split_contours(pts)
        assert left.size > 0 or right.size > 0

    # ── 真实 mesh 端到端验证 ──

    @pytest.mark.skipif(not _HAS_REAL_MESH, reason="real mesh not available")
    def test_real_pipeline_nonempty(self, real_vertices: np.ndarray | None) -> None:
        """端到端管线：真实顶点 → 左右轮廓均非空且形状正确。"""
        left, right = extract_split_contours(real_vertices)
        assert len(left) > 10, f"left too short: {len(left)}"
        assert len(right) > 10, f"right too short: {len(right)}"
        assert left.shape[1] == 2
        assert right.shape[1] == 2

    @pytest.mark.skipif(not _HAS_REAL_MESH, reason="real mesh not available")
    def test_real_pipeline_x_separation(self, real_vertices: np.ndarray | None) -> None:
        """端到端管线：左 X 中位 < 右 X 中位。"""
        left, right = extract_split_contours(real_vertices)
        assert np.median(left[:, 0]) < np.median(right[:, 0])

    @pytest.mark.skipif(not _HAS_REAL_MESH, reason="real mesh not available")
    def test_real_pipeline_x_range_check(self, real_vertices: np.ndarray | None) -> None:
        """端到端管线：左 X 整体 ≤ center_x ≤ 右 X。"""
        left, right = extract_split_contours(real_vertices)
        l_max, r_min = left[:, 0].max(), right[:, 0].min()
        # 允许颈部附近微小重叠
        assert l_max - r_min < 60.0, f"X overlap too large: left max={l_max:.1f}, right min={r_min:.1f}"


# ── 可视化集成测试 ───────────────────────────────────────────────────


def test_missing_mesh_raises():
    """不存在的 project_id 应报错。"""
    with pytest.raises(FileNotFoundError):
        load_mesh_by_project("nonexistent_99999")

