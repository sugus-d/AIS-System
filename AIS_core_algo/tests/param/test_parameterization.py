"""Tests for parameterization package public API.

Covers the four exports: TEMPLATE_LANDMARKS, parse_landmarks_json,
find_landmark_vertices, harmonic_parameterize.
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from parameterization import (
    find_landmark_vertices,
    harmonic_parameterize,
    parse_landmarks_json,
    TEMPLATE_LANDMARKS,
)

# ---------------------------------------------------------------------------
# TEMPLATE_LANDMARKS
# ---------------------------------------------------------------------------


class TestTemplateLandmarks:
    """TEMPLATE_LANDMARKS is a dict[str, tuple[float, float]]."""

    def test_contains_all_expected_keys(self):
        expected_keys = {
            "neck_root_L",
            "neck_root_R",
            "shoulder_transition_L",
            "shoulder_transition_R",
            "scapular_peaks_L",
            "scapular_peaks_R",
            "axilla_L",
            "axilla_R",
            "waist_L",
            "waist_R",
            "waist_lower_L",
            "waist_lower_R",
            "neck_root_spine_point",
            "scapular_spine_point",
            "axilla_spine_point",
            "waist_spine_point",
            "waist_lower_spine_point",
            "thoracic_spine_point",
        }
        assert set(TEMPLATE_LANDMARKS.keys()) == expected_keys

    def test_lr_symmetry(self):
        """L/R pairs symmetric in U, equal in V."""
        lr_pairs = [
            ("neck_root_L", "neck_root_R"),
            ("shoulder_transition_L", "shoulder_transition_R"),
            ("scapular_peaks_L", "scapular_peaks_R"),
            ("axilla_L", "axilla_R"),
            ("waist_L", "waist_R"),
            ("waist_lower_L", "waist_lower_R"),
        ]
        for left_name, right_name in lr_pairs:
            left_u, left_v = TEMPLATE_LANDMARKS[left_name]
            right_u, right_v = TEMPLATE_LANDMARKS[right_name]
            assert left_u == -right_u, f"{left_name} U={left_u} != -({right_name} U={right_u})"
            assert left_v == right_v, f"{left_name} V={left_v} != {right_name} V={right_v}"

    def test_spine_on_midline(self):
        """Spine landmarks must have U=0."""
        for name in ["neck_root_spine_point", "scapular_spine_point", "axilla_spine_point", "waist_spine_point", "waist_lower_spine_point", "thoracic_spine_point"]:
            assert TEMPLATE_LANDMARKS[name][0] == 0.0, f"{name} not on midline"

    def test_uv_in_expected_ranges(self):
        """U in [-2.5, 2.5], V in [-4, 2] as documented."""
        for name, (u, v) in TEMPLATE_LANDMARKS.items():
            assert -2.5 <= u <= 2.5, f"{name} U={u} outside [-2.5, 2.5]"
            assert -4.0 <= v <= 2.0, f"{name} V={v} outside [-4, 2]"

    def test_known_values_correct(self):
        """Exact UV values match the documentation in template.py."""
        expected = {
            "neck_root_L": (-0.75, 2.0),
            "neck_root_R": (0.75, 2.0),
            "shoulder_transition_L": (-1.75, 1.75),
            "shoulder_transition_R": (1.75, 1.75),
            "scapular_peaks_L": (-1.25, 1.0),
            "scapular_peaks_R": (1.25, 1.0),
            "axilla_L": (-2.5, 0.0),
            "axilla_R": (2.5, 0.0),
            "waist_L": (-2.0, -3.0),
            "waist_R": (2.0, -3.0),
            "neck_root_spine_point": (0.0, 2.0),
            "scapular_spine_point": (0.0, 1.0),
            "axilla_spine_point": (0.0, 0.0),
            "waist_spine_point": (0.0, -3.0),
            "waist_lower_spine_point": (0.0, -4.0),
            "thoracic_spine_point": (0.0, -1.5),
            "waist_lower_L": (-2.3, -4.0),
            "waist_lower_R": (2.3, -4.0),
        }
        assert expected == TEMPLATE_LANDMARKS


# ---------------------------------------------------------------------------
# parse_landmarks_json
# ---------------------------------------------------------------------------


def _write_gt(tmp_path: Path, gt: dict) -> Path:
    """写入 ground_truth.json 并返回路径。"""
    p = tmp_path / "ground_truth.json"
    p.write_text(json.dumps(gt))
    return p


class TestParseLandmarksJSON:
    """parse_landmarks_json reads ground_truth.json from labeling platform / prelabel."""

    def test_raises_on_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_landmarks_json("/nonexistent_ais_test_file.json")

    def test_raises_on_empty_json(self, tmp_path: Path) -> None:
        gt_file = _write_gt(tmp_path, {"_features": {}})
        with pytest.raises(ValueError, match="No landmarks"):
            parse_landmarks_json(str(gt_file))

    def test_parses_single_bilateral(self, tmp_path: Path) -> None:
        gt = {
            "neck_root_L": [10.0, 20.0, 30.0],
            "neck_root_R": [40.0, 50.0, 60.0],
            "waist_L": [1.0, 2.0, 3.0],
        }
        gt_file = _write_gt(tmp_path, gt)
        result = parse_landmarks_json(str(gt_file))
        assert "neck_root_L" in result
        assert "neck_root_R" in result
        assert "waist_L" in result
        np.testing.assert_array_almost_equal(result["neck_root_L"], [10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(result["neck_root_R"], [40.0, 50.0, 60.0])

    def test_parses_spine_points(self, tmp_path: Path) -> None:
        spine = {key: [float(i), i + 1.0, i + 2.0] for i, key in enumerate(
            ["neck_root_spine_point", "scapular_spine_point", "axilla_spine_point",
             "waist_spine_point", "waist_lower_spine_point", "thoracic_spine_point"])}
        gt_file = _write_gt(tmp_path, spine)
        result = parse_landmarks_json(str(gt_file))
        assert "neck_root_spine_point" in result
        assert "thoracic_spine_point" in result
        np.testing.assert_array_almost_equal(result["axilla_spine_point"], [2.0, 3.0, 4.0])

    def test_ignores_features_metadata(self, tmp_path: Path) -> None:
        gt = {
            "_features": {"labeling_status": "labeled"},
            "axilla_L": [1.0, 2.0, 3.0],
            "axilla_R": [4.0, 5.0, 6.0],
        }
        gt_file = _write_gt(tmp_path, gt)
        result = parse_landmarks_json(str(gt_file))
        assert "axilla_L" in result
        assert not any(k.startswith("_") for k in result)

    def test_partial_sides_parsed_only(self, tmp_path: Path) -> None:
        """L-only / 缺失 side 的 bilateral 只解析存在的部分。"""
        gt = {"shoulder_transition_L": [1.0, 2.0, 3.0]}
        gt_file = _write_gt(tmp_path, gt)
        result = parse_landmarks_json(str(gt_file))
        assert "shoulder_transition_L" in result
        assert "shoulder_transition_R" not in result

    def test_no_landmarks_raises_error(self, tmp_path: Path) -> None:
        """JSON 无任何可解析 landmark 时抛 ValueError。"""
        gt_file = _write_gt(tmp_path, {"_features": {}, "unknown": [1, 2, 3]})
        with pytest.raises(ValueError, match="No landmarks"):
            parse_landmarks_json(str(gt_file))


# ---------------------------------------------------------------------------
# find_landmark_vertices
# ---------------------------------------------------------------------------


class TestFindLandmarkVertices:
    """find_landmark_vertices matches parsed CSVs to nearest mesh vertices."""

    def _quad_mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [1, 3, 2]])
        return mesh

    def test_matches_four_landmark_vertices(self):
        """With 4 landmarks matching template keys, find_landmark_vertices should
        return the nearest vertex indices and target UV coordinates."""
        mesh = self._quad_mesh()
        landmarks = {
            "neck_root_L": np.array([0.0, 0.0, 0.0]),  # nearest vertex 0
            "neck_root_R": np.array([1.0, 0.0, 0.0]),  # nearest vertex 1
            "waist_L": np.array([0.0, 1.0, 0.0]),  # nearest vertex 2
            "waist_R": np.array([1.0, 1.0, 0.0]),  # nearest vertex 3
        }
        template = {
            "neck_root_L": (0.15, 0.90),
            "neck_root_R": (0.85, 0.90),
            "waist_L": (0.15, 0.22),
            "waist_R": (0.85, 0.22),
        }
        k, y = find_landmark_vertices(mesh, landmarks, template)
        assert len(k) == 4
        assert len(y) == 4
        # Nearest vertex for [0,0,0] is index 0
        assert k[0] == 0
        np.testing.assert_array_almost_equal(y[0], [0.15, 0.90])

    def test_raises_if_fewer_than_4_matched(self):
        mesh = self._quad_mesh()
        landmarks = {"neck_root_L": np.array([0.0, 0.0, 0.0])}
        template = {"neck_root_L": (0.15, 0.90)}
        with pytest.raises(ValueError, match="(?i)only"):
            find_landmark_vertices(mesh, landmarks, template)

    def test_raises_on_no_matching_keys(self):
        mesh = self._quad_mesh()
        landmarks = {"unknown_key": np.array([0.0, 0.0, 0.0])}
        template = {"neck_root_L": (0.15, 0.90)}
        with pytest.raises(ValueError):
            find_landmark_vertices(mesh, landmarks, template)

    def test_extra_landmarks_beyond_template_ignored(self):
        """Landmark entries not in template should be silently ignored."""
        mesh = self._quad_mesh()
        landmarks = {
            "neck_root_L": np.array([0.0, 0.0, 0.0]),
            "neck_root_R": np.array([1.0, 0.0, 0.0]),
            "waist_L": np.array([0.0, 1.0, 0.0]),
            "waist_R": np.array([1.0, 1.0, 0.0]),
            "extra_key": np.array([0.5, 0.5, 0.0]),
        }
        template = {
            "neck_root_L": (0.15, 0.90),
            "neck_root_R": (0.85, 0.90),
            "waist_L": (0.15, 0.22),
            "waist_R": (0.85, 0.22),
        }
        k, y = find_landmark_vertices(mesh, landmarks, template)
        assert len(k) == 4
        assert len(y) == 4

    def test_duplicate_nearest_vertex(self):
        """Two landmarks nearest the same vertex are both recorded."""
        mesh = self._quad_mesh()
        landmarks = {
            "neck_root_L": np.array([0.0, 0.0, 0.0]),
            "neck_root_R": np.array([1.0, 0.0, 0.0]),
            "waist_L": np.array([0.01, 0.0, 0.0]),  # also nearest to vertex 0
            "waist_R": np.array([1.0, 1.0, 0.0]),
        }
        template = {
            "neck_root_L": (-1.0, 2.0),
            "neck_root_R": (1.0, 2.0),
            "waist_L": (-2.0, -3.0),
            "waist_R": (2.0, -3.0),
        }
        k, _ = find_landmark_vertices(mesh, landmarks, template)
        assert len(k) == 4
        assert k[2] == 0  # waist_L also maps to vertex 0


# ---------------------------------------------------------------------------
# Helpers for harmonic_parameterize tests
# ---------------------------------------------------------------------------


def _make_test_mesh() -> o3d.geometry.TriangleMesh:
    """构建 5 顶点、4 三角面的测试网格，用于调和参数化测试。"""
    """5-vertex, 4-triangle mesh for harmonic parameterization tests."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [0.0, 200.0, 0.0],
                [100.0, 200.0, 0.0],
                [50.0, 100.0, 10.0],
            ],
            dtype=np.float64,
        )
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        np.array(
            [
                [0, 1, 4],
                [0, 4, 2],
                [1, 3, 4],
                [2, 4, 3],
            ],
            dtype=np.int32,
        )
    )
    mesh.compute_vertex_normals()
    return mesh


def _make_test_landmark_k_y() -> tuple[np.ndarray, np.ndarray]:
    """绑定 4 个角顶点到目标 UV 坐标。"""
    """Pin the 4 corner vertex IDs to target UV coordinates."""
    k = np.array([0, 1, 2, 3], dtype=np.int64)
    y = np.array(
        [
            [-1.0, -3.0],
            [1.0, -3.0],
            [-1.0, 2.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )
    return k, y


# ---------------------------------------------------------------------------
# Additional helpers from test_harmonic.py consolidation
# ---------------------------------------------------------------------------


def _make_quad_mesh() -> o3d.geometry.TriangleMesh:
    """4-vertex, 2-triangle quad with landmarks at all corners."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [0.0, 1.0, 0.2],
            [1.0, 1.0, 0.3],
        ]
    )
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [1, 3, 2]])
    return mesh


def _make_pentagon_mesh() -> o3d.geometry.TriangleMesh:
    """5-vertex, 4-triangle mesh with one free interior vertex."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],  # interior vertex
        ]
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        [
            [0, 1, 4],
            [0, 4, 2],
            [1, 3, 4],
            [2, 4, 3],
        ]
    )
    return mesh


# ---------------------------------------------------------------------------
# harmonic_parameterize
# ---------------------------------------------------------------------------


class TestHarmonicParameterize:
    """harmonic_parameterize maps mesh to UV with landmark constraints."""

    def test_basic_parameterization(self):
        mesh = _make_test_mesh()
        k, y = _make_test_landmark_k_y()
        reg, uv = harmonic_parameterize(mesh, k, y)
        assert isinstance(reg, o3d.geometry.TriangleMesh)
        assert uv.shape == (5, 2)

    def test_preserves_topology(self):
        mesh = _make_test_mesh()
        k, y = _make_test_landmark_k_y()
        reg, _ = harmonic_parameterize(mesh, k, y)
        np.testing.assert_array_equal(np.asarray(mesh.triangles), np.asarray(reg.triangles))

    def test_landmark_constraints_satisfied(self):
        mesh = _make_test_mesh()
        k, y = _make_test_landmark_k_y()
        _, uv = harmonic_parameterize(mesh, k, y)
        np.testing.assert_allclose(uv[k], y, atol=1e-10)

    def test_z_preserved(self):
        mesh = _make_test_mesh()
        k, y = _make_test_landmark_k_y()
        reg, _ = harmonic_parameterize(mesh, k, y)
        np.testing.assert_allclose(
            np.asarray(reg.vertices)[:, 2],
            np.asarray(mesh.vertices)[:, 2],
            atol=1e-10,
        )

    def test_all_vertices_pinned(self):
        """All vertices are landmarks -- UV must match exactly."""
        mesh = _make_quad_mesh()
        landmark_ids = np.array([0, 1, 2, 3], dtype=np.int64)
        landmark_uv = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        _, uv = harmonic_parameterize(mesh, landmark_ids, landmark_uv)
        np.testing.assert_array_almost_equal(uv, landmark_uv)

    def test_interior_vertex_harmonic_mean(self):
        """Free interior vertex should be harmonic mean of corner landmarks."""
        mesh = _make_pentagon_mesh()
        landmark_ids = np.array([0, 1, 2, 3], dtype=np.int64)
        landmark_uv = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        _, uv = harmonic_parameterize(mesh, landmark_ids, landmark_uv)
        assert uv.shape == (5, 2)
        np.testing.assert_array_almost_equal(uv[4], [0.5, 0.5])

    def test_empty_mesh_raises(self):
        empty = o3d.geometry.TriangleMesh()
        with pytest.raises(ValueError, match="(?i)no triangles"):
            harmonic_parameterize(
                empty,
                np.array([0], dtype=np.int64),
                np.array([[0.0, 0.0]]),
            )

    @pytest.mark.parametrize("n_landmarks", [0, 1, 2, 3])
    def test_insufficient_landmarks_raises(self, n_landmarks: int) -> None:
        """Fewer than 4 landmarks should raise ValueError."""
        mesh = _make_quad_mesh()
        ids = np.arange(n_landmarks, dtype=np.int64)
        uv = np.zeros((n_landmarks, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="(?i)at least 4 landmark"):
            harmonic_parameterize(mesh, ids, uv)

    def test_landmark_out_of_range_raises(self):
        """Landmark vertex IDs that don't exist should raise IndexError."""
        mesh = _make_quad_mesh()
        landmark_ids = np.array([0, 1, 2, 999], dtype=np.int64)
        landmark_uv = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        with pytest.raises(IndexError):
            harmonic_parameterize(mesh, landmark_ids, landmark_uv)

    def test_negative_landmark_ids_raises(self):
        """Negative landmark vertex IDs should raise IndexError."""
        mesh = _make_quad_mesh()
        landmark_ids = np.array([0, 1, -1, 3], dtype=np.int64)
        landmark_uv = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        with pytest.raises(IndexError):
            harmonic_parameterize(mesh, landmark_ids, landmark_uv)

    def test_output_uv_dtype_and_shape(self):
        """Returned uv_coords must be an ndarray of shape (N, 2)."""
        mesh = _make_quad_mesh()
        landmark_ids = np.array([0, 1, 2, 3], dtype=np.int64)
        landmark_uv = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        _, uv = harmonic_parameterize(mesh, landmark_ids, landmark_uv)
        assert isinstance(uv, np.ndarray)
        assert uv.shape == (4, 2)

    def test_uv_bounded_by_convex_hull(self):
        """Free vertex UVs should stay within the convex hull of landmarks."""
        mesh = _make_pentagon_mesh()
        landmark_ids = np.array([0, 1, 2, 3], dtype=np.int64)
        landmark_uv = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        _, uv = harmonic_parameterize(mesh, landmark_ids, landmark_uv)
        assert uv.min() >= 0.0 - 1e-10
        assert uv.max() <= 1.0 + 1e-10

    def test_vertex_normals_valid(self):
        """Output mesh should have valid vertex normals."""
        mesh = _make_quad_mesh()
        landmark_ids = np.array([0, 1, 2, 3], dtype=np.int64)
        landmark_uv = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float64,
        )
        result_mesh, _ = harmonic_parameterize(mesh, landmark_ids, landmark_uv)
        normals = np.asarray(result_mesh.vertex_normals)
        assert normals.shape == (4, 3)
        assert not np.any(np.isnan(normals))
        assert np.all(np.linalg.norm(normals, axis=1) > 0)
