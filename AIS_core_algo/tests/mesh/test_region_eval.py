"""Tests for mesh.roi.region_eval -- split lines, triangle classification, deltas."""

import json
from pathlib import Path

import numpy as np
import pytest

from mesh.roi.region_eval import (
    classify_triangles,
    compute_landmark_splits,
    compute_region_deltri,
    compute_thresholds,
    load_thresholds,
    region_report_text,
)


class TestComputeLandmarkSplits:
    """compute_landmark_splits -- basic vertex extents."""

    def test_basic(self):
        v = np.array([[0.0, 0.0, 0.0], [10.0, 100.0, 5.0]])
        splits = compute_landmark_splits(None, v)
        # Y_min=0, Y_max=100, span=100
        # Y_neck = 100 - 100 * 0.20 = 80
        assert splits["Y_neck"] == pytest.approx(80.0)
        # Y_waist = 0 + 100 * 0.15 = 15
        assert splits["Y_waist"] == pytest.approx(15.0)
        # X_center = (0+10)/2 = 5, span = 10
        # X_left = 5 - 10 * 0.35 = 1.5
        assert splits["X_left"] == pytest.approx(1.5)
        # X_right = 5 + 10 * 0.35 = 8.5
        assert splits["X_right"] == pytest.approx(8.5)

    def test_fallback_different_span(self):
        v = np.array([[-5.0, 10.0, 0.0], [5.0, 110.0, 0.0]])
        splits = compute_landmark_splits(None, v)
        # Y_min=10, Y_max=110, span=100, fallback ratios
        assert splits["Y_neck"] == pytest.approx(90.0)    # 110 - 100*0.20
        assert splits["Y_waist"] == pytest.approx(25.0)   # 10 + 100*0.15
        assert splits["X_left"] == pytest.approx(-3.5)    # 0 - 10*0.35
        assert splits["X_right"] == pytest.approx(3.5)    # 0 + 10*0.35

    def test_single_vertex(self):
        v = np.array([[2.0, 50.0, 0.0]])
        splits = compute_landmark_splits(None, v)
        assert splits["Y_neck"] == pytest.approx(50.0)
        assert splits["Y_waist"] == pytest.approx(50.0)
        assert splits["X_left"] == pytest.approx(2.0)
        assert splits["X_right"] == pytest.approx(2.0)


class TestClassifyTriangles:
    """classify_triangles -- region assignment for triangle centers."""

    _splits = {"Y_neck": 80.0, "Y_waist": 15.0, "X_left": 1.5, "X_right": 8.5}

    def test_empty(self):
        centers = np.empty((0, 3))
        labels = classify_triangles(centers, self._splits)
        assert len(labels) == 0

    def test_all_regions(self):
        """Construct one point per region and verify label.

        Each row maps to one of the 5 region labels:
        neck, core, hem, side_L, side_R (side rows duplicated to show
        both X boundary cases).
        """
        points = np.array(
            [
                [5.0, 90.0, 0.0],
                [5.0, 50.0, 0.0],
                [5.0, 10.0, 0.0],
                [0.0, 50.0, 0.0],
                [0.0, 15.0, 0.0],
                [9.0, 50.0, 0.0],
                [9.0, 15.0, 0.0],
            ],
            dtype=np.float64,
        )
        expected = np.array([0, 1, 2, 3, 3, 4, 4], dtype=np.int32)
        labels = classify_triangles(points, self._splits)
        assert np.array_equal(labels, expected), f"got {labels}, expected {expected}"

    def test_unassigned_outside_all(self):
        """No point goes unassigned -- every coordinate matches a region."""
        # With the exhaustive conditions, no point can remain at -1.
        points = np.array(
            [
                [100.0, -100.0, 0.0],  # hem (Y < Y_waist)
                [9.0, 80.0, 0.0],  # side_R (Y = Y_neck, not > 80)
            ],
        )
        labels = classify_triangles(points, self._splits)
        assert np.all(labels >= 0), f"got labels {labels}"

    def test_boundary_on_split_line(self):
        """Point exactly on Y_waist boundary: Y=15 not >15, X<1.5 -> side_L."""
        point = np.array([[0.0, 15.0, 0.0]])
        labels = classify_triangles(point, self._splits)
        _SIDE_L = 3
        assert labels[0] == _SIDE_L, f"expected side_L ({_SIDE_L}), got {labels[0]}"


class TestComputeRegionDeltri:
    """compute_region_deltri -- triangle count comparison."""

    def test_identical_meshes(self):
        """Identical algo and GT -> all deltas = 0."""
        v = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [5.0, 100.0, 0.0],
                [5.0, 10.0, 0.0],
            ],
            dtype=np.float64,
        )
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        results = compute_region_deltri(v, triangles, v, triangles)
        assert len(results) == 5
        for entry in results:
            assert entry["algo"] == entry["gt"]
            assert entry["delta"] == 0
            assert entry["delta_pct"] == pytest.approx(0.0)

    def test_different_triangle_count(self):
        """Algo has one fewer triangle in neck region."""
        v = np.array(
            [
                [0.0, 85.0, 0.0],
                [10.0, 85.0, 0.0],
                [5.0, 95.0, 0.0],
                [2.0, 20.0, 0.0],
                [8.0, 20.0, 0.0],
                [5.0, 30.0, 0.0],
                [2.0, 5.0, 0.0],
                [8.0, 5.0, 0.0],
                [5.0, 10.0, 0.0],
            ],
            dtype=np.float64,
        )
        # algo: 2 triangles (1 neck, 1 core) -- no hem
        algo_t = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        # gt: 3 triangles (1 neck, 1 core, 1 hem)
        gt_t = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)
        results = compute_region_deltri(v, algo_t, v, gt_t)
        result_map = {r["region"]: r for r in results}
        # neck: both have 1 -> delta=0
        neck = result_map["neck"]
        assert neck["algo"] == 1
        assert neck["gt"] == 1
        # hem: algo=0, gt=1 -> delta=-1 (algo has 1 fewer)
        hem = result_map["hem"]
        assert hem["algo"] == 0
        assert hem["gt"] == 1
        assert hem["delta"] == -1

    def test_empty_triangles(self):
        """No triangles -> zero counts everywhere."""
        v = np.array([[0.0, 0.0, 0.0], [10.0, 100.0, 0.0]], dtype=np.float64)
        triangles = np.empty((0, 3), dtype=np.int32)
        results = compute_region_deltri(v, triangles, v, triangles)
        for entry in results:
            assert entry["algo"] == 0
            assert entry["gt"] == 0


class TestRegionReportText:
    """region_report_text -- formatted table output."""

    def test_basic_format(self):
        results = [
            {"region": "neck", "algo": 5, "gt": 4, "delta": 1, "delta_pct": 25.0},
            {"region": "core", "algo": 10, "gt": 10, "delta": 0, "delta_pct": 0.0},
        ]
        text = region_report_text(results, subject="test_subj")
        assert "Region delta report for test_subj" in text
        assert "neck" in text
        assert "25.0%" in text
        assert "0.0%" in text

    def test_no_subject(self):
        results = [
            {"region": "neck", "algo": 1, "gt": 1, "delta": 0, "delta_pct": 0.0},
        ]
        text = region_report_text(results)
        assert "Region delta report" in text
        assert "for" not in text.splitlines()[0]


class TestComputeThresholds:
    """compute_thresholds -- percentile computation and JSON save."""

    def test_single_subject(self, tmp_path: Path):
        all_results = [
            [
                {"region": "neck", "algo": 10, "gt": 10, "delta": 0, "delta_pct": 0.0},
                {"region": "core", "algo": 20, "gt": 18, "delta": 2, "delta_pct": 11.1},
            ],
        ]
        out = tmp_path / "thresholds.json"
        thresholds = compute_thresholds(all_results, str(out))
        assert "neck" in thresholds
        assert "core" in thresholds
        assert thresholds["neck"]["p50"] == 0.0
        assert out.exists()

    def test_multiple_subjects(self, tmp_path: Path):
        results_1 = [
            {"region": "neck", "algo": 10, "gt": 10, "delta": 0, "delta_pct": 0.0},
        ]
        results_2 = [
            {"region": "neck", "algo": 10, "gt": 8, "delta": 2, "delta_pct": 25.0},
        ]
        results_3 = [
            {"region": "neck", "algo": 10, "gt": 5, "delta": 5, "delta_pct": 100.0},
        ]
        all_results = [results_1, results_2, results_3]
        out = tmp_path / "thresh.json"
        thresholds = compute_thresholds(all_results, str(out))
        neck = thresholds["neck"]
        _P50_MID = 25.0
        assert neck["p50"] == _P50_MID
        # np.percentile with default linear interpolation: p90=85.0, p95=92.5
        _P90 = 85.0
        assert neck["p90"] == _P90
        _P95 = 92.5
        assert neck["p95"] == _P95

    def test_json_content(self, tmp_path: Path):
        all_results = [
            [
                {"region": "hem", "algo": 5, "gt": 5, "delta": 0, "delta_pct": 0.0},
            ],
        ]
        out = tmp_path / "region.json"
        compute_thresholds(all_results, str(out))
        loaded = json.loads(out.read_text())
        assert "hem" in loaded
        assert loaded["hem"]["p50"] == 0.0


class TestLoadThresholds:
    """load_thresholds -- load from JSON or return None."""

    def test_load_existing(self, tmp_path: Path):
        expected = {"neck": {"p50": 0.0, "p90": 5.0, "p95": 10.0}}
        fp = tmp_path / "thresh.json"
        fp.write_text(json.dumps(expected))
        loaded = load_thresholds(str(fp))
        assert loaded == expected

    def test_missing_returns_none(self, tmp_path: Path):
        fp = tmp_path / "nonexistent.json"
        loaded = load_thresholds(str(fp))
        assert loaded is None
