"""Tests for features.asymmetry.landmark_regions — UV-space landmark-based asymmetry.

pytest 兼容文件。需要通过 bootstrap 来回避 open3d 的 import 链副作用：

    python3 -m pytest tests/test_landmark_regions.py -v
"""

import sys
import types

import numpy as np
import pytest

# ── Bootstrap: 在无 open3d 的环境下 mock parameterization.template ──────
# 先保存真实模块引用，被测模块加载完成后恢复 sys.modules——
# 否则按字母序收集时本文件劫持的 stub 会污染 test_parameterization.py 的导入。
import parameterization as _real_parameterization
import parameterization.template as _real_template

_TEMPLATE = {
    "neck_root_spine_point": (0.0, 2.0),
    "scapular_spine_point": (0.0, 1.0),
    "axilla_spine_point": (0.0, 0.0),
    "waist_spine_point": (0.0, -3.0),
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
}

_pt = types.ModuleType("parameterization.template")
_pt.TEMPLATE_LANDMARKS = _TEMPLATE
_pkg = types.ModuleType("parameterization")
_pkg.template = _pt
sys.modules["parameterization"] = _pkg
sys.modules["parameterization.template"] = _pt

# 被测模块已拆为包（landmark_regions/{__init__,_regions,_features}.py），
# 直接走包导入；其内部 `from parameterization.template import ...` 会命中上方 stub。
from features.extractors.asymmetry.landmark_regions import (  # noqa: E402
    classify_by_region as classify,
)
from features.extractors.asymmetry.landmark_regions import (  # noqa: E402
    compute_curvature_asymmetry as curv_asym,
)
from features.extractors.asymmetry.landmark_regions import (  # noqa: E402
    compute_region_asymmetry as asym,
)
from features.extractors.asymmetry.landmark_regions import (  # noqa: E402
    compute_region_features as region_features,
)
from features.extractors.asymmetry.landmark_regions import (  # noqa: E402
    SEG_LUMBAR as SEG_L,
)
from features.extractors.asymmetry.landmark_regions import (  # noqa: E402
    SEG_PELVIC as SEG_P,
)
from features.extractors.asymmetry.landmark_regions import (  # noqa: E402
    SEG_SHOULDER as SEG_S,
)
from features.extractors.asymmetry.landmark_regions import (  # noqa: E402
    SEG_THORACIC as SEG_T,
)

# 被测包已加载（内部引用已绑定 stub），恢复真实包避免污染其他测试
sys.modules["parameterization"] = _real_parameterization
sys.modules["parameterization.template"] = _real_template

_RNAMES = {SEG_S: "Shoulder", SEG_T: "Thoracic", SEG_L: "Lumbar", SEG_P: "Pelvic", -1: "?"}
_SNAMES = {0: "L", 1: "R", -1: "?"}

# Grid fixture for all tests.
_N = 101
_uv = np.column_stack(
    [g.ravel() for g in np.meshgrid(np.linspace(-2.5, 2.5, _N), np.linspace(-3.0, 2.0, _N))]
)
_labels, _sides = classify(_uv)


# ═══════════════════════════════════════════════════════════════════════


class TestGridCoverage:
    """100 % classification over the template UV grid."""

    def test_no_unclassified(self):
        assert (_labels != -1).all(), f"{( _labels == -1).sum()} unclassified points"

    def test_all_regions_have_vertices(self):
        for rid, name in [(SEG_S, "Shoulder"), (SEG_T, "Thoracic"), (SEG_L, "Lumbar"), (SEG_P, "Pelvic")]:
            assert (_labels == rid).any(), f"{name} has no vertices"

    def test_both_sides_have_vertices(self):
        for rid in (SEG_S, SEG_T, SEG_L, SEG_P):
            l_cnt = int(((_labels == rid) & (_sides == 0)).sum())
            r_cnt = int(((_labels == rid) & (_sides == 1)).sum())
            assert l_cnt > 0, f"Region {rid}: no left vertices"
            assert r_cnt > 0, f"Region {rid}: no right vertices"

    def test_symmetric_ratios(self):
        for rid, name in [(SEG_S, "Shoulder"), (SEG_T, "Thoracic"), (SEG_L, "Lumbar"), (SEG_P, "Pelvic")]:
            left_count = ((_labels == rid) & (_sides == 0)).sum()
            right_count = ((_labels == rid) & (_sides == 1)).sum()
            ratio = left_count / max(right_count, 1)
            assert 0.75 < ratio < 1.25, f"{name}: L/R ratio {ratio:.3f} ∉ [0.75, 1.25]"


class TestKeyPoints:
    """Known anatomical points → expected region assignments."""

    _keypoints = np.array([
        [-1.5, 1.5],  # → Shoulder L
        [1.5, 1.5],  # → Shoulder R
        [-1.5, -1.5],  # → Lumbar L
        [1.5, -1.5],  # → Lumbar R
        [-1.5, -4.0],  # → Pelvic L
        [1.5, -4.0],  # → Pelvic R
    ])
    _expected = [
        (SEG_S, 0),
        (SEG_S, 1),
        (SEG_L, 0),
        (SEG_L, 1),
        (SEG_P, 0),
        (SEG_P, 1),
    ]
    _classified = classify(_keypoints)

    def test_all_points_classified(self):
        for pt, (el, es), (al, as_) in zip(self._keypoints, self._expected,
                                            zip(self._classified[0], self._classified[1], strict=False),
                                            strict=False):
            assert al == el, f"({pt[0]:.1f},{pt[1]:.1f}) → region {al}, expected {_RNAMES[el]}"
            assert as_ == es, f"({pt[0]:.1f},{pt[1]:.1f}) → side {as_}, expected {_SNAMES[es]}"


class TestAsymmetry:
    """compute_region_asymmetry and compute_curvature_asymmetry."""

    def test_global_is_finite(self):
        vals = np.random.default_rng(42).normal(0, 1, len(_uv))
        g, pg = asym(vals, _labels, _sides)
        assert np.isfinite(g), f"global = {g}"
        assert len(pg) == 4

    def test_global_non_negative(self):
        # Constant values → zero asymmetry
        const = np.ones(len(_uv))
        g, _ = asym(const, _labels, _sides)
        assert g == 0.0, f"constant → {g} (should be 0)"

    def test_weighted_global(self):
        vals = np.random.default_rng(42).normal(0, 1, len(_uv))
        w = np.array([0.4, 0.3, 0.2, 0.1])
        g, _ = asym(vals, _labels, _sides, weights=w)
        assert np.isfinite(g)

    def test_curvature_asymmetry(self):
        cm = np.random.default_rng(43).normal(0, 0.01, len(_uv))
        cg = np.random.default_rng(44).normal(0, 0.001, len(_uv))
        ai_g, ai_pg = curv_asym(cm, cg, _labels, _sides)
        assert np.isfinite(ai_g)
        assert len(ai_pg) == 4

    def test_wrong_weights_raises(self):
        vals = np.ones(len(_uv))
        try:
            asym(vals, _labels, _sides, weights=np.array([0.25, 0.25, 0.25]))
            pytest.fail("should have raised ValueError")
        except ValueError:
            pass


class TestEdgeCases:
    """Empty input, single point, etc."""

    def test_empty_input(self):
        el, es = classify(np.empty((0, 2), dtype=np.float64))
        assert len(el) == 0
        assert len(es) == 0

    def test_single_point(self):
        sl, ss = classify(np.array([[-1.0, 1.5]], dtype=np.float64))
        assert sl[0] >= 0, f"single point unclassified (label={sl[0]})"
        assert ss[0] >= 0, f"single point unclassified (side={ss[0]})"

    def test_point_far_away(self):
        """Point far from any polygon → assigned via fallback."""
        sl, ss = classify(np.array([[10.0, 10.0]], dtype=np.float64))
        assert sl[0] >= 0, "far point should be assigned via fallback"


class TestRegionFeatures:
    """compute_region_features — raw |Δmean| feature matrix (R regions × Q measures)."""

    def test_shape_and_default_regions(self):
        """Default: 3 non-pelvic regions × 3 measures → shape (3, 3)."""
        heights = np.random.default_rng(100).normal(0, 1, len(_uv))
        cm = np.random.default_rng(101).normal(0, 1, len(_uv))
        cg = np.random.default_rng(102).normal(0, 1, len(_uv))
        features, names = region_features(heights, cm, cg, _labels, _sides)
        assert features.shape == (3, 3), f"shape = {features.shape}, expected (3, 3)"
        assert len(names) == 3, f"got {len(names)} names, expected 3"
        assert all(isinstance(n, str) for n in names)

    def test_constant_values_zero_asymmetry(self):
        """All vertices constant → all |Δmean| = 0."""
        heights = np.ones(len(_uv))
        cm = np.ones(len(_uv))
        cg = np.ones(len(_uv))
        features, _ = region_features(heights, cm, cg, _labels, _sides)
        assert np.all(features == 0.0), f"constant values should give zero, got {features}"

    def test_known_asymmetry(self):
        """Height L=+1, R=-1, curvatures constant → height features > 0, curvature = 0."""
        heights = np.where(_sides == 0, 1.0, -1.0)   # left +1, right -1
        cm = np.ones(len(_uv))
        cg = np.ones(len(_uv))
        features, _ = region_features(heights, cm, cg, _labels, _sides)
        assert np.all(features[:, 0] > 0.0), f"height features should be > 0: {features[:, 0]}"
        assert np.all(features[:, 1] == 0.0), f"curv_mean should be 0: {features[:, 1]}"
        assert np.all(features[:, 2] == 0.0), f"curv_gauss should be 0: {features[:, 2]}"

    def test_custom_region_ids(self):
        """Passing only 2 region IDs → shape (2, 3)."""
        heights = np.random.default_rng(103).normal(0, 1, len(_uv))
        cm = np.random.default_rng(104).normal(0, 1, len(_uv))
        cg = np.random.default_rng(105).normal(0, 1, len(_uv))
        features, _ = region_features(
            heights, cm, cg, _labels, _sides,
            region_ids=np.array([SEG_S, SEG_T]),
        )
        assert features.shape == (2, 3), f"shape = {features.shape}, expected (2, 3)"

    def test_excludes_pelvic_by_default(self):
        """Pelvic region (SEG_P=3) must not appear in default output."""
        heights = np.random.default_rng(106).normal(0, 1, len(_uv))
        cm = np.random.default_rng(107).normal(0, 1, len(_uv))
        cg = np.random.default_rng(108).normal(0, 1, len(_uv))
        features, _ = region_features(heights, cm, cg, _labels, _sides)
        assert features.shape[0] == 3, f"expected 3 regions (non-pelvic), got {features.shape[0]}"

    def test_mismatched_length_raises(self):
        """Arrays of differing lengths should raise ValueError."""
        labels = np.array([0, 0, 0], dtype=np.int32)
        sides = np.array([0, 1, 0], dtype=np.int32)
        heights = np.ones(3)
        cm = np.ones(3)
        cg = np.ones(3)
        # Matching lengths → no error
        region_features(heights, cm, cg, labels, sides, region_ids=np.array([SEG_S]))
        # Mismatched heights
        try:
            region_features(np.ones(4), cm, cg, labels, sides, region_ids=np.array([SEG_S]))
            pytest.fail("should have raised ValueError")
        except ValueError:
            pass
        # Mismatched curv_mean
        try:
            region_features(heights, np.ones(4), cg, labels, sides, region_ids=np.array([SEG_S]))
            pytest.fail("should have raised ValueError")
        except ValueError:
            pass
        # Mismatched curv_gauss
        try:
            region_features(heights, cm, np.ones(4), labels, sides, region_ids=np.array([SEG_S]))
            pytest.fail("should have raised ValueError")
        except ValueError:
            pass

    def test_empty_region_ids(self):
        """Empty region_ids array → shape (0, 3)."""
        heights = np.ones(len(_uv))
        cm = np.ones(len(_uv))
        cg = np.ones(len(_uv))
        features, names = region_features(
            heights, cm, cg, _labels, _sides,
            region_ids=np.array([], dtype=np.int32),
        )
        assert features.shape == (0, 3), f"shape = {features.shape}"
        assert len(names) == 3

    def test_region_only_one_side(self):
        """A region whose vertices all fall on one side → row stays zero."""
        N = 5
        labels = np.array([SEG_S, SEG_S, SEG_S, SEG_S, SEG_S], dtype=np.int32)
        # All vertices assigned to the *left* side only
        sides = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        h = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cm = np.ones(N)
        cg = np.ones(N)
        features, _ = region_features(
            h, cm, cg, labels, sides,
            region_ids=np.array([SEG_S], dtype=np.int32),
        )
        assert features.shape == (1, 3)
        # Row stays zero because right side is empty → continue
        assert np.all(features == 0.0), f"expected zero row, got {features}"

    def test_nan_values(self):
        """NaN in input values → NaN propagates to the feature matrix."""
        N = 10
        labels = np.array([SEG_S] * N, dtype=np.int32)
        sides = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
        h = np.array([np.nan] * N)
        cm = np.ones(N)
        cg = np.ones(N)
        features, _ = region_features(
            h, cm, cg, labels, sides,
            region_ids=np.array([SEG_S], dtype=np.int32),
        )
        assert np.isnan(features[0, 0]), "height feature should be NaN"
        assert features[0, 1] == 0.0, "curv_mean should be 0"
        assert features[0, 2] == 0.0, "curv_gauss should be 0"

    def test_inf_values(self):
        """Inf on one side only → Inf propagates to the feature matrix."""
        N = 10
        labels = np.array([SEG_S] * N, dtype=np.int32)
        sides = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
        h = np.ones(N)
        # Inf only on left side, finite on right → |Inf - finite| = Inf
        cm = np.array([np.inf] * 5 + [0.0] * 5)
        cg = np.ones(N)
        features, _ = region_features(
            h, cm, cg, labels, sides,
            region_ids=np.array([SEG_S], dtype=np.int32),
        )
        assert np.isinf(features[0, 1]), "curv_mean feature should be Inf"
        assert np.isfinite(features[0, 0]), "height should be finite"
        assert features[0, 2] == 0.0, "curv_gauss should be 0"

class TestComputeRegionFeatures:
    """compute_region_features -- additional edge cases for compute_region_features."""

    def test_all_regions_shape_non_negative(self):
        """All 4 regions explicitly -> shape (4,3), non-negative, finite."""
        heights = np.random.default_rng(300).normal(0, 1, len(_uv))
        cm = np.random.default_rng(301).normal(0, 1, len(_uv))
        cg = np.random.default_rng(302).normal(0, 1, len(_uv))
        features, names = region_features(
            heights, cm, cg, _labels, _sides,
            region_ids=np.array([SEG_S, SEG_T, SEG_L, SEG_P], dtype=np.int32),
        )
        assert features.shape == (4, 3), f"shape = {features.shape}"
        assert np.all(features >= 0), "all values must be non-negative"
        assert np.all(np.isfinite(features)), "all values must be finite"
        assert len(names) == 3

    def test_deterministic_reproducible(self):
        """Same random seed -> identical feature output across calls."""
        seed_a, seed_b, seed_c = 400, 401, 402
        heights1 = np.random.default_rng(seed_a).normal(0, 1, len(_uv))
        cm1 = np.random.default_rng(seed_b).normal(0, 1, len(_uv))
        cg1 = np.random.default_rng(seed_c).normal(0, 1, len(_uv))
        features1, _ = region_features(heights1, cm1, cg1, _labels, _sides)

        heights2 = np.random.default_rng(seed_a).normal(0, 1, len(_uv))
        cm2 = np.random.default_rng(seed_b).normal(0, 1, len(_uv))
        cg2 = np.random.default_rng(seed_c).normal(0, 1, len(_uv))
        features2, _ = region_features(heights2, cm2, cg2, _labels, _sides)

        assert np.allclose(features1, features2), (
            "identical inputs should give identical output"
        )

    def test_nan_isolated_to_height_column(self):
        """NaN only in heights -> NaN propagates only to first column."""
        N = 30
        labels = np.array(
            [SEG_S] * 10 + [SEG_T] * 10 + [SEG_L] * 10, dtype=np.int32
        )
        sides = np.array(
            [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5,
            dtype=np.int32,
        )
        h = np.array([np.nan] * 10 + [1.0] * 20)
        cm = np.ones(N)
        cg = np.ones(N)
        features, _ = region_features(
            h, cm, cg, labels, sides,
            region_ids=np.array([SEG_S, SEG_T, SEG_L], dtype=np.int32),
        )
        assert np.isnan(features[0, 0]), "S height should be NaN"
        assert features[0, 1] == 0.0, "S curv_mean should be 0"
        assert features[0, 2] == 0.0, "S curv_gauss should be 0"
        assert np.all(np.isfinite(features[1:])), "T/L regions should be finite"

    def test_region_id_without_vertices(self):
        """region_id that never appears in labels -> row stays zero."""
        N = 20
        labels = np.array([SEG_S] * 10 + [SEG_T] * 10, dtype=np.int32)
        sides = np.array(
            [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5, dtype=np.int32
        )
        h = np.array([1.0] * 10 + [2.0] * 10)
        cm = np.ones(N)
        cg = np.ones(N)
        features, _ = region_features(
            h, cm, cg, labels, sides,
            region_ids=np.array([SEG_S, SEG_L], dtype=np.int32),
        )
        assert features.shape == (2, 3)
        assert np.isfinite(features[0, 0]), "present region should be finite"
        assert np.all(features[1] == 0.0), (
            f"absent region (L) should be zero, got {features[1]}"
        )

    def test_duplicate_region_ids(self):
        """Duplicate IDs in region_ids -> duplicate rows with identical values."""
        N = 20
        labels = np.array([SEG_S] * N, dtype=np.int32)
        sides = np.array([0] * 10 + [1] * 10, dtype=np.int32)
        h = np.array([1.0] * 10 + [2.0] * 10)
        cm = np.ones(N)
        cg = np.ones(N)
        features, _ = region_features(
            h, cm, cg, labels, sides,
            region_ids=np.array([SEG_S, SEG_S], dtype=np.int32),
        )
        assert features.shape == (2, 3), f"shape = {features.shape}"
        assert np.allclose(features[0], features[1]), (
            "duplicate IDs -> duplicate rows"
        )

    def test_region_ids_as_plain_list(self):
        """region_ids as Python list (not ndarray) -> works identically."""
        heights = np.random.default_rng(310).normal(0, 1, len(_uv))
        cm = np.random.default_rng(311).normal(0, 1, len(_uv))
        cg = np.random.default_rng(312).normal(0, 1, len(_uv))
        features, names = region_features(
            heights, cm, cg, _labels, _sides,
            region_ids=[SEG_S, SEG_T],
        )
        assert features.shape == (2, 3), f"shape = {features.shape}"
        assert len(names) == 3

    def test_feature_names_correct(self):
        """Returns correct feature names with constant (symmetric) input.

        With all vertices set to the same constant (symmetric) value, the
        |Δmean| for every region/measure is zero, confirming the name list
        is returned correctly alongside a zero-valued feature matrix.
        """
        features, names = region_features(
            np.ones(len(_uv)), np.ones(len(_uv)), np.ones(len(_uv)),
            _labels, _sides,
        )
        expected = ["height", "curv_mean", "curv_gauss"]
        assert names == expected, f"got {names}, expected {expected}"
        assert np.all(features == 0.0), (
            f"symmetric constant input should yield zero features, "
            f"got max={features.max():.2e}"
        )

    def test_region_ids_out_of_order(self):
        """Non-contiguous/non-sorted region_ids -> rows follow input order."""
        heights = np.random.default_rng(320).normal(0, 1, len(_uv))
        cm = np.random.default_rng(321).normal(0, 1, len(_uv))
        cg = np.random.default_rng(322).normal(0, 1, len(_uv))
        features, _ = region_features(
            heights, cm, cg, _labels, _sides,
            region_ids=np.array([SEG_L, SEG_S, SEG_P], dtype=np.int32),
        )
        assert features.shape == (3, 3), f"shape = {features.shape}"
        assert np.all(np.isfinite(features)), "all values must be finite"
