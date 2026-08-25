"""landmarks.complete — 扁平 landmarks 缺失补全 + Procrustes 相似变换测试。"""

from __future__ import annotations

import types

import numpy as np
import pytest

from landmarks.complete import complete_landmarks_flat
from parameterization.procrustes import compute_procrustes


def _make_flat() -> dict:
    """构造 18 键扁平 landmarks（6 对 bilateral + 6 个 spine）。"""
    flat = {}
    for name in ["neck_root", "shoulder_transition", "scapular_peaks", "axilla", "waist", "waist_lower"]:
        flat[f"{name}_L"] = [1.0, 2.0, 3.0]
        flat[f"{name}_R"] = [4.0, 5.0, 6.0]
    for idx, key in enumerate(
        [
            "neck_root_spine_point",
            "scapular_spine_point",
            "axilla_spine_point",
            "waist_spine_point",
            "waist_lower_spine_point",
            "thoracic_spine_point",
        ]
    ):
        flat[key] = [float(idx), 0.0, 0.0]
    return flat


class TestProcrustes:
    def test_recovers_known_transform(self):
        """scale·R·src + t 精确恢复 scale/rotation/translation（3D）。"""
        rng = np.random.default_rng(0)
        src = rng.normal(size=(8, 3))
        angle = 0.7
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        scale = 2.0
        trans = np.array([1.0, 2.0, 3.0])
        dst = scale * (src @ rot.T) + trans

        recovered_scale, recovered_rot, recovered_trans = compute_procrustes(src, dst)
        assert recovered_scale == pytest.approx(scale, abs=1e-9)
        assert np.allclose(recovered_rot, rot, atol=1e-9)
        assert np.allclose(recovered_trans, trans, atol=1e-9)

    def test_2d_and_3d_give_identical_reconstruction(self):
        """同一实现覆盖 2D/3D（此前 2D/3D 各一份同源算法）。"""
        rng = np.random.default_rng(1)
        for dim in (2, 3):
            src = rng.normal(size=(6, dim))
            scale, rot, trans = 1.5, np.eye(dim), rng.normal(size=dim)
            dst = scale * (src @ rot.T) + trans
            s_r, r_r, t_r = compute_procrustes(src, dst)
            assert s_r == pytest.approx(scale, abs=1e-9)
            assert np.allclose(r_r, rot, atol=1e-9)
            assert np.allclose(t_r, trans, atol=1e-9)

    def test_degenerate_few_points_returns_finite(self):
        """少于 3 点仍返回有限值（不崩溃；补全调用方会因点数不足保持原样）。"""
        src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        scale, rot, trans = compute_procrustes(src, src)
        assert np.isfinite(scale) and np.isfinite(rot).all() and np.isfinite(trans).all()

    def test_near_collinear_keeps_norm_ratio_scale(self):
        """近共线数据保持范数比缩放口径（Umeyama 奇异值公式在此分歧 ~1.4，勿回退）。

        参数化 landmark 近似共线，此口径直接决定 UV → 特征 → cobb（曾致 1° 回归）。
        """
        src = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, 0.2], [3.0, 0.3]])
        tgt = src * 2.0 + 1.0  # 纯缩放 2× + 平移 1
        scale, rot, trans = compute_procrustes(src, tgt)
        assert scale == pytest.approx(2.0, rel=1e-6)
        assert np.allclose(src * scale @ rot.T + trans, tgt, atol=1e-6)


class TestCompleteLandmarks:
    def test_complete_noop_when_full(self):
        """完整 18 键直接返回（不触碰 mesh）。"""
        flat = _make_flat()
        result = complete_landmarks_flat(flat, types.SimpleNamespace())
        assert result == flat

    def test_complete_fills_missing_to_nearest_vertex(self):
        """缺失键映射回 mesh 最近顶点；mesh 为空心网格时保持原样。"""
        flat = _make_flat()
        missing_flat = {k: v for k, v in flat.items() if k != "waist_lower_L"}
        mesh = types.SimpleNamespace(vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        result = complete_landmarks_flat(missing_flat, mesh)
        # 缺失键被补回，且坐标是某个现有顶点
        assert "waist_lower_L" in result
        assert result["waist_lower_L"] in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    def test_complete_too_few_known_keeps_unchanged(self):
        """已知点 < 3 时保持原样（调用方会因不完整报错）。"""
        flat = {"neck_root_L": [1.0, 2.0, 3.0], "neck_root_R": [4.0, 5.0, 6.0]}
        result = complete_landmarks_flat(flat, types.SimpleNamespace(vertices=np.zeros((3, 3))))
        assert result == flat
