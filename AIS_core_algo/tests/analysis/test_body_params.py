"""analysis.body_params — 9 个体征参数（论文表 2）计算测试。

覆盖纯几何计算 compute_cosmetic（对称/倾斜/长度/比值）与
gt_to_csv_row 的扁平 18 键 → CSV 行风格转换。
"""

from __future__ import annotations

import math

import pytest

from analysis.body_params import compute_cosmetic, gt_to_csv_row

# 对称身姿的 landmark 集（左右镜像、无倾斜），便于断言零值
_SYMMETRIC_GT = {
    "shoulder_transition_L": [-20.0, 30.0, -500.0],
    "shoulder_transition_R": [20.0, 30.0, -500.0],
    "scapular_peaks_L": [-10.0, -20.0, -480.0],
    "scapular_peaks_R": [10.0, -20.0, -480.0],
    "waist_lower_L": [-30.0, -320.0, -580.0],
    "waist_lower_R": [30.0, -320.0, -580.0],
    "neck_root_spine_point": [0.0, 40.0, -490.0],
    "scapular_spine_point": [0.0, -10.0, -470.0],
    "axilla_spine_point": [0.0, -60.0, -520.0],
    "waist_spine_point": [0.0, -150.0, -550.0],
    "waist_lower_spine_point": [0.0, -320.0, -580.0],
    "thoracic_spine_point": [0.0, -100.0, -540.0],
}


def _row() -> dict:
    """对称 gt → CSV 行风格 dict。"""
    return gt_to_csv_row(dict(_SYMMETRIC_GT), "S001")


class TestGtToCsvRow:
    def test_converts_flat_keys_to_csv_style(self):
        row = _row()
        assert row["subject_id"] == "S001"
        # 扁平键 → "键(x,y,z)" 字符串
        assert row["shoulder_transition_L(x,y,z)"] == "(-20.0,30.0,-500.0)"
        assert len(row) == 1 + len([k for k in _SYMMETRIC_GT])  # subject_id + 12 键


class TestComputeCosmetic:
    def test_symmetric_pose_zero_values(self):
        """对称身姿：IB/倾斜/不对称指数应为 0，Sh.W 为左右距。"""
        result = compute_cosmetic(_row())
        assert set(result) == {
            "Sh.IB", "Sh.A", "Sca.IB", "Sca.A", "ASIS.A", "Trunk.L", "Sh.W", "Sh.AI", "Pe.AI",
        }
        assert result["Sh.IB"] == pytest.approx(0.0)
        assert result["Sh.A"] == pytest.approx(0.0)
        assert result["Sca.IB"] == pytest.approx(0.0)
        assert result["ASIS.A"] == pytest.approx(0.0)
        assert result["Sh.AI"] == pytest.approx(1.0)  # 左右对称 → dL/dR = 1
        assert result["Pe.AI"] == pytest.approx(1.0)
        assert result["Sh.W"] == pytest.approx(40.0)  # |20 − (−20)|

    def test_asymmetric_pose_indices(self):
        """左肩抬高 + 右腰外移 → 对应不对称参数非零。"""
        gt = dict(_SYMMETRIC_GT)
        gt["shoulder_transition_L"] = [-20.0, 40.0, -500.0]  # 左肩抬高 10mm
        gt["waist_lower_R"] = [50.0, -320.0, -580.0]  # 右腰外移
        result = compute_cosmetic(gt_to_csv_row(gt, "S001"))
        assert result["Sh.IB"] == pytest.approx(-10.0)  # R − L = 30 − 40
        assert result["Sh.A"] != pytest.approx(0.0)
        assert result["Pe.AI"] != pytest.approx(1.0)

    def test_missing_landmarks_default_zero(self):
        """缺 landmark 时按 (0,0,0) 处理不抛错。"""
        result = compute_cosmetic({})
        assert result["Sh.W"] == pytest.approx(0.0)
        assert result["Sh.IB"] == pytest.approx(0.0)

    def test_trunk_length(self):
        """Trunk.L = 肩中点与腰下中点的欧氏距离。"""
        result = compute_cosmetic(_row())
        # 肩中点 (0,30)、腰下中点 (0,−320) → 距离 350
        assert result["Trunk.L"] == pytest.approx(350.0)

    def test_scapula_angle(self):
        """Sca.A = 左右肩胛连线倾角（atan2 口径）。"""
        result = compute_cosmetic(_row())
        # 对称：无倾角
        assert result["Sca.A"] == pytest.approx(0.0)

    def test_shoulder_angle_sloped(self):
        """右肩抬高 → 肩线倾角为正（atan2(dy, dx)）。"""
        gt = dict(_SYMMETRIC_GT)
        gt["shoulder_transition_R"] = [20.0, 50.0, -500.0]  # 右肩抬高 20mm
        result = compute_cosmetic(gt_to_csv_row(gt, "S001"))
        assert result["Sh.A"] == pytest.approx(math.degrees(math.atan2(20.0, 40.0)))
