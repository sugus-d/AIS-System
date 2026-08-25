"""M3 Landmark 定位黄金测试 — 固定 ROI 网格端到端。

输入：tests/data/numerics/mesh/roi_S0006.ply（M2 输出的 ROI 网格，固化以解耦 ROI 管线）。
"""

from __future__ import annotations

import open3d as o3d
import pytest

from tests.numerics.conftest import DATA_DIR, walk_golden

ROI_PLY = DATA_DIR / "mesh" / "roi_S0006.ply"

GOLDEN = {
    "axilla_L": ("(3,)", "-711.3436889648", "1f66f326d5bf33e692c84d1602c1fc56"),
    "axilla_R": ("(3,)", "-380.5547027588", "6f0822e57ac25b7e0fc83d5157cc37bf"),
    "axilla_spine_point": ("(3,)", "-543.7065620422", "ee0f5f06146eb505a83cf497ed0e8ae4"),
    "lateral_profiles_left_contour": ("(161, 2)", "-38640.8297900359", "da81b358af5a8810af2e7a77846962db"),
    "lateral_profiles_right_contour": ("(139, 2)", "-5424.4163066745", "c96b40dfc0df12f473caed54bfe9bdad"),
    "lateral_profiles_widths": ("(150,)", "37067.1757206439", "be75ecc822cfacf332bfebe617d7957e"),
    "lateral_profiles_y_centers": ("(150,)", "-27606.7594528198", "ca6dad7f7f83bc26b6366f8d6b5d5a02"),
    "neck_root_L": ("(3,)", "-451.4977548122", "b7cecf07c063940103aa279b4cdca5ac"),
    "neck_root_R": ("(3,)", "-321.3201703429", "81652627005837961040fce87c96ea0b"),
    "neck_root_spine_point": ("(3,)", "-352.1984093189", "c1889d3262d43e8854ba656e5efa4551"),
    "pelvic_line": ("(2, 3)", "-1676.5130957142", "c5f6574a07f7c98bf462d86dde1b04a1"),
    "scapular_peaks_L": ("(3,)", "-506.0485248566", "cb7447cbad2778e9745190b1baa384a7"),
    "scapular_peaks_R": ("(3,)", "-370.2922821045", "40110fe8722650a8498080b7c2c5e79a"),
    "scapular_spine_point": ("(3,)", "-423.7973251343", "2b914fa1505efab40e666febd8ed5614"),
    "shoulder_line": ("(2, 3)", "-820.3453451649", "c2c3c1363e0ed38cdea855d5723d5815"),
    "shoulder_transition_L": ("(3,)", "-522.4345016479", "aaa4596646d6a2ebae14bdd3bbd79e1b"),
    "shoulder_transition_R": ("(3,)", "-305.6081199646", "7608bc30bfae4bf90d4bc2b13e2a8390"),
    "thoracic_spine_point": ("(3,)", "-665.1596221924", "042a5f2d1a5a0284ad73edec2b8e7cda"),
    "waist_L": ("(3,)", "-957.6289749146", "48724c5f642d88b4b0a087e7d7615ba8"),
    "waist_lower_L": ("(3,)", "-1072.6211395264", "4ee97921907d4cc3ff6f79e6148fd291"),
    "waist_lower_R": ("(3,)", "-779.2889099121", "ff43376579f3d196bb169dba3718eea2"),
    "waist_lower_spine_point": ("(3,)", "-872.7204704285", "54431e5310b816ccd68fde18ae3740f8"),
    "waist_R": ("(3,)", "-666.9179840088", "6ab4ee18bf150637695fbba2028d3d47"),
    "waist_spine_point": ("(3,)", "-788.9231185913", "bf8f6756cf84c90b3c89eff9273a8917"),
}


@pytest.mark.slow
def test_extract_landmarks_golden() -> None:
    """extract_landmarks 全部 landmark 坐标与黄金值逐位一致。"""
    from landmarks.extract import extract_landmarks

    if not ROI_PLY.exists():
        pytest.skip(f"ROI mesh 缺失（敏感数据不随仓库分发，本地放置后运行）: {ROI_PLY}")
    roi_mesh = o3d.io.read_triangle_mesh(str(ROI_PLY))
    lms = extract_landmarks(roi_mesh)
    for key in sorted(lms):
        if not key.endswith("_debug"):
            walk_golden(key, lms[key], GOLDEN)
