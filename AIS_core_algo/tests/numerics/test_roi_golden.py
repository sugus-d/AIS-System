"""M2 ROI 提取黄金测试 — 固定 subject 真实网格端到端。

数据：tests/data/numerics/mesh/STD_fuse_mesh_20250619.ply（S0006 subject）。
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from tests.numerics.conftest import assert_golden, DATA_DIR

PLY_PATH = DATA_DIR / "mesh" / "STD_fuse_mesh_20250619.ply"

GOLDEN_V = ("(25495, 3)", "-15553743.6802908629", "0ea76dfdebd3efff9b402f44de9f888f")
GOLDEN_T = ("(49286, 3)", "1908437566.0000000000", "2c732e3a76bc49f8209e05389ee2bffc")


@pytest.mark.slow
def test_run_pipeline_golden() -> None:
    """run_roi_pipeline 输出的顶点/面与黄金值逐位一致。"""
    from mesh.roi.pipeline import run_roi_pipeline

    if not PLY_PATH.exists():
        pytest.skip(f"真实扫描 mesh 缺失（敏感数据不随仓库分发，本地放置后运行）: {PLY_PATH}")
    mesh = o3d.io.read_triangle_mesh(str(PLY_PATH))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles)
    roi_v, roi_t = run_roi_pipeline(vertices, triangles)
    assert_golden("roi_v", roi_v, *GOLDEN_V)
    assert_golden("roi_t", roi_t, *GOLDEN_T)
