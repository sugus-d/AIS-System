"""ROI 提取管线 — BFS 生长 → 边界腐蚀去衣服 → 曲线切割去裤子。

生产管线（commands/batch_process_all、prediction 自动模式、评估脚本共用）。
所有切割均沿自然分界线（粗糙度边界）曲线进行，禁止超过 3cm 的直线切割。
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from mesh.roi._mesh_erosion import strip_boundary_tris
from mesh.roi._pants_cut import remove_pants
from mesh.roi.bfs import largest_component, mesh_bfs
from utils.logger import logger

# ROI 管线 BFS 结果顶点数下限（太少跳过后续切割）
_MIN_BFS_VERTS = 100


def run_roi_pipeline(
    vertices: np.ndarray,
    triangles: np.ndarray,
    roughness_threshold: float = 0.20,
    angle_threshold_deg: float = 45.0,
) -> tuple[np.ndarray, np.ndarray]:
    """完整 ROI 提取管线：BFS 生长 → 去衣服 → 去裤子。

    Parameters
    ----------
    vertices : (N, 3) 顶点坐标。
    triangles : (M, 3) 三角面索引。
    roughness_threshold : BFS 粗糙度停止阈值，默认 0.20（调优值）。
    angle_threshold_deg : 法线角阈值，默认 45°。

    Returns
    -------
    (result_vertices, result_triangles): 裁剪后 mesh。
    """
    # ── Layer 1: BFS region growing ──
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    bf = mesh_bfs(
        mesh,
        angle_threshold_deg=angle_threshold_deg,
        roughness_threshold=roughness_threshold,
        fill_holes=True,
        max_hole_boundary=200,
        max_hole_area=5000,
    )
    bf = largest_component(bf)
    mv = np.asarray(bf.vertices, dtype=np.float64)
    mt = np.asarray(bf.triangles, dtype=np.int32)
    logger.info("BFS: %dv / %dt", len(mv), len(mt))
    if len(mv) < _MIN_BFS_VERTS:
        return mv, mt

    # ── Layer 2: Remove clothing (boundary erosion) ──
    mv, mt = strip_boundary_tris(mv, mt, iterations=3)

    # ── Layer 3: Remove pants ──
    mv, mt = remove_pants(mv, mt)

    logger.info("Pipeline result: %dv / %dt (%d removed total)", len(mv), len(mt), len(triangles) - len(mt))
    return mv, mt
