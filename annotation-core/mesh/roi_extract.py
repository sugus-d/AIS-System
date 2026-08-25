"""域级 ROI 调度 — BFS 生长 → xy-hull 提取 → 法线角切割（头发/布料突出物）。

# 注意：本文件是 mesh/roi/ 包的功能组件编排者（生产路径是 commands/batch_process_all.py::run_roi_pipeline），mesh/roi/extract.py 是其中的 extract_by_xy_hull 组件，两者职责不同。

调度已从 mesh/roi/ 包外置，mesh/roi/ 只保留功能组件（bfs/cleanup/extract）。
本模块 :func:`extract_back_roi` 是早期实现，被 labeling 平台算法注册表
（mesh/roi/registry.py）与测试引用；生产路径是 commands/batch_process_all.py
::run_roi_pipeline（BFS 生长 → 边界腐蚀去衣服 → 曲线切割去裤子）。
"""

import numpy as np
import open3d as o3d

from utils.logger import logger
from utils.mesh import clone_mesh

from .roi.bfs import largest_component, mesh_bfs
from .roi.cleanup import cut_normal_angle_pipeline
from .roi.extract import extract_by_xy_hull


def extract_back_roi(
    mesh: o3d.geometry.TriangleMesh,
    return_debug_data: bool = False,
    grid_resolution_mm: float = 5.0,
    subject_id: str = "unknown",
    angle_threshold_deg: float = 45.0,
    roughness_threshold: float = 0.25,
    roughness_radius: float = 20.0,
    cut_angle_deg: float | None = 15.0,
    cut_dilation: int = 0,
    cut_min_area: float = 150.0,
    cut_min_al_ratio: float = 5.0,
    **kwargs: object,
) -> o3d.geometry.TriangleMesh:
    """Extract back ROI by mesh-level BFS on original triangles.

    Extended with an optional post-processing cut that removes near-horizontal
    protrusions (hair, fabric strips) via normal-angle detection.

    Parameters
    ----------
    cut_angle_deg:
        Angle threshold for protrusion cutting (degrees).  ``None`` (default)
        skips the cut entirely.  A value of 10–20° is typical.
    cut_dilation:
        Dilation steps for the bridge region before cutting (default 0).
    cut_min_area:
        Minimum area (mm²) for a cut segment to be considered valid; smaller
        segments are re-attached (default 150).
    cut_min_al_ratio:
        Minimum area / boundary-length ratio for a valid cut segment
        (default 5.0).
    """
    if mesh.is_empty() or not mesh.has_triangles():
        empty = clone_mesh(mesh)
        return (empty, {}) if return_debug_data else empty

    result = mesh_bfs(
        mesh,
        angle_threshold_deg=angle_threshold_deg,
        roughness_threshold=roughness_threshold,
        roughness_radius=roughness_radius,
    )
    logger.info(f"BFS: {len(np.asarray(result.vertices))}v, {len(np.asarray(result.triangles))}t")

    result = largest_component(result)
    result = extract_by_xy_hull(mesh, result)

    # ── Optional protrusion cut ──────────────────────────────────────────
    if cut_angle_deg is not None:
        result, analysis, _ = cut_normal_angle_pipeline(
            result,
            angle_threshold=cut_angle_deg,
            dilation_steps=cut_dilation,
            min_area=cut_min_area,
            min_al_ratio=cut_min_al_ratio,
        )
        logger.info(
            "Protrusion cut (angle=%s°): %d removals",
            cut_angle_deg,
            len(analysis.get("removals", [])),
        )

    if not return_debug_data:
        return result
    return result, {}


__all__ = [
    "extract_back_roi",
]
