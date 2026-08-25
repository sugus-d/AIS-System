"""编排层：背部粗糙度图渲染。

自适应阈值（_compute_adaptive_threshold）+ 粗糙度可视化，接收 CLI 参数，
调用 visualization 渲染层出图。
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from commands.plot_shared import _save_figure
from utils.logger import logger


def _compute_adaptive_threshold(mesh: o3d.geometry.TriangleMesh) -> float:
    """根据 mesh 粗糙度分布自适应计算阈值。

    取四分位距（IQR）的中段偏下位置：q25 + 0.5 * (q75 - q25)，
    在剔除高粗糙度四肢区域的同时保留躯干主要表面。
    """
    from mesh.roi.bfs import compute_mesh_roughness

    roughness = compute_mesh_roughness(mesh)
    q75 = float(np.percentile(roughness, 75))
    q25 = float(np.percentile(roughness, 25))
    threshold = round(q25 + 0.5 * (q75 - q25), 3)
    logger.info(f"Roughness: q25={q25:.3f} q75={q75:.3f} -> adaptive_threshold={threshold}")
    return threshold


def render_roughness(
    subject: str,
    output_dir: str,
    skip_run: bool,
    threshold: float | None,
) -> None:
    """用粗糙度 BFS 提取 ROI 并渲染对比图。

    工作流：加载全身 mesh → 计算自适应粗糙度阈值（或手动指定）→
    用 BFS 区域生长分离躯干与四肢 → 与原始 mesh 对比出切除三角面 → 渲染。
    """
    from mesh.roi._cut_analysis import compute_removed_triangles
    from mesh.roi.bfs import mesh_bfs
    from visualization.cut_panels import render_roi_extract_panel

    # 加载 mesh（从 subject 文件夹取最新的 STD_fuse_mesh ply 文件）
    mesh_dir = f"data/mesh/{subject}"
    files = [f for f in os.listdir(mesh_dir) if f.endswith(".ply") and "STD_fuse_mesh" in f]
    if not files:
        logger.error(f"No STD_fuse_mesh ply found for {subject} in {mesh_dir}")
        return
    mesh_path = os.path.join(mesh_dir, sorted(files)[-1])
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    ov = np.asarray(mesh.vertices)
    ot = np.asarray(mesh.triangles)
    logger.info(f"Full mesh: {len(ov)}v, {len(ot)}t")

    # 阈值
    if threshold is None:
        threshold = _compute_adaptive_threshold(mesh)

    # BFS 区域生长：从躯干种子出发，以角度 + 粗糙度联合判据逐步扩散，含四肢桥接补全
    res = mesh_bfs(
        mesh,
        angle_threshold_deg=45.0,
        roughness_threshold=threshold,
    )
    rv = np.asarray(res.vertices)
    rt = np.asarray(res.triangles)

    kpct = 100 * len(rv) / len(ov)
    yspan = rv[:, 1].max() - rv[:, 1].min()
    logger.info(
        f"th={threshold}: kept={len(rv)}v ({kpct:.0f}%), "
        f"Y=[{rv[:, 1].min():.0f},{rv[:, 1].max():.0f}], span={yspan:.0f}mm"
    )

    # 切除三角面
    rem = compute_removed_triangles(ov, ot, rv)
    rem_set = {int(ti) for ti in rem}
    logger.info(f"removed={len(rem_set)}t")

    # 渲染
    fig, ax = plt.subplots(figsize=(8, 10))
    render_roi_extract_panel(
        ax,
        rv,
        rt,
        ov,
        ot,
        rem_set,
        title=(
            f"subject={subject}  rough_th={threshold}\n"
            f"kept={len(rv)}v ({kpct:.0f}%)  Y={rv[:, 1].min():.0f}~{rv[:, 1].max():.0f}mm\n"
            f"removed={len(rem_set)}t"
        ),
    )

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    _save_figure(fig, os.path.join(output_dir, f"roughness_{subject}.png"))
