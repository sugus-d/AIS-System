"""编排层：UV 参数化可视化。

渲染测地边界、内外判定与参数化结果（复用 mesh/parameterization 计算，
调用 visualization 渲染层出图）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
from matplotlib.path import Path as MPath

from commands.plot_parameterization_measures import render_measures
from commands.plot_shared import _save_figure
from parameterization.template import TEMPLATE_LANDMARKS  # noqa: E402
from utils.logger import logger  # noqa: E402

# 平滑所需的最少点数，少于该值跳过平滑（点数太少时高斯滤波会失真）
_MIN_SMOOTH_POINTS = 5

_OUTER_NAMES: list[str] = [
    "neck_root_L",
    "shoulder_transition_L",
    "axilla_L",
    "waist_L",
    "waist_lower_L",
    "waist_lower_spine_point",
    "waist_lower_R",
    "waist_R",
    "axilla_R",
    "shoulder_transition_R",
    "neck_root_R",
    "neck_root_spine_point",
]


def _geodesic_boundary(
    V: np.ndarray,
    F: np.ndarray,
    k: np.ndarray,
    y: np.ndarray,
    smoothing: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build geodesic boundary through outer landmarks.

    通过模板 landmark 中的"外部"点（颈根、肩转、腋窝、腰、脊柱端），
    沿 mesh 表面计算首尾相连的测地线，形成闭合边界。
    各段测地线用高斯滤波平滑后再映射回顶点索引，拼接为完整边界环。
    """
    import pygeodesic.geodesic as geo
    from scipy.ndimage import gaussian_filter1d

    from parameterization.procrustes import compute_procrustes

    names = list(TEMPLATE_LANDMARKS.keys())
    outer_idx = [names.index(n) for n in _OUTER_NAMES]
    ov = [int(k[i]) for i in outer_idx]

    geoalg = geo.PyGeodesicAlgorithmExact(V, F)
    bverts: list[int] = []
    for i in range(len(ov)):
        s, d = ov[i], ov[(i + 1) % len(ov)]
        # 沿曲面计算两点间最短路径（测地线）
        _, path_raw = geoalg.geodesicDistance(s, d)
        path = list(reversed(path_raw))
        # 将测地线路径点映射到最近的 mesh 顶点
        pv: list[int] = []
        for pt in path:
            vi = int(np.argmin(np.linalg.norm(V - np.array(pt), axis=1)))
            if not pv or vi != pv[-1]:
                pv.append(vi)
        pts = V[pv].copy()
        # 高斯平滑消除锯齿，保持端点位置固定
        if len(pts) > _MIN_SMOOTH_POINTS:
            pts = gaussian_filter1d(pts, sigma=smoothing, axis=0)
            pts[0] = V[pv[0]]
            pts[-1] = V[pv[-1]]
        seg: list[int] = []
        for pt in pts:
            vi = int(np.argmin(np.linalg.norm(V - pt, axis=1)))
            if not seg or vi != seg[-1]:
                seg.append(vi)
        # 跳过上一段的最后一个顶点（避免重复）
        if i > 0:
            seg = seg[1:]
        bverts.extend(seg)
    bverts_arr = np.array(bverts, dtype=np.int64)

    # Procrustes 对齐：将 landmark 模板 (y) 仿射到 mesh 对应点 (k) 的二维投影
    s, R, t = compute_procrustes(V[k, :2], y)
    Va = V.copy()
    Va[:, :2] = s * V[:, :2] @ R.T + t

    return bverts_arr, Va, np.array(ov)


def _inside_outside(
    V: np.ndarray,
    F: np.ndarray,
    k: np.ndarray,
    bverts: np.ndarray,
    Va: np.ndarray,
) -> np.ndarray:
    """Return boolean mask (N,) — True = inside the geodesic boundary.

    合并两种策略：
    1. 二维多边形包含测试——将对齐后的顶点投影到对齐平面，判断是否在边界多边形内部；
    2. 三维 BFS 泛洪——从 neck_root_spine_point 种子点出发，沿邻接边在不跨越边界顶点的前提下扩散。
    最终取二者并集，确保无遗漏。
    """
    from collections import deque

    NV = len(V)
    bpath = MPath(Va[bverts, :2], closed=True)
    in_2d = bpath.contains_points(Va[:, :2])

    lm_set = set(int(vi) for vi in k)
    bset = set(int(v) for v in bverts) - lm_set
    # 构建顶点邻接表，用于 BFS 泛洪
    al: list[list[int]] = [[] for _ in range(NV)]
    for a, b, c in F:
        al[a].extend([b, c])
        al[b].extend([a, c])
        al[c].extend([a, b])
    al = [list(set(nb)) for nb in al]

    names = list(TEMPLATE_LANDMARKS.keys())
    seed = int(k[names.index("neck_root_spine_point")])
    # 从脊柱中点种子出发，BFS 泛洪（不跨越边界顶点）
    inside_3d = {seed}
    q: deque[int] = deque([seed])
    while q:
        v = q.popleft()
        for nb in al[v]:
            if nb not in inside_3d and nb not in bset:
                inside_3d.add(nb)
                q.append(nb)
    inside_3d.update(bverts)
    # 向内扩张 2 层，填补边界附近的缝隙
    for _ in range(2):
        frontier: set[int] = set()
        for v in inside_3d:
            for nb in al[v]:
                if nb not in inside_3d:
                    frontier.add(nb)
        inside_3d.update(frontier)

    mask = np.zeros(NV, dtype=bool)
    mask[list(inside_3d)] = True
    in_2d_idx = np.where(in_2d)[0]
    mask[in_2d_idx] = True
    return mask


def render_parameterization(
    subject: str,
    output_dir: str,
    skip_run: bool,
    cut_only: bool,
    show_heightmap: bool,
    show_surfaces: bool,
    smoothing: float,
) -> None:
    """Render geodesic cut and harmonic parameterization."""
    from parameterization.landmark_io import find_landmark_vertices, parse_landmarks_json
    from visualization.parameterization_panels import draw_cut

    roi_dir = Path("results/roi") / subject
    if not roi_dir.exists():
        logger.error(f"ROI dir not found: {roi_dir}")
        return

    # Load
    mesh = o3d.io.read_triangle_mesh(str(roi_dir / "roi.ply"))
    V0 = np.asarray(mesh.vertices, dtype=np.float64)
    ms = mesh.simplify_quadric_decimation(10000)
    V = np.asarray(ms.vertices, dtype=np.float64)
    F = np.asarray(ms.triangles, dtype=np.int64)
    lm = parse_landmarks_json(f"results/ground-truth/{subject}/ground_truth.json")
    k_orig, y = find_landmark_vertices(mesh, lm, TEMPLATE_LANDMARKS)
    k = np.array([np.argmin(np.linalg.norm(V - V0[ki], axis=1)) for ki in k_orig])

    # Geodesic boundary
    bverts, Va, ov = _geodesic_boundary(V, F, k, y, smoothing)

    # Inside/outside mask
    mask = _inside_outside(V, F, k, bverts, Va)

    # Draw cut（编排层创建 figure；内部标注点索引按 TEMPLATE_LANDMARKS 顺序计算）
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    cut_path = out_dir / "boundary_cut" / f"geodesic_cut_{subject}.png"
    cut_path.parent.mkdir(parents=True, exist_ok=True)
    _template_names = list(TEMPLATE_LANDMARKS.keys())
    _inner_pos = [_template_names.index(n) for n in
                  ["scapular_peaks_L", "scapular_peaks_R", "scapular_spine_point", "axilla_spine_point"]]
    _inner_idx = np.asarray([k[p] for p in _inner_pos], dtype=np.int64)
    fig_cut, ax_cut = plt.subplots(figsize=(12, 12))
    draw_cut(ax_cut, V, Va, bverts, ov, _inner_idx, mask, subject)
    _save_figure(fig_cut, cut_path)

    # ── 综合可视化（--show-heightmap 或 --show-surfaces）──
    if show_heightmap or show_surfaces:
        param_dir = out_dir / "boundary_cut" / subject
        uv_path = param_dir / "uv_coords.npy"
        mesh_path = param_dir / "mesh_cut.ply"
        if not uv_path.exists():
            logger.error(f"UV not found: {uv_path}, run pipeline first")
            return
        render_measures(subject, out_dir, uv_path, str(mesh_path))
