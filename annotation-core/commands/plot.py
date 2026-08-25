#!/usr/bin/env python3
"""统一绘图入口——按 --domain 选择绘图类型。

合并自 plot_compare_subjects / plot_landmarks / plot_parameterization / plot_roughness 四个脚本。

用法:
    uv run python -m commands.plot --domain compare S0119,S0113
    uv run python -m commands.plot --domain landmarks S0004
    uv run python -m commands.plot --domain parameterization S0004 --show-heightmap
    uv run python -m commands.plot --domain roughness S0119 --threshold 0.20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.path import Path as MPath

from parameterization.template import TEMPLATE_LANDMARKS  # noqa: E402
from utils.logger import logger  # noqa: E402

# 各 domain 的默认输出目录（与原独立脚本一致）
_DEFAULT_OUTPUT_DIRS = {
    "compare": "results/archive/debug_roi",
    "landmarks": "results/landmarks",
    "parameterization": "results/parameterization",
    "roughness": "results/archive/debug_roi",
}


def _save_figure(
    fig: plt.Figure, out_path: str | Path, facecolor: str = "white", pad_inches: float | None = None
) -> None:
    """统一保存 figure：dpi/裁剪参数一致，保存后关闭并记录日志。"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=pad_inches, facecolor=facecolor)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# domain: compare —— 多 subject 法线切除结果对比
# ---------------------------------------------------------------------------


def _split_into_full_components(
    removed_tri_indices: list[int],
    original_triangles: np.ndarray,
    full_adjacency: list[set[int]] | None = None,
) -> list[list[int]]:
    """将切除的三角面按连通性拆分，每个连通分量作为一个独立组件返回。

    组件拆分后，每个组件可以独立判断是否为有效切除（valid），避免将整个切除区域
    一刀切地标记为有效或无效——同一切除操作的不同连通分量可能有不同分类。
    """
    from mesh.roi._mesh_graph import build_triangle_adjacency, find_connected_components

    if not removed_tri_indices:
        return []
    removed_set = set(removed_tri_indices)
    if full_adjacency is None:
        full_adjacency = build_triangle_adjacency(original_triangles)
    local_to_global = list(removed_set)
    global_to_local = {g: li for li, g in enumerate(local_to_global)}
    local_adj: list[set[int]] = []
    for gidx in local_to_global:
        local_adj.append({global_to_local[n] for n in full_adjacency[gidx] if n in removed_set})
    raw = find_connected_components(local_adj)
    return [[local_to_global[li] for li in comp] for comp in raw]


def _classify_component_by_analysis(
    component: list[int],
    original_triangles: np.ndarray,
    analysis_removals: list[dict],
) -> bool:
    """判断一个切除连通分量是否属于有效切除。

    遍历所有 analysis 中标记为 valid 的切除操作，若该分量中有三角面出现在
    任一 valid 切除的 triangle_indices 列表中，则该分量被视为有效切除。
    """
    comp_set = set(component)
    for rem in analysis_removals:
        if not rem.get("valid"):
            continue
        for ti in rem.get("triangle_indices", []):
            if ti in comp_set:
                return True
    return False


def _prepare_panel_data(
    original_triangles: np.ndarray,
    analysis: dict,
    removed_tris: list[int],
) -> dict:
    """准备面板数据：将切除三角面拆分为有效区域（valid）和回填区域（invalid）。

    有效区域 = 与 analysis 中 valid 切除操作有交集的连通分量
    无效区域 = analysis.restored_tris（被回填的三角面）
    返回 dict 含 valid_tris 和 invalid_tris 两个 set。
    """
    from mesh.roi._mesh_graph import build_triangle_adjacency

    full_adj = build_triangle_adjacency(original_triangles)
    analysis_rems = analysis.get("removals", [])
    components = _split_into_full_components(removed_tris, original_triangles, full_adj)
    valid_tri_set: set[int] = set()
    for comp in components:
        if _classify_component_by_analysis(comp, original_triangles, analysis_rems):
            valid_tri_set.update(comp)
    invalid_tri_set: set[int] = set(analysis.get("restored_tris", []))
    return {"valid_tris": valid_tri_set, "invalid_tris": invalid_tri_set}


def render_compare(
    subject: str,
    output_dir: str,
    skip_run: bool,
    angle: int,
    dilate: int,
    min_area: float,
    min_al_ratio: float,
) -> None:
    """多 subject 法线切除结果对比。subject 可传入逗号分隔的多个 ID。

    生成 3×N 网格，每个格子显示一个 subject 在给定角度下的切除分析结果。
    绿色 = 回填区域（无效），红色 = 有效切除区域。
    """
    from mesh.roi.cleanup import cut_normal_angle_pipeline
    from visualization.cut_panels import (
        _setup_panel_axes,
        draw_cut_boundary_edges,
        render_invalid_removed_area,
        render_mesh_panel,
        render_valid_removed_area,
    )

    subject_ids = [s.strip() for s in subject.split(",")]
    n = len(subject_ids)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 5, rows * 5))
    grid = fig.add_gridspec(rows, cols, wspace=0.05, hspace=0.15, left=0.01, right=0.99, top=0.97, bottom=0.01)

    for idx, subject_id in enumerate(subject_ids):
        mesh_filepath = os.path.join("results", "cache", subject_id, "extract_roi", "output.ply")
        if not os.path.exists(mesh_filepath):
            logger.warning(f"Skip {subject_id}: mesh not found")
            continue

        try:
            original_mesh = o3d.io.read_triangle_mesh(mesh_filepath)
            original_mesh.compute_vertex_normals()
            original_vertices = np.asarray(original_mesh.vertices, dtype=np.float64)
            original_triangles = np.asarray(original_mesh.triangles)

            kept, analysis, removed = cut_normal_angle_pipeline(
                original_mesh,
                angle,
                dilate,
                min_area=min_area,
                min_al_ratio=min_al_ratio,
            )
            logger.info(f"[{idx + 1}/{n}] {subject_id}: {len(removed)} removed")

            pd = _prepare_panel_data(original_triangles, analysis, removed)

            row, col = divmod(idx, cols)
            ax = fig.add_subplot(grid[row, col])
            _setup_panel_axes(ax)

            kv = np.asarray(kept.vertices, dtype=np.float64)
            kt = np.asarray(kept.triangles)
            render_mesh_panel(ax, kv, kt)
            render_valid_removed_area(ax, original_vertices, original_triangles, pd["valid_tris"])
            render_invalid_removed_area(ax, original_vertices, original_triangles, pd["invalid_tris"])
            draw_cut_boundary_edges(ax, analysis.get("cut_edges", []), analysis.get("segments", []), original_vertices)

            n_valid = sum(1 for r in analysis["removals"] if r["valid"])
            n_restored = len(analysis.get("restored_tris", []))
            vertices_removed = len(original_vertices) - len(kv)
            ax.set_title(
                f"{subject_id}  -{vertices_removed}v\nvalid: {n_valid}  restored: {n_restored}",
                fontsize=8,
                pad=2,
            )

        except Exception as exc:
            logger.error(f"[{idx + 1}/{n}] {subject_id}: {exc}")
            row, col = divmod(idx, cols)
            ax = fig.add_subplot(grid[row, col])
            ax.text(0.5, 0.5, f"{subject_id}\nERROR", ha="center", va="center", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    os.makedirs(output_dir, exist_ok=True)
    _save_figure(fig, os.path.join(output_dir, "subjects_comparison.png"), pad_inches=0)


# ---------------------------------------------------------------------------
# domain: landmarks —— 解剖标注叠加在曲率图上（双面板）
# ---------------------------------------------------------------------------


def render_landmarks(
    subject: str,
    cache_dir: str,
    output_dir: str,
    skip_run: bool,
) -> None:
    """渲染带解剖标注的曲率图（2 面板）。

    上方面板：曲率图 + 所有解剖标注点（neck_root、shoulder_transition、axilla、waist、scapular_peaks、spine），
    下方面板：waist 调试信息（仅在存在时显示）。

    waist 定位依赖轮廓窄茎最细处，结果易受噪声影响，
    因此需要独立的调试面板来验证 waist 是否定位在正确的 Y 层级。
    """
    from utils.io import load_landmarks
    from utils.mesh import load_cached_mesh
    from visualization._data_utils import load_cached_numpy
    from visualization.landmarks_panels import render_curvature_landmarks_panel, render_waist_debug_panel

    # 加载缓存
    lmks = load_landmarks(subject, cache_dir)
    mesh = load_cached_mesh(subject, cache_dir)
    curv = load_cached_numpy(cache_dir, subject, "curvature", "mean_curvature.npy")

    missing: list[str] = []
    if lmks is None:
        missing.append("landmarks.pkl")
    if mesh is None:
        missing.append("cached mesh")
    if curv is None:
        missing.append("curvature/mean_curvature.npy")

    if missing:
        logger.error(f"Missing cached inputs for {subject}: {', '.join(missing)}")
        return

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        logger.error(f"Mesh has no triangles for {subject}")
        return

    out_dir = os.path.join(output_dir, subject)
    os.makedirs(out_dir, exist_ok=True)

    # 双面板布局：上=曲率+landmarks，下=waist 调试
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16), facecolor="black")

    # 主面板：曲率图 + 各解剖标记点叠层
    render_curvature_landmarks_panel(ax1, vertices, triangles, curv, lmks, subject)

    # waist 定位算法需要额外验证，因此设独立调试面板
    # 仅当 pipeline 输出了 waist_debug 数据时才显示，否则隐藏
    waist_debug = lmks.get("waist_debug", {})
    if waist_debug:
        render_waist_debug_panel(ax2, waist_debug)
    else:
        ax2.set_visible(False)

    _save_figure(fig, os.path.join(out_dir, "landmarks.png"), facecolor="black")


# ---------------------------------------------------------------------------
# domain: parameterization —— 测地线切分与调和参数化
# ---------------------------------------------------------------------------

# 平滑所需的最少点数，少于该值跳过平滑（点数太少时高斯滤波会失真）
_MIN_SMOOTH_POINTS = 5
# subplot 网格中的列索引：3D 列与 UV 列
_3D_COLUMN = 2
_UV_COLUMN = 3

_OUTER_NAMES: list[str] = [
    "neck_root_L",
    "shoulder_transition_L",
    "axilla_L",
    "waist_L",
    "waist_lower_L",
    "spine_P4",
    "waist_lower_R",
    "waist_R",
    "axilla_R",
    "shoulder_transition_R",
    "neck_root_R",
    "spine_P0",
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
    2. 三维 BFS 泛洪——从 spine_P0 种子点出发，沿邻接边在不跨越边界顶点的前提下扩散。
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
    seed = int(k[names.index("spine_P0")])
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

    # Draw cut
    out_dir = Path(output_dir)
    cut_path = out_dir / "boundary_cut" / f"geodesic_cut_{subject}.png"
    cut_path.parent.mkdir(parents=True, exist_ok=True)
    fig_cut, ax_cut = draw_cut(V, Va, k, bverts, ov, mask, subject)
    _save_figure(fig_cut, cut_path)

    # ── 综合可视化（--show-heightmap 或 --show-surfaces）──
    if show_heightmap or show_surfaces:
        from mesh.curvature import calculate_curvature
        from mesh.roi.bfs import compute_mesh_roughness

        param_dir = out_dir / "boundary_cut" / subject
        uv_path = param_dir / "uv_coords.npy"
        mesh_path = param_dir / "mesh_cut.ply"
        if not uv_path.exists():
            logger.error(f"UV not found: {uv_path}, run pipeline first")
            return

        uv = np.load(uv_path)
        cm = o3d.io.read_triangle_mesh(str(mesh_path))
        Vc = np.asarray(cm.vertices)
        Fc = np.asarray(cm.triangles)

        # ── 计算 5 个指标 ──
        h = Vc[:, 2]
        km = calculate_curvature(cm, "mean")
        kg = calculate_curvature(cm, "gaussian")
        if km is None:
            km = np.zeros(len(Vc))
        if kg is None:
            kg = np.zeros(len(Vc))
        rough_f = compute_mesh_roughness(cm)
        rough_v = np.zeros(len(Vc))
        np.add.at(rough_v, Fc.ravel(), np.repeat(rough_f, 3))
        fcnt = np.zeros(len(Vc))
        np.add.at(fcnt, Fc.ravel(), 1)
        rough_v /= np.maximum(fcnt, 1)
        cm.compute_vertex_normals()
        vn = np.asarray(cm.vertex_normals)
        na = np.degrees(np.arccos(np.clip(np.abs(vn[:, 1]), 0, 1)))

        # 地标 3D 位置（UV 反查）
        lm_v = {}
        for name, target_uv in TEMPLATE_LANDMARKS.items():
            vi = np.argmin(np.linalg.norm(uv - np.array(target_uv), axis=1))
            lm_v[name] = Vc[vi]

        # 成对连线（左右对称）
        _LR_PAIRS = [
            ("neck_root_L", "neck_root_R"),
            ("shoulder_transition_L", "shoulder_transition_R"),
            ("scapular_peaks_L", "scapular_peaks_R"),
            ("axilla_L", "axilla_R"),
            ("waist_L", "waist_R"),
            ("waist_lower_L", "waist_lower_R"),
        ]
        _SPINE_CHAIN = ["spine_P0", "spine_P1", "spine_P2", "spine_P5", "spine_P3", "spine_P4"]

        # 色彩范围
        def _clim(arr: np.ndarray) -> float:
            return max(np.median(np.abs(arr[np.isfinite(arr)])) * 5, 1e-6)

        # 3×4 布局
        fig, axes = plt.subplots(3, 4, figsize=(22, 15))
        fig.suptitle(f"Surface measures — {subject}", fontsize=14, y=0.98)

        # ═══ Row 0: Landmarks (gray) + Height ═══

        # 预构建 UV→3D 插值器（三角网格内线性插值，比 argmin 平滑）
        from scipy.interpolate import LinearNDInterpolator

        _uv_interp_x = LinearNDInterpolator(uv, Vc[:, 0], fill_value=np.nan)
        _uv_interp_y = LinearNDInterpolator(uv, Vc[:, 1], fill_value=np.nan)

        def _uv_curve_to_3d(name_a: str, name_b: str, n_samples: int = 100) -> np.ndarray:
            """Sample a UV line between two landmarks, interpolate 3D smoothly."""
            uva = np.array(TEMPLATE_LANDMARKS[name_a], dtype=np.float64)
            uvb = np.array(TEMPLATE_LANDMARKS[name_b], dtype=np.float64)
            uv_line = np.column_stack(
                [
                    np.linspace(uva[0], uvb[0], n_samples),
                    np.linspace(uva[1], uvb[1], n_samples),
                ]
            )
            xs = _uv_interp_x(uv_line)
            ys = _uv_interp_y(uv_line)
            valid = ~(np.isnan(xs) | np.isnan(ys))
            return np.column_stack([xs[valid], ys[valid]])

        # --- 左: 3D landmark 连线 (曲线) ---
        ax = axes[0, 0]
        ax.tripcolor(
            Vc[:, 0],
            Vc[:, 1],
            Fc,
            np.ones(len(Vc)),
            cmap="Greys",
            vmin=0,
            vmax=2,
            alpha=0.5,
            edgecolors="#bbb",
            linewidths=0.1,
        )
        ax.set_aspect("equal")
        ax.set_title("Landmarks (3D)", fontsize=11)

        # 脊柱曲线
        spine_curve = np.concatenate(
            [_uv_curve_to_3d(_SPINE_CHAIN[i], _SPINE_CHAIN[i + 1]) for i in range(len(_SPINE_CHAIN) - 1)]
        )
        ax.plot(spine_curve[:, 0], spine_curve[:, 1], "r-", lw=2, alpha=0.7, zorder=3)

        # 左右对称曲线
        for l_name, r_name in _LR_PAIRS:
            curve = _uv_curve_to_3d(l_name, r_name)
            ax.plot(curve[:, 0], curve[:, 1], "b-", lw=1.5, alpha=0.5, zorder=3)

        # 地标点
        for n in list(TEMPLATE_LANDMARKS.keys()):
            p = lm_v[n][:2]
            is_spine = n.startswith("spine")
            ax.scatter(
                p[0],
                p[1],
                c="gold" if is_spine else "cyan",
                s=40 if is_spine else 30,
                marker="D" if is_spine else "o",
                edgecolors="k",
                linewidths=0.5,
                zorder=5,
            )
            short = n.replace("shoulder_transition", "st").replace("scapular_peaks", "sp")
            short = short.replace("waist_lower", "wl").replace("neck_root", "nr").replace("waist", "wa")
            ax.annotate(short, (p[0], p[1]), fontsize=5, ha="left", va="bottom", color="black", zorder=6)

        # --- 右: UV landmark 连线 (直线) ---
        ax = axes[0, 1]
        ax.tripcolor(
            uv[:, 0],
            uv[:, 1],
            Fc,
            np.ones(len(uv)),
            cmap="Greys",
            vmin=0,
            vmax=2,
            alpha=0.5,
            edgecolors="#bbb",
            linewidths=0.1,
        )
        ax.set_aspect("equal")
        ax.set_title("Landmarks (UV)", fontsize=11)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-4.5, 2.5)

        for i in range(len(_SPINE_CHAIN) - 1):
            uv_a = TEMPLATE_LANDMARKS[_SPINE_CHAIN[i]]
            uv_b = TEMPLATE_LANDMARKS[_SPINE_CHAIN[i + 1]]
            ax.plot([uv_a[0], uv_b[0]], [uv_a[1], uv_b[1]], "r-", lw=2, alpha=0.7, zorder=3)

        for l_name, r_name in _LR_PAIRS:
            uv_a = TEMPLATE_LANDMARKS[l_name]
            uv_b = TEMPLATE_LANDMARKS[r_name]
            ax.plot([uv_a[0], uv_b[0]], [uv_a[1], uv_b[1]], "b-", lw=1.5, alpha=0.5, zorder=3)

        for n in list(TEMPLATE_LANDMARKS.keys()):
            u, v = TEMPLATE_LANDMARKS[n]
            is_spine = n.startswith("spine")
            ax.scatter(
                u,
                v,
                c="gold" if is_spine else "cyan",
                s=40 if is_spine else 30,
                marker="D" if is_spine else "o",
                edgecolors="k",
                linewidths=0.5,
                zorder=5,
            )
            short = n.replace("shoulder_transition", "st").replace("scapular_peaks", "sp")
            short = short.replace("waist_lower", "wl").replace("neck_root", "nr").replace("waist", "wa")
            ax.annotate(short, (u, v), fontsize=5, ha="left", va="bottom", color="black", zorder=6)

        # Height (3D, UV)
        for col, pts in [(_3D_COLUMN, Vc), (_UV_COLUMN, uv)]:
            ax = axes[0, col]
            ax.tripcolor(pts[:, 0], pts[:, 1], Fc, h, cmap="viridis", shading="gouraud")
            if col == _UV_COLUMN:
                ax.set_xlim(-3, 3)
                ax.set_ylim(-4.5, 2.5)
            ax.set_aspect("equal")
            ax.set_title("Height (3D)" if col == _3D_COLUMN else "Height (UV)", fontsize=11)

        # Height (3D, UV)
        for col, pts, title in [(_3D_COLUMN, Vc, "Height (3D)"), (_UV_COLUMN, uv, "Height (UV)")]:
            ax = axes[0, col]
            ax.tripcolor(pts[:, 0], pts[:, 1], Fc, h, cmap="viridis", shading="gouraud")
            if col == _UV_COLUMN:
                ax.set_xlim(-3, 3)
                ax.set_ylim(-4.5, 2.5)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=11)

        # ═══ Row 1: Mean Curv + Gauss Curv ═══
        for pair_idx, (data, name_base) in enumerate([(km, "Mean Curv"), (kg, "Gauss Curv")]):
            clm = (-_clim(data), _clim(data))
            for col_offset, (pts, suffix) in enumerate([(Vc, "3D"), (uv, "UV")]):
                ax = axes[1, pair_idx * 2 + col_offset]
                ax.tripcolor(pts[:, 0], pts[:, 1], Fc, data, cmap="jet", shading="gouraud", vmin=clm[0], vmax=clm[1])
                if col_offset == 1:
                    ax.set_xlim(-3, 3)
                    ax.set_ylim(-4.5, 2.5)
                ax.set_aspect("equal")
                ax.set_title(f"{name_base} ({suffix})", fontsize=11)

        # ═══ Row 2: Roughness + Normal Angle ═══
        r_clim = tuple(np.percentile(rough_v[np.isfinite(rough_v)], [5, 95]))
        for pair_idx, (data, name_base, clm_in) in enumerate(
            [(rough_v, "Roughness", r_clim), (na, "Normal Angle", (45, 90))]
        ):
            for col_offset, (pts, suffix) in enumerate([(Vc, "3D"), (uv, "UV")]):
                ax = axes[2, pair_idx * 2 + col_offset]
                ax.tripcolor(
                    pts[:, 0], pts[:, 1], Fc, data, cmap="jet", shading="gouraud", vmin=clm_in[0], vmax=clm_in[1]
                )
                if col_offset == 1:
                    ax.set_xlim(-3, 3)
                    ax.set_ylim(-4.5, 2.5)
                ax.set_aspect("equal")
                ax.set_title(f"{name_base} ({suffix})", fontsize=11)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        _save_figure(fig, out_dir / "boundary_cut" / subject / "all_measures.png")


# ---------------------------------------------------------------------------
# domain: roughness —— 粗糙度引导的 BFS 区域生长结果
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI 主入口
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """构建 argparse 主入口，注册全部 domain 的公共与独有参数。"""
    parser = argparse.ArgumentParser(
        prog="plot.py",
        description="统一绘图入口：--domain 选择绘图类型（compare/landmarks/parameterization/roughness）。",
    )
    parser.add_argument("--domain", choices=sorted(_DEFAULT_OUTPUT_DIRS), required=True, help="绘图类型")
    parser.add_argument("subject", help="subject ID（compare 域可传逗号分隔的多个 ID）")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认按 domain 取各自默认值）")
    parser.add_argument("--skip-run", action="store_true", help="跳过 pipeline 重建缓存")
    # compare 独有
    parser.add_argument("--angle", type=int, default=15, help="切除角度")
    parser.add_argument("--dilate", type=int, default=0, help="桥面扩张层数")
    parser.add_argument("--min-area", type=float, default=150.0, help="最小面积 mm²")
    parser.add_argument("--min-al-ratio", type=float, default=5.0, help="面积/边长比")
    # landmarks 独有
    parser.add_argument("--cache-dir", default="results/cache", help="缓存目录")
    # parameterization 独有
    parser.add_argument("--cut-only", action="store_true", help="只画 cut 图")
    parser.add_argument("--show-heightmap", action="store_true", help="显示高度图")
    parser.add_argument("--show-surfaces", action="store_true", help="显示粗糙程度和法向量")
    parser.add_argument("--smoothing", type=float, default=5.0, help="测地线平滑 sigma")
    # roughness 独有
    parser.add_argument("--threshold", "-t", type=float, default=None, help="粗糙度阈值，不指定则自适应")
    return parser


def main() -> None:
    """CLI 入口：解析参数后按 --domain 分发到对应渲染函数。"""
    matplotlib.use("Agg")
    args = _build_parser().parse_args()
    output_dir = args.output_dir or _DEFAULT_OUTPUT_DIRS[args.domain]

    if args.domain == "compare":
        render_compare(
            args.subject, output_dir, args.skip_run, args.angle, args.dilate, args.min_area, args.min_al_ratio
        )
    elif args.domain == "landmarks":
        render_landmarks(args.subject, args.cache_dir, output_dir, args.skip_run)
    elif args.domain == "parameterization":
        render_parameterization(
            args.subject,
            output_dir,
            args.skip_run,
            args.cut_only,
            args.show_heightmap,
            args.show_surfaces,
            args.smoothing,
        )
    else:
        render_roughness(args.subject, output_dir, args.skip_run, args.threshold)


if __name__ == "__main__":
    main()
