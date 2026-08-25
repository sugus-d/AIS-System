"""编排层：参数化测量图渲染。

基于 UV 参数化结果与模板 landmark（TEMPLATE_LANDMARKS）生成测量可视化，
复用 plot_shared._save_figure 保存。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from commands.plot_shared import _save_figure
from parameterization.template import TEMPLATE_LANDMARKS  # noqa: E402

# subplot 网格中的列索引：3D 列与 UV 列
_3D_COLUMN = 2
_UV_COLUMN = 3

_LR_PAIRS = [
    ("neck_root_L", "neck_root_R"),
    ("shoulder_transition_L", "shoulder_transition_R"),
    ("scapular_peaks_L", "scapular_peaks_R"),
    ("axilla_L", "axilla_R"),
    ("waist_L", "waist_R"),
    ("waist_lower_L", "waist_lower_R"),
]
_SPINE_CHAIN = ["neck_root_spine_point", "scapular_spine_point", "axilla_spine_point", "thoracic_spine_point", "waist_spine_point", "waist_lower_spine_point"]


def render_measures(subject: str, output_dir: str, uv_path: str, mesh_path: str) -> None:
    """渲染综合表面指标可视化（3×4 subplot），由 plot_parameterization 调用。

    由 render_parameterization 的"综合可视化"块拆出：
    加载 UV / mesh_cut，计算 5 个表面指标（高度、平均/高斯曲率、粗糙度、法向角），
    绘制 3D/UV 并排的 3×4 子图。
    """
    from scipy.interpolate import LinearNDInterpolator

    from mesh.curvature import calculate_curvature
    from mesh.roi.bfs import compute_mesh_roughness

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

    # 色彩范围
    def _clim(arr: np.ndarray) -> float:
        return max(np.median(np.abs(arr[np.isfinite(arr)])) * 5, 1e-6)

    # 3×4 布局
    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    fig.suptitle(f"Surface measures — {subject}", fontsize=14, y=0.98)

    # ═══ Row 0: Landmarks (gray) + Height ═══

    # 预构建 UV→3D 插值器（三角网格内线性插值，比 argmin 平滑）
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
    _save_figure(fig, output_dir / "boundary_cut" / subject / "all_measures.png")
