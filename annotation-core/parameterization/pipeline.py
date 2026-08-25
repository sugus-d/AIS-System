"""End-to-end back-surface parameterisation pipeline.

1. Load & simplify mesh
2. Geodesic boundary from 10 outer landmarks
3. 2D polygon classification → cut mesh
4. Cut-mesh rim → target-polygon constraints
5. Harmonic parameterization (landmarks + rim)
6. Curvature computation + 8-panel visualisation
7. Save mesh_cut.ply + uv_coords.npy
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d
from scipy.interpolate import griddata

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mesh.curvature import calculate_curvature
from parameterization.geodesic_cut import classify_and_cut, geodesic_boundary, mesh_rim
from parameterization.harmonic import harmonic_parameterize
from parameterization.landmark_io import find_landmark_vertices, parse_landmarks_json
from parameterization.procrustes import compute_procrustes
from parameterization.template import TEMPLATE_LANDMARKS
from utils.logger import logger

_OUTER_NAMES = [
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


def _map_rim_to_polygon(
    rim_v: list[int],
    Vca: np.ndarray,
    target_poly: np.ndarray,
    lm_set: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """将边界环顶点最近点投影到目标多边形上。

    对每个非地标的边界顶点，在多边形所有边上寻找最近投影点，
    作为调和映射的额外约束。

    Args:
        rim_v:       边界环顶点索引列表。
        Vca:         Procrustes 对齐后的顶点坐标。
        target_poly: 目标多边形顶点（按顺时针顺序）。
        lm_set:      地标顶点索引集合（跳过这些顶点）。

    Returns:
        (bk, by):
            bk: (K,) 边界约束顶点索引。
            by: (K, 2) 边界约束的目标 UV 坐标。
    """
    bk, by = [], []
    for bv in rim_v:
        if bv in lm_set:
            continue
        best_d, best_pt = float("inf"), None
        for i in range(len(target_poly)):
            p1, p2 = target_poly[i], target_poly[(i + 1) % len(target_poly)]
            v = p2 - p1
            w = Vca[bv, :2] - p1
            t = np.clip(np.dot(w, v) / max(np.dot(v, v), 1e-10), 0, 1)
            proj = p1 + t * v
            d = np.linalg.norm(Vca[bv, :2] - proj)
            if d < best_d:
                best_d, best_pt = d, proj
        if best_pt is not None:
            bk.append(bv)
            by.append(best_pt)
    return np.array(bk, dtype=np.int64), np.array(by, dtype=np.float64)


def _flip_count(uv: np.ndarray, F: np.ndarray) -> int:
    """计算 UV 参数化中翻转三角面的数量。

    翻转指三角面在 UV 空间中方向相反（面积为负），
    通过叉积符号判断。

    Args:
        uv: (N, 2) UV 坐标。
        F:  (P, 3) 三角面。

    Returns:
        翻转三角面的数量。
    """
    vt = uv[F.astype(np.int64)]
    cross = (vt[:, 1, 0] - vt[:, 0, 0]) * (vt[:, 2, 1] - vt[:, 0, 1]) - (vt[:, 2, 0] - vt[:, 0, 0]) * (
        vt[:, 1, 1] - vt[:, 0, 1]
    )
    return int(np.sum(cross < 0))


def _clim(curv: np.ndarray | None) -> float:
    """计算曲率可视化对称色彩范围。

    使用曲率绝对值的 5 倍中位数作为色限，
    确保可视化范围对异常值鲁棒。

    Args:
        curv: (N,) 曲率数组，或 None。

    Returns:
        对称色彩范围值。
    """
    if curv is None:
        return 1.0
    return max(np.median(np.abs(curv[np.isfinite(curv)])) * 5, 1e-6)


# ---------------------------------------------------------------------------


def run_pipeline(
    subject_id: str,
    output_dir: str = "results/parameterization/boundary_cut",
    smoothing: float = 6.0,
    target_vertices: int = 10000,
    method: str = "biharmonic",
    mesh_path: str | None = None,
) -> np.ndarray:
    """运行单个 subject 的完整参数化管线。

    流程：
    1. 加载并简化网格
    2. 从 10 个外部地标计算测地边界
    3. 2D 多边形内外判断 → 切割网格
    4. 切割网格边界环 → 目标多边形约束
    5. 调和参数化（地标 + 边界约束）
    6. 曲率计算 + 8 面板可视化
    7. 保存 mesh_cut.ply + uv_coords.npy

    Args:
        subject_id:      受试者 ID，如 "S0004"。
        output_dir:      输出目录。
        smoothing:       测地路径平滑 sigma。
        target_vertices: 网格简化目标顶点数。
        method:          "harmonic" 或 "biharmonic"（默认）。
        mesh_path:       mesh 文件路径，默认为 data/ground_truth/{sid}/roi.ply。

    Returns:
        uv_coords: (K, 2) 切割网格的 UV 坐标。
    """
    out = Path(output_dir) / subject_id
    out.mkdir(parents=True, exist_ok=True)
    if mesh_path is None:
        mesh_path = str(Path("data/ground_truth") / subject_id / "roi.ply")

    # ── Load & simplify ──────────────────────────────────────────────
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    V0 = np.asarray(mesh.vertices, dtype=np.float64)
    ms = mesh.simplify_quadric_decimation(target_vertices)
    V = np.asarray(ms.vertices, dtype=np.float64)
    F = np.asarray(ms.triangles, dtype=np.int64)

    lm = parse_landmarks_json(f"results/ground-truth/{subject_id}/ground_truth.json")
    k_orig, y = find_landmark_vertices(mesh, lm, TEMPLATE_LANDMARKS)

    # Verify all landmarks required for boundary cutting are present
    missing_outer = [n for n in _OUTER_NAMES if n not in lm]
    if missing_outer:
        raise ValueError(
            f"Missing {len(missing_outer)} outer landmarks: {missing_outer}. "
            f"Subject {subject_id} likely has outdated ground_truth.json."
        )

    k = np.array([np.argmin(np.linalg.norm(V - V0[ki], axis=1)) for ki in k_orig])

    # Remove unreferenced vertices (pygeodesic requirement)
    ref = np.zeros(len(V), dtype=bool)
    for a, b, c in F:
        ref[a] = ref[b] = ref[c] = True
    if not np.all(ref):
        o2n = np.full(len(V), -1, dtype=np.int64)
        keep = np.where(ref)[0]
        o2n[keep] = np.arange(len(keep))
        V = V[keep]
        F = o2n[F.ravel()].reshape(-1, 3)
        k = o2n[k]

    logger.info(f"Loaded {subject_id}: {len(V)}v, {len(F)}f")

    # ── Geodesic boundary ────────────────────────────────────────────
    bverts, Va = geodesic_boundary(V, F, k, y, _OUTER_NAMES, smoothing)
    logger.info(f"Geodesic boundary: {len(bverts)} vertices")

    # ── Cut mesh ─────────────────────────────────────────────────────
    outer_idx = [list(TEMPLATE_LANDMARKS.keys()).index(n) for n in _OUTER_NAMES]
    y_outer = y[: len(TEMPLATE_LANDMARKS)]
    Vc, Fc, kc, yc, o2n_full, order_map = classify_and_cut(V, F, k, y_outer, Va, bverts)

    # ── Rim → target polygon constraints ────────────────────────────
    rim = mesh_rim(Fc)
    lm_set = set(int(ki) for ki in kc)

    src_c, tgt_c = Vc[kc, :2], yc
    s_c, R_c, t_c = compute_procrustes(src_c, tgt_c)
    Vca = Vc.copy()
    Vca[:, :2] = s_c * Vc[:, :2] @ R_c.T + t_c

    target_poly = np.array([TEMPLATE_LANDMARKS[n] for n in _OUTER_NAMES])
    bk, by = _map_rim_to_polygon(rim, Vca, target_poly, lm_set)
    k_all = np.concatenate([kc, bk])
    y_all = np.concatenate([yc, by])
    logger.info(f"Constraints: {len(kc)} LM + {len(bk)} rim = {len(k_all)}")

    # ── Harmonic parameterization ────────────────────────────────────
    cm = o3d.geometry.TriangleMesh()
    cm.vertices = o3d.utility.Vector3dVector(Vc)
    cm.triangles = o3d.utility.Vector3iVector(Fc)
    _, uv = harmonic_parameterize(cm, k_all, y_all, method=method)

    nflip = _flip_count(uv, Fc)
    err = np.max(np.abs(uv[kc] - yc))
    logger.info(f"UV: [{uv[:, 0].min():.2f},{uv[:, 0].max():.2f}]×[{uv[:, 1].min():.2f},{uv[:, 1].max():.2f}]")
    logger.info(f"LM err={err:.2e} | Flips: {nflip}/{len(Fc)} ({100 * nflip / len(Fc):.1f}%)")

    # ── Curvature ────────────────────────────────────────────────────
    curv_gauss = calculate_curvature(cm, curv_type="gaussian")
    curv_mean = calculate_curvature(cm, curv_type="mean")
    orig_mesh = o3d.geometry.TriangleMesh()
    orig_mesh.vertices = o3d.utility.Vector3dVector(V)
    orig_mesh.triangles = o3d.utility.Vector3iVector(F)
    curv_gauss_orig = calculate_curvature(orig_mesh, curv_type="gaussian")
    curv_mean_orig = calculate_curvature(orig_mesh, curv_type="mean")
    for c in [curv_gauss, curv_mean, curv_gauss_orig, curv_mean_orig]:
        if c is not None:
            lo, hi = np.percentile(c[np.isfinite(c)], [1, 99])
            np.clip(c, lo, hi, out=c)

    # ── Save ─────────────────────────────────────────────────────────
    o3d.io.write_triangle_mesh(str(out / "mesh_cut.ply"), cm)
    np.save(out / "uv_coords.npy", uv)

    # ── Visualize ────────────────────────────────────────────────────
    names = list(TEMPLATE_LANDMARKS.keys())
    fig, axes = plt.subplots(2, 4, figsize=(28, 14))

    # (0,0) — cut overview
    ax = axes[0, 0]
    ax.scatter(Va[:, 0], Va[:, 1], c="gray", s=1, alpha=0.3)
    iv = np.zeros(len(V), dtype=bool)
    iv[o2n_full >= 0] = True
    ax.scatter(Va[iv, 0], Va[iv, 1], c=V[iv, 2], s=2, cmap="viridis", alpha=0.5)
    ax.plot(Va[bverts, 0], Va[bverts, 1], "r-", lw=2.5)
    for i, n in enumerate(_OUTER_NAMES):
        vi_ = int(k[outer_idx[i]])
        ax.scatter(Va[vi_, 0], Va[vi_, 1], c="red", s=100, edgecolors="white", lw=1.5, zorder=4)
        lbl = (
            n.replace("shoulder_transition", "ST")
            .replace("neck_root", "NR")
            .replace("spine_", "P")
            .replace("axilla", "AX")
            .replace("waist", "WA")
            .replace("_", "")
        )
        ax.text(Va[vi_, 0] + 0.15, Va[vi_, 1], lbl, fontsize=8, c="red", fontweight="bold")
    for n in ["scapular_peaks_L", "scapular_peaks_R", "spine_P1", "spine_P2"]:
        i = names.index(n)
        vi_ = int(k[i])
        ax.scatter(Va[vi_, 0], Va[vi_, 1], c="cyan", s=60, marker="D", edgecolors="white", zorder=4)
        ax.text(
            Va[vi_, 0] + 0.15,
            Va[vi_, 1],
            n.replace("scapular_peaks", "SP").replace("spine_", "P").replace("_", ""),
            fontsize=7,
            c="cyan",
        )
    ax.set_aspect("equal")
    ax.set_title(f"Cut: {len(Vc)}v/{len(V)}v", fontsize=13)

    # (0,1) — harmonic UV
    ax = axes[0, 1]
    ax.tripcolor(uv[:, 0], uv[:, 1], Fc, Vc[:, 2], cmap="viridis", shading="gouraud")
    for i, (n, (u, v)) in enumerate(TEMPLATE_LANDMARKS.items()):
        if i < len(kc):
            ax.scatter(u, v, c="red", s=40, edgecolors="white", zorder=5)
            ax.text(u, v, n.split("_")[0][:3], fontsize=6)
    ax.set_title(f"Harmonic+rim ({nflip}/{len(Fc)}={100 * nflip / len(Fc):.1f}%)", fontsize=13)
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-4.2, 2.2)
    ax.set_aspect("equal")

    # (0,2) — height map
    ax = axes[0, 2]
    xi = np.linspace(-2.5, 2.5, 256)
    yi = np.linspace(-4, 2, 256)
    Zg = griddata(uv, Vc[:, 2], (*np.meshgrid(xi, yi),), method="linear", fill_value=np.nan)
    im = ax.imshow(Zg, extent=[-2.5, 2.5, -4, 2], cmap="plasma", origin="lower", aspect="equal")
    for _, (_, (u, v)) in enumerate(TEMPLATE_LANDMARKS.items()):
        ax.scatter(u, v, c="white", s=30, edgecolors="k", zorder=5)
    ax.set_title("Height Map", fontsize=13)
    plt.colorbar(im, ax=ax, shrink=0.6)

    axes[0, 3].axis("off")

    # Row 1 — curvature
    cg, cm_ = _clim(curv_gauss_orig), _clim(curv_mean_orig)
    cgu, cmu = _clim(curv_gauss), _clim(curv_mean)
    for col, data, title, lim3, lim_uv in [
        (0, curv_gauss_orig, "3D Gauss", cg, None),
        (1, curv_gauss, "UV Gauss", None, cgu),
        (2, curv_mean_orig, "3D Mean", cm_, None),
        (3, curv_mean, "UV Mean", None, cmu),
    ]:
        ax = axes[1, col]
        if data is not None:
            vlim = lim3 if lim3 is not None else lim_uv
            pts = Va[:, :2] if col in (0, 2) else uv
            tri = F if col in (0, 2) else Fc
            ax.tripcolor(pts[:, 0], pts[:, 1], tri, data, cmap="jet", shading="gouraud", vmin=-vlim, vmax=vlim)
            if col in (1, 3):
                for i, (_, (u_, v_)) in enumerate(TEMPLATE_LANDMARKS.items()):
                    if i < len(kc):
                        ax.scatter(u_, v_, c="black", s=20, zorder=5)
            ax.set_title(f"{title} (±{vlim:.4f})", fontsize=13)
        ax.set_aspect("equal")
        if col in (1, 3):
            ax.set_xlim(-2.8, 2.8)
            ax.set_ylim(-4.2, 2.2)

    plt.tight_layout()
    plt.savefig(out / "parameterization_result.png", dpi=150, bbox_inches="tight")
    logger.info(f"Saved to {out}/")

    return uv


def main_cli() -> None:
    """命令行入口：测地切割 + 调和参数化。"""
    parser = argparse.ArgumentParser(description="Geodesic cut + harmonic parameterization")
    parser.add_argument("subject", help="Subject ID (e.g. S0004)")
    parser.add_argument("--smoothing", type=float, default=6.0)
    parser.add_argument("--output", type=str, default="results/parameterization/boundary_cut")
    args = parser.parse_args()
    run_pipeline(args.subject, args.output, args.smoothing)


if __name__ == "__main__":
    main_cli()
