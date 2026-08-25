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

matplotlib.use("Agg")

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
    "waist_lower_spine_point",
    "waist_lower_R",
    "waist_R",
    "axilla_R",
    "shoulder_transition_R",
    "neck_root_R",
    "neck_root_spine_point",
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


# ---------------------------------------------------------------------------


def run_pipeline(
    subject_id: str,
    output_dir: str = "results/parameterization/boundary_cut",
    smoothing: float = 6.0,
    target_vertices: int = 10000,
    method: str = "biharmonic",
    mesh_path: str | None = None,
    landmarks_path: str | None = None,
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
        landmarks_path:  landmarks JSON 路径，默认为
                         results/ground-truth/{sid}/ground_truth.json
                         （predict 可传入预测用 landmarks，避免污染 GT 目录）。

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

    landmarks_json = landmarks_path or f"results/ground-truth/{subject_id}/ground_truth.json"
    lm = parse_landmarks_json(landmarks_json)
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
