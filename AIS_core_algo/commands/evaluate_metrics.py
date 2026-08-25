#!/usr/bin/env python3
"""ROI 评估指标 — Chamfer 距离、多维度达标指标、区域评估、报告表格。

由 commands/evaluate_roi 拆出：指标计算相关函数与验收阈值。
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# ── roi 评估 ──
DATA_DIR = "data/ground_truth"
# 受检者 ID 已匿名化（S#### 示例）；本地运行时用 --subjects 传入真实受检者编号
ROI_SUBJECTS = [
    "S0015",
    "S0071",
    "S0038",
    "S0092",
    "S0007",
    "S0023",
    "S0048",
    "S0018",
    "S0016",
    "S0074",
]

# ── 验收阈值（参照 .claude/goals/roi-optimization-v2.md）──
THRESHOLDS = {
    "v_ratio": 0.50,  # 顶点数偏差率 < 50%
    "x_min_dev": 40.0,  # mm < 40
    "x_max_dev": 40.0,  # mm < 40
    "y_min_dev": 35.0,  # mm < 35
    "y_max_dev": 60.0,  # mm < 60（颈部可放宽）
    "chamfer": 15.0,  # mm < 15
}
# Chamfer 计算要求点集最少点数
_MIN_CHAMFER_POINTS = 3
# 指标总数（6 个验收维度）
_TOTAL_METRIC_COUNT = 6


def _o3d_to_trimesh(o3d_mesh) -> object:
    """将 open3d mesh 转 trimesh。"""
    import trimesh

    v = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    f = np.asarray(o3d_mesh.triangles)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def _get_boundary_points(mesh) -> np.ndarray | None:
    """提取 mesh 的所有外边界点。"""
    import trimesh

    t = mesh if isinstance(mesh, trimesh.Trimesh) else _o3d_to_trimesh(mesh)

    if t.is_empty or len(t.faces) == 0:
        return None

    # 找所有边界边（只属于1个三角面的边）
    edge_count: dict[tuple[int, int], int] = {}
    for face in t.faces:
        for i in range(3):
            edge = tuple(sorted([int(face[i]), int(face[(i + 1) % 3])]))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary_verts = set()
    for edge, count in edge_count.items():
        if count == 1:
            boundary_verts.add(edge[0])
            boundary_verts.add(edge[1])

    if not boundary_verts:
        return None
    return t.vertices[list(boundary_verts)]


def _compute_chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """计算两个点集之间的双向 Chamfer distance（mm）。"""
    if pts_a is None or pts_b is None or len(pts_a) < _MIN_CHAMFER_POINTS or len(pts_b) < _MIN_CHAMFER_POINTS:
        return float("inf")

    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)

    dist_a_to_b = tree_b.query(pts_a)[0].mean()
    dist_b_to_a = tree_a.query(pts_b)[0].mean()

    return float((dist_a_to_b + dist_b_to_a) / 2)


def compute_metrics(algo_mesh, gt_mesh) -> dict:
    """计算算法输出与 GT 之间的全维度指标（带符号）。"""
    algo_v = np.asarray(algo_mesh.vertices, dtype=np.float64)
    gt_v = np.asarray(gt_mesh.vertices, dtype=np.float64)

    # 顶点数偏差
    n_v_algo = len(algo_v)
    n_v_gt = len(gt_v)
    v_ratio = abs(n_v_algo - n_v_gt) / max(n_v_gt, 1)

    # X/Y/Z 范围（带符号：正 = algo 更大/更高，负 = algo 更小/更低）
    def _range_metrics(a, g, label) -> dict[str, float]:
        return {
            f"{label}_min_algo": float(a.min()),
            f"{label}_max_algo": float(a.max()),
            f"{label}_min_gt": float(g.min()),
            f"{label}_max_gt": float(g.max()),
            f"{label}_min_dev": float(a.min() - g.min()),  # signed
            f"{label}_max_dev": float(a.max() - g.max()),  # signed
            f"{label}_span_algo": float(a.ptp()),
            f"{label}_span_gt": float(g.ptp()),
        }

    result = {
        "n_vertices_algo": n_v_algo,
        "n_vertices_gt": n_v_gt,
        "v_ratio": v_ratio,
        **_range_metrics(algo_v[:, 0], gt_v[:, 0], "x"),
        **_range_metrics(algo_v[:, 1], gt_v[:, 1], "y"),
        **_range_metrics(algo_v[:, 2], gt_v[:, 2], "z"),
    }

    # 外轮廓 Chamfer distance
    algo_boundary = _get_boundary_points(algo_mesh)
    gt_boundary = _get_boundary_points(gt_mesh)
    result["chamfer"] = _compute_chamfer_distance(algo_boundary, gt_boundary)
    result["chamfer_valid"] = algo_boundary is not None and gt_boundary is not None

    # 达标判断
    passes = []
    if v_ratio < THRESHOLDS["v_ratio"]:
        passes.append("v_ratio")
    if abs(result["x_min_dev"]) < THRESHOLDS["x_min_dev"]:
        passes.append("x_min")
    if abs(result["x_max_dev"]) < THRESHOLDS["x_max_dev"]:
        passes.append("x_max")
    if abs(result["y_min_dev"]) < THRESHOLDS["y_min_dev"]:
        passes.append("y_min")
    if abs(result["y_max_dev"]) < THRESHOLDS["y_max_dev"]:
        passes.append("y_max")
    if result["chamfer"] < THRESHOLDS["chamfer"]:
        passes.append("chamfer")

    result["n_pass"] = len(passes)
    result["n_total"] = _TOTAL_METRIC_COUNT
    result["pass"] = result["n_pass"] == _TOTAL_METRIC_COUNT

    return result


def run_region_eval(algo_mesh, gt_mesh, splits: dict | None = None, landmarks: dict | None = None) -> list[dict]:
    """对算法输出 mesh 与 GT mesh 执行区域化三角面评估。"""
    from mesh.roi.region_eval import compute_region_deltri

    algo_v = np.asarray(algo_mesh.vertices)
    algo_t = np.asarray(algo_mesh.triangles)
    gt_v = np.asarray(gt_mesh.vertices)
    gt_t = np.asarray(gt_mesh.triangles)
    return compute_region_deltri(algo_v, algo_t, gt_v, gt_t, splits=splits, landmarks=landmarks)


def report_table(results: dict[str, dict]) -> str:
    """生成指标报告表格。"""
    hdr = (
        f"{'Subject':<12} {'v':>6} {'v_gt':>6} {'xmin':>6} {'xmax':>6} {'ymin':>6} {'ymax':>6} {'chamf':>7} {'PASS':>6}"
    )
    sep = "-" * (len(hdr) + 10)
    lines = [hdr, sep]

    n_pass_all = 0
    for sid in ROI_SUBJECTS:
        if sid not in results:
            continue
        r = results[sid]
        passes = all(
            [
                r["v_ratio"] < THRESHOLDS["v_ratio"],
                r["x_min_dev"] < THRESHOLDS["x_min_dev"],
                r["x_max_dev"] < THRESHOLDS["x_max_dev"],
                r["y_min_dev"] < THRESHOLDS["y_min_dev"],
                r["y_max_dev"] < THRESHOLDS["y_max_dev"],
                r["chamfer"] < THRESHOLDS["chamfer"],
            ]
        )
        if passes:
            n_pass_all += 1

        line = (
            f"{sid:<12} {r['n_vertices_algo']:>6} {r['n_vertices_gt']:>6} "
            f"{r['x_min_dev']:>5.0f}  {r['x_max_dev']:>5.0f} "
            f"{r['y_min_dev']:>5.0f}  {r['y_max_dev']:>5.0f} "
            f"{r['chamfer']:>6.1f} {'✅' if passes else '❌':>6}"
        )
        lines.append(line)

    lines.append(sep)
    lines.append(f"Pass: {n_pass_all}/{len(ROI_SUBJECTS)}")
    return "\n".join(lines)
