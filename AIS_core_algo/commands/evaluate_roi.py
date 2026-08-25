#!/usr/bin/env python3
"""ROI 评估域 — ROI 全链路验收（指标计算见 commands.evaluate_metrics）。"""

from __future__ import annotations

import json
import os

import numpy as np
import open3d as o3d

from commands.evaluate_cut import _load_gt_landmarks
from commands.evaluate_metrics import (
    _TOTAL_METRIC_COUNT,
    compute_metrics,
    DATA_DIR,
    report_table,
    ROI_SUBJECTS,
)
from utils.logger import logger
from utils.paths import EVAL_EVALUATION_DIR

ROI_OUTPUT_DIR = str(EVAL_EVALUATION_DIR)


# ══════════════════════════ roi — ROI 全链路验收 ══════════════════════════


def _run_roi_algorithm(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """运行生产 ROI 管线（commands/batch_process_all.run_roi_pipeline），返回裁剪后 o3d mesh。

    原 compare_pants_algorithms.run_new_algorithm 已在重构中被
    commands/batch_process_all.py::run_roi_pipeline 取代（experiments/commands/
    下存档版本已无该函数）。
    """
    from mesh.roi.pipeline import run_roi_pipeline

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    result_vertices, result_triangles = run_roi_pipeline(vertices, triangles)
    result = o3d.geometry.TriangleMesh()
    result.vertices = o3d.utility.Vector3dVector(result_vertices)
    result.triangles = o3d.utility.Vector3iVector(result_triangles)
    return result


def _load_mesh(path: str) -> o3d.geometry.TriangleMesh | None:
    """加载 mesh，支持 open3d 和 trimesh 格式。"""
    if not os.path.isfile(path):
        return None
    try:
        return o3d.io.read_triangle_mesh(path)
    except Exception:
        return None


def run_pipeline(subject_id: str) -> str:
    """运行 ROI 全链路，返回输出 mesh 路径。"""
    orig_path = f"{DATA_DIR}/{subject_id}/original.ply"
    mesh = o3d.io.read_triangle_mesh(orig_path)

    # 兜底：如果 BFS 或曲线失败，这里不要回退到旧算法
    # 让新算法自己处理，否则评估的是旧算法
    try:
        result = _run_roi_algorithm(mesh)
    except Exception:
        result = o3d.geometry.TriangleMesh()

    os.makedirs(f"{ROI_OUTPUT_DIR}/meshes", exist_ok=True)
    out_path = f"{ROI_OUTPUT_DIR}/meshes/{subject_id}_algo.ply"
    o3d.io.write_triangle_mesh(out_path, result)
    return out_path


def evaluate_roi(
    subjects: list[str] | None = None,
    use_old: bool = False,
    skip_run: bool = False,
    output_dir: str = ROI_OUTPUT_DIR,
    regions: bool = False,
    regions_only: bool = False,
    stats: bool = False,
) -> None:
    """ROI 全链路验收 — 多维度指标对比算法输出与 GT。"""
    os.makedirs(output_dir, exist_ok=True)

    if stats:
        import glob as _glob

        from mesh.roi.region_eval import compute_region_deltri, compute_thresholds

        gt_paths = sorted(_glob.glob(f"{DATA_DIR}/*/roi.ply"))
        logger.info(f"Stats mode: {len(gt_paths)} GT subjects found")

        all_results: list[list[dict]] = []
        total = len(gt_paths)
        for idx, gt_path in enumerate(gt_paths, start=1):
            sid = gt_path.split("/")[-2]
            try:
                gt_mesh = o3d.io.read_triangle_mesh(gt_path)
                orig_mesh = o3d.io.read_triangle_mesh(f"{DATA_DIR}/{sid}/original.ply")

                algo_mesh = _run_roi_algorithm(orig_mesh)

                algo_v = np.asarray(algo_mesh.vertices)
                algo_t = np.asarray(algo_mesh.triangles)
                gt_v = np.asarray(gt_mesh.vertices)
                gt_t = np.asarray(gt_mesh.triangles)

                results = compute_region_deltri(algo_v, algo_t, gt_v, gt_t)
                all_results.append(results)
                logger.info(f"  [{idx}/{total}] {sid}: OK")
            except Exception as exc:
                logger.warning(f"  [{idx}/{total}] {sid}: FAILED - {exc}")

        output_path = os.path.join(output_dir, "region_thresholds.json")
        thresholds = compute_thresholds(all_results, output_path)
        logger.info(f"Thresholds saved: {output_path}")

        for region, threshold in thresholds.items():
            logger.info(
                f"  {region}: p50={threshold['p50']:.1f}%, "
                f"p90={threshold['p90']:.1f}%, p95={threshold['p95']:.1f}% (n={threshold['n']})"
            )
        return

    if subjects is None:
        subjects = ROI_SUBJECTS

    results = {}

    for sid in subjects:
        gt_path = f"{DATA_DIR}/{sid}/roi.ply"
        if not os.path.isfile(gt_path):
            logger.warning(f"GT not found: {gt_path}, skipping")
            continue

        gt_mesh = o3d.io.read_triangle_mesh(gt_path)

        if use_old:
            # 用旧算法（extract_back_roi）
            from mesh.roi_extract import extract_back_roi

            orig_path = f"{DATA_DIR}/{sid}/original.ply"
            orig = o3d.io.read_triangle_mesh(orig_path)
            algo_mesh = extract_back_roi(orig)
        elif skip_run:
            # 直接从已有 mesh 文件加载
            algo_path = f"{output_dir}/meshes/{sid}_algo.ply"
            if os.path.isfile(algo_path):
                algo_mesh = o3d.io.read_triangle_mesh(algo_path)
            else:
                logger.warning(f"No cached mesh for {sid}, running pipeline")
                algo_path = run_pipeline(sid)
                algo_mesh = o3d.io.read_triangle_mesh(algo_path)
        else:
            algo_path = run_pipeline(sid)
            algo_mesh = o3d.io.read_triangle_mesh(algo_path)

        if algo_mesh.is_empty() or not algo_mesh.has_triangles():
            logger.warning(f"{sid}: algo produced empty mesh")
            results[sid] = {
                "n_vertices_algo": 0,
                "n_vertices_gt": len(np.asarray(gt_mesh.vertices)),
                "v_ratio": 1.0,
                "x_min_dev": 999,
                "x_max_dev": 999,
                "y_min_dev": 999,
                "y_max_dev": 999,
                "chamfer": 999,
                "chamfer_valid": False,
                "n_pass": 0,
                "n_total": _TOTAL_METRIC_COUNT,
                "pass": False,
            }
            continue

        r = {} if regions_only else compute_metrics(algo_mesh, gt_mesh)

        if regions or regions_only:
            from mesh.roi.region_eval import run_region_eval

            lm = _load_gt_landmarks(sid)
            r["regions"] = run_region_eval(algo_mesh, gt_mesh, landmarks=lm)

        results[sid] = r

        if regions_only:
            logger.info(f"{sid}: regions evaluated")
        else:
            logger.info(
                f"{sid}: v_ratio={r['v_ratio']:.2%} "
                f"xmin={r['x_min_dev']:.0f} xmax={r['x_max_dev']:.0f} "
                f"ymin={r['y_min_dev']:.0f} ymax={r['y_max_dev']:.0f} "
                f"chamfer={r['chamfer']:.1f} "
                f"{'✅' if r['pass'] else '❌'}"
            )

    # 输出主报告（regions-only 时跳过）
    if not regions_only:
        table = report_table(results)
        logger.info(f"\n{table}")

        report_path = os.path.join(output_dir, "report.txt")
        with open(report_path, "w") as f:
            f.write(table)
        logger.info(f"Report saved: {report_path}")

    # 区域评估结果报告
    if regions or regions_only:
        from mesh.roi.region_eval import region_report_text

        for sid in subjects:
            if sid in results and "regions" in results[sid]:
                logger.info("")
                logger.info(region_report_text(results[sid]["regions"], sid))

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    logger.info(f"Results saved: {json_path}")
