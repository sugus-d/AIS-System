#!/usr/bin/env python3
"""统一评估入口 — 按 --domain 选择评估类型。

将原 commands/evaluate_back_v1.py / evaluate_batch.py / evaluate_cut.py /
evaluate_roi.py 四个脚本合并为单入口，各 domain 逻辑保留为模块内函数。

用法:
  uv run python -m commands.evaluate --domain back [--recompute]    # 特征方案评估
  uv run python -m commands.evaluate --domain batch                  # 批量参数组合评估
  uv run python -m commands.evaluate --domain cut [--subjects ...]   # 曲线切割评估
  uv run python -m commands.evaluate --domain roi [--subjects | --old | --skip-run | --regions | --regions-only | --stats | --output ...]  # ROI 全链路验收
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    import trimesh


from modeling.metrics import compute_4class_metrics, SEVERITY_BINS
from modeling.training.result_paths import RESULTS_DIR
from utils.logger import logger

# ── cut / batch 共享 ──
DATA_DIR = Path("data/ground_truth")
OUTPUT_DIR = Path("results/eval/cut_eval")
# mesh 顶点数下限，小于该值视为无效
_MIN_MESH_VERTS = 50
# 曲面点匹配判定距离（mm），小于该距离视为命中
_MATCH_DISTANCE_MM = 5.0
# 覆盖率达标阈值
_COVERAGE_TARGET = 0.95
# 多余率达标阈值
_EXCESS_TARGET = 0.08

# ── back 评估 ──
RESULTS_FILE = RESULTS_DIR / "all_results_back_v1.json"
TRUE_LABELS = Path("results/extraction/features_extraction/back_v1/basic.csv")
BACK_OUTPUT_DIR = RESULTS_DIR / "back_v1"
SUMMARY_CSV = BACK_OUTPUT_DIR / "summary.csv"
SUMMARY_JSON = BACK_OUTPUT_DIR / "summary.json"

# ── roi 评估 ──
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
ROI_OUTPUT_DIR = "results/eval/evaluation"

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


# ══════════════════════════ cut / batch 共享 ══════════════════════════


def get_valid_subjects() -> list[str]:
    """获取 data/ground_truth 下同时具备 original.ply / roi.ply + ground_truth.json 的 subject。"""
    valid = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "original.ply").exists() or not (d / "roi.ply").exists():
            continue
        if not (Path("results/ground-truth") / d.name / "ground_truth.json").exists():
            continue
        valid.append(d.name)
    return valid


def _load_gt_landmarks(subject_id: str) -> dict[str, float]:
    """从 ground_truth.json 读 landmark，展开为 region_eval 的 flat dict（{name}_{side}_{axis}）。"""
    from parameterization.landmark_io import parse_landmarks_json

    gt_path = Path("results/ground-truth") / subject_id / "ground_truth.json"
    if not gt_path.exists():
        return {}
    lm = parse_landmarks_json(gt_path)
    out: dict[str, float] = {}
    for short, vec in lm.items():
        for i, axis in enumerate("xyz"):
            out[f"{short}_{axis}"] = float(vec[i])
    return out


def _neck_mask(
    algo_v: np.ndarray,
    algo_t: np.ndarray,
    y_top_fraction: float = 0.15,
    x_center_fraction: float = 0.30,
) -> np.ndarray:
    """脖子区域的布尔 mask（True = 脖子三角面）。"""
    centers = algo_v[algo_t].mean(axis=1)
    y_max = float(algo_v[:, 1].max())
    y_min = float(algo_v[:, 1].min())
    y_thr = y_max - (y_max - y_min) * y_top_fraction

    x_center = float(algo_v[:, 0].mean())
    x_span = float(algo_v[:, 0].ptp())
    x_half = x_span * x_center_fraction / 2

    return (centers[:, 1] >= y_thr) & (np.abs(centers[:, 0] - x_center) <= x_half)


def compute_cut_metrics(
    algo_v: np.ndarray,
    algo_t: np.ndarray,
    gt_v: np.ndarray,
    gt_t: np.ndarray,
    exclude_neck: bool = True,
) -> dict:
    """覆盖率/多余率/Chamfer，可选排除脖子区域。"""
    if len(algo_v) < _MIN_MESH_VERTS or len(gt_v) < _MIN_MESH_VERTS:
        return {"error": "mesh too small", "coverage": 0.0, "excess": 0.0}

    ac = algo_v[algo_t].mean(axis=1) if len(algo_t) > 0 else algo_v
    gc = gt_v[gt_t].mean(axis=1) if len(gt_t) > 0 else gt_v
    if len(ac) == 0 or len(gc) == 0:
        return {"error": "no centroids", "coverage": 0.0, "excess": 0.0}

    # 脖子 mask（排除）
    neck_mask = _neck_mask(algo_v, algo_t) if exclude_neck else None

    # 只取非脖子的三角面
    if neck_mask is not None and neck_mask.any():
        keep = ~neck_mask
        ac = ac[keep]

    gt_tree = cKDTree(gc)
    algo_tree = cKDTree(ac)

    bfs_to_gt = gt_tree.query(ac)[0]  # 算法 → GT
    gt_to_bfs = algo_tree.query(gc)[0]  # GT → 算法

    coverage = float((gt_to_bfs < _MATCH_DISTANCE_MM).mean())
    excess = float((bfs_to_gt >= _MATCH_DISTANCE_MM).mean())
    chamfer = float((bfs_to_gt.mean() + gt_to_bfs.mean()) / 2)

    return {
        "coverage": round(coverage, 4),
        "excess": round(excess, 4),
        "chamfer": round(chamfer, 2),
        "algo_v": len(algo_v),
        "algo_t": len(algo_t),
        "gt_v": len(gt_v),
        "gt_t": len(gt_t),
    }


# ══════════════════════════ cut — 曲线切割评估 ══════════════════════════


def evaluate_cut_one(subject: str) -> dict | None:
    """跑完整的 BFS + 曲线切割 + GT 比对。"""
    try:
        orig = o3d.io.read_triangle_mesh(str(DATA_DIR / subject / "original.ply"))
        gt = o3d.io.read_triangle_mesh(str(DATA_DIR / subject / "roi.ply"))
    except Exception:
        return None

    gt_v = np.asarray(gt.vertices, dtype=np.float64)
    gt_t = np.asarray(gt.triangles)
    if len(gt_v) < _MIN_MESH_VERTS:
        return None

    try:
        from commands.batch_process_all import run_roi_pipeline

        vertices = np.asarray(orig.vertices, dtype=np.float64)
        triangles = np.asarray(orig.triangles)
        cut_v, cut_t = run_roi_pipeline(vertices, triangles)
    except Exception as e:
        return {"error": str(e)[:100], "coverage": 0.0, "excess": 0.0}

    result = compute_cut_metrics(cut_v, cut_t, gt_v, gt_t, exclude_neck=True)
    result["subject"] = subject
    return result


def evaluate_cut(subjects_arg: str | None = None) -> None:
    """曲线切割评估 — 全量 GT 验证，可选 --subjects 逗号分隔指定子集。"""
    subjects = subjects_arg.split(",") if subjects_arg else get_valid_subjects()
    logger.info(f"Subjects: {len(subjects)}")

    results = []
    t0 = time.time()
    for sid in subjects:
        r = evaluate_cut_one(sid)
        if r is None:
            logger.warning(f"  {sid}: skip")
            continue
        results.append(r)
        err = r.get("error", "")
        log = f"  {sid}: cov={r.get('coverage', 0):.4f} exc={r.get('excess', 0):.4f}"
        if err:
            log += f" ERR={err}"
        logger.info(log)

    elapsed = time.time() - t0
    ok = [r for r in results if "error" not in r or not r["error"]]
    failed = [r for r in results if r.get("error")]

    print("\n" + "=" * 80)
    print("曲线切割评估结果 (pipeline: BFS→erosion→pants_cut)")
    print(f"耗时: {elapsed:.0f}s  subject: {len(ok)} OK / {len(failed)} FAIL")
    print("=" * 80)

    if ok:
        covs = [r["coverage"] for r in ok]
        excs = [r["excess"] for r in ok]
        chs = [r.get("chamfer", 0) for r in ok if "chamfer" in r]
        print(f"覆盖率:  均值={np.mean(covs):.4f}  中位数={np.median(covs):.4f}")
        print(f"多余率:  均值={np.mean(excs):.4f}  中位数={np.median(excs):.4f}")
        if chs:
            print(f"Chamfer: 均值={np.mean(chs):.2f}mm  中位数={np.median(chs):.2f}mm")

        print("\n各 subject:")
        for r in sorted(ok, key=lambda x: -x.get("excess", 0)):
            print(f"  {r['subject']}: cov={r['coverage']:.4f}  exc={r['excess']:.4f}  v={r.get('algo_v', '?')}")

    if failed:
        print(f"\n失败 ({len(failed)}):")
        for r in failed:
            print(f"  {r['subject']}: {r.get('error', '?')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "eval_pipeline.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {path}")


# ══════════════════════════ batch — 批量参数组合评估 ══════════════════════════


def run_and_eval(sid: str, erosion: int, pants_ratio: float) -> dict | None:
    """Run pipeline with given params and return metrics."""
    try:
        orig = o3d.io.read_triangle_mesh(str(DATA_DIR / sid / "original.ply"))
        gt = o3d.io.read_triangle_mesh(str(DATA_DIR / sid / "roi.ply"))
    except Exception:
        return None

    gt_v = np.asarray(gt.vertices, dtype=np.float64)
    gt_t = np.asarray(gt.triangles)
    if len(gt_v) < _MIN_MESH_VERTS:
        return None

    try:
        from mesh.roi._bfs_impl import largest_component
        from mesh.roi._mesh_erosion import strip_boundary_tris
        from mesh.roi._pants_cut import remove_pants
        from mesh.roi.bfs import mesh_bfs

        vertices = np.asarray(orig.vertices, dtype=np.float64)
        triangles = np.asarray(orig.triangles)

        # BFS
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        bf = mesh_bfs(
            mesh,
            roughness_threshold=0.20,
            angle_threshold_deg=45,
            fill_holes=True,
            max_hole_boundary=200,
            max_hole_area=5000,
        )
        bf = largest_component(bf)
        v = np.asarray(bf.vertices, dtype=np.float64)
        t = np.asarray(bf.triangles, dtype=np.int32)

        # Erosion
        v, t = strip_boundary_tris(v, t, iterations=erosion)

        # Pants cut
        v, t = remove_pants(v, t, roughness_ratio=pants_ratio)
    except Exception as e:
        return {"error": str(e)[:100], "coverage": 0.0, "excess": 0.0}

    result = compute_cut_metrics(v, t, gt_v, gt_t, exclude_neck=True)
    result["subject"] = sid
    result["algo_v"] = len(v)
    result["algo_t"] = len(t)
    return result


def evaluate_batch() -> None:
    """批量评估不同参数组合 — BFS→erosion→pants_cut。"""
    CONFIGS = [
        {"name": "baseline", "erosion": 3, "pants_ratio": 1.8},
        {"name": "it1", "erosion": 4, "pants_ratio": 1.5},
        {"name": "it2", "erosion": 3, "pants_ratio": 1.3},
        {"name": "it3", "erosion": 5, "pants_ratio": 1.8},
        {"name": "it4", "erosion": 5, "pants_ratio": 1.5},
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = get_valid_subjects()
    logger.info(f"Subjects: {len(subjects)}")

    for cfg in CONFIGS:
        logger.info("=" * 60)
        logger.info(f"Config: {cfg['name']}  erosion={cfg['erosion']}  pants_ratio={cfg['pants_ratio']:.1f}")
        logger.info("=" * 60)

        results = []
        t0 = time.time()
        for sid in subjects:
            r = run_and_eval(sid, cfg["erosion"], cfg["pants_ratio"])
            if r is None:
                continue
            results.append(r)
            err = r.get("error", "")
            log = f"  {sid}: cov={r.get('coverage', 0):.4f} exc={r.get('excess', 0):.4f}"
            if err:
                log += f" ERR={err}"
            logger.info(log)

        elapsed = time.time() - t0
        ok = [r for r in results if "error" not in r or not r["error"]]

        path = OUTPUT_DIR / f"eval_{cfg['name']}.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        if ok:
            covs = np.array([r["coverage"] for r in ok])
            excs = np.array([r["excess"] for r in ok])
            print(
                f"[{cfg['name']}] N={len(ok)}  cov={covs.mean():.4f}  exc={excs.mean():.4f}  "
                f"cov>95%={(covs >= _COVERAGE_TARGET).sum()}  exc<8%={(excs < _EXCESS_TARGET).sum()}  "
                f"chamfer={np.median([r.get('chamfer', 0) for r in ok]):.2f}mm  ({elapsed:.0f}s)"
            )
        logger.info(f"Saved: {path}\n")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for cfg in CONFIGS:
        path = OUTPUT_DIR / f"eval_{cfg['name']}.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        ok = [r for r in d if "error" not in r or not r.get("error", "")]
        if not ok:
            continue
        covs = np.array([r["coverage"] for r in ok])
        excs = np.array([r["excess"] for r in ok])
        p5 = np.percentile(covs, 5)
        print(
            f"  {cfg['name']:<10} erosion={cfg['erosion']} pants={cfg['pants_ratio']:.1f}  "
            f"cov={covs.mean():.4f}  exc={excs.mean():.4f}  "
            f"cov5%={p5:.4f}  exc<8%={(excs < _EXCESS_TARGET).sum()}/{len(ok)}"
        )


# ══════════════════════════ back — 特征方案评估 ══════════════════════════


def _load_true_labels() -> tuple[np.ndarray, np.ndarray]:
    """加载真实 max_cobb 值。"""
    df = pd.read_csv(TRUE_LABELS).dropna(subset=["max_cobb"])
    return df["subject_id"].values, df["max_cobb"].values.astype(float)


def evaluate_back_one(
    name: str,
    preds: np.ndarray,
    y_true: np.ndarray,
    subjects: np.ndarray,
    config: dict | None = None,
) -> dict:
    """评估单个模型，保存到 per-model 目录，返回 summary 行。"""
    out_dir = BACK_OUTPUT_DIR / name
    out_dir.mkdir(exist_ok=True)

    metrics = compute_4class_metrics(y_true, preds)

    # Per-subject predictions CSV
    y_class = np.digitize(y_true, SEVERITY_BINS[1:-1])
    p_class = np.digitize(preds, SEVERITY_BINS[1:-1])
    labels = ["Normal", "Mild", "Moderate", "Severe"]
    pdf = pd.DataFrame(
        {
            "subject_id": subjects,
            "max_cobb_true": y_true,
            "max_cobb_pred": np.round(preds, 2),
            "class_true": [labels[c] for c in y_class],
            "class_pred": [labels[c] for c in p_class],
        }
    )
    pdf.to_csv(out_dir / "predictions.csv", index=False)

    # Metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Config
    if config:
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    logger.info(
        f"  {name:<14s} | RMSE={metrics['rmse']:<6.2f} | "
        f"Macro-F1={metrics['macro_f1']:<6.4f} | "
        f"Acc={metrics['total_accuracy']:<6.4f} | "
        f"Per-class: " + " ".join(f"{k}={v['accuracy'] * 100:.0f}%" for k, v in metrics["per_class"].items())
    )

    # Summary row
    return {
        "model": name,
        "rmse": metrics["rmse"],
        "r": metrics["r"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "total_accuracy": metrics["total_accuracy"],
        "f1_20": config.get("f1_20", 0) if config else 0,
        **{f"acc_{k.lower()}": v["accuracy"] for k, v in metrics["per_class"].items()},
        **{f"f1_{k.lower()}": v["f1"] for k, v in metrics["per_class"].items()},
        **{f"support_{k.lower()}": v["support"] for k, v in metrics["per_class"].items()},
    }


def evaluate_back(recompute: bool = False) -> None:
    """Back v1 全模型评估 — 从预测结果计算 4 分类指标并保存。"""
    BACK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_FILE.exists():
        logger.error(f"结果文件不存在: {RESULTS_FILE}")
        return

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    subjects, y_true = _load_true_labels()
    logger.info(f"加载 {len(data)} 个模型, {len(y_true)} subjects")

    rows = []
    for d in data:
        name = d["algo"]
        metrics_path = BACK_OUTPUT_DIR / name / "metrics.json"

        if metrics_path.exists() and not recompute:
            logger.info(f"  {name}: 缓存命中，跳过")
            with open(metrics_path) as f:
                metrics = json.load(f)
            rows.append(
                {
                    "model": name,
                    "rmse": metrics["rmse"],
                    "r": metrics["r"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "total_accuracy": metrics["total_accuracy"],
                    "f1_20": d.get("f1", 0),
                }
            )
            continue

        preds = np.array(d["preds"])
        config = {
            "scheme": "train_back_v1",
            "f1_20": d.get("f1", 0),
            "hp_n_iter": d.get("best_params", {}).get("n_iter", 40),
        }
        row = evaluate_back_one(name, preds, y_true, subjects, config)
        rows.append(row)

    # Summary
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    df.to_csv(SUMMARY_CSV, index=False)
    df.to_json(SUMMARY_JSON, orient="records", indent=2)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Back v1 全模型评估汇总 ({len(df)} 模型)")
    logger.info(f"{'=' * 70}")
    logger.info(
        f"{'Model':<14s} {'RMSE':<7s} {'MacroF1':<9s} {'Acc':<7s} "
        f"{'Normal':<9s} {'Mild':<9s} {'Moderate':<11s} {'Severe':<9s}"
    )
    logger.info("-" * 70)
    for _, r in df.iterrows():
        logger.info(
            f"{r['model']:<14s} {r['rmse']:<7.2f} {r['macro_f1']:<9.4f} {r['total_accuracy']:<7.3f} "
            f"{r.get('acc_normal', 0):<9.3f} {r.get('acc_mild', 0):<9.3f} "
            f"{r.get('acc_moderate', 0):<11.3f} {r.get('acc_severe', 0):<9.3f}"
        )

    logger.info(f"\n结果保存到: {BACK_OUTPUT_DIR}/")
    logger.info(f"  summary: {SUMMARY_CSV}")


# ══════════════════════════ roi — ROI 全链路验收 ══════════════════════════


def _run_roi_algorithm(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """运行生产 ROI 管线（commands/batch_process_all.run_roi_pipeline），返回裁剪后 o3d mesh。

    原 compare_pants_algorithms.run_new_algorithm 已在重构中被
    commands/batch_process_all.py::run_roi_pipeline 取代（experiments/commands/
    下存档版本已无该函数）。
    """
    from commands.batch_process_all import run_roi_pipeline

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


def _o3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
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
        print(f"\n{table}")

        report_path = os.path.join(output_dir, "report.txt")
        with open(report_path, "w") as f:
            f.write(table)
        logger.info(f"Report saved: {report_path}")

    # 区域评估结果报告
    if regions or regions_only:
        from mesh.roi.region_eval import region_report_text

        for sid in subjects:
            if sid in results and "regions" in results[sid]:
                print()
                print(region_report_text(results[sid]["regions"], sid))

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    logger.info(f"Results saved: {json_path}")


def main() -> None:
    """统一评估入口 — --domain 必选，选择评估类型。"""
    parser = argparse.ArgumentParser(description="统一评估入口 — 按 --domain 选择评估类型")
    parser.add_argument(
        "--domain",
        required=True,
        choices=["back", "batch", "cut", "roi"],
        help="评估类型: back=特征方案 / batch=批量参数组合 / cut=曲线切割 / roi=ROI 验收",
    )
    parser.add_argument("--recompute", action="store_true", help="[back] 即使已有结果也重新计算")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--subjects", type=str, help="[cut/roi] 逗号分隔的 subject ID")
    group.add_argument("--old", action="store_true", help="[roi] 评估旧算法 baseline")
    parser.add_argument("--skip-run", action="store_true", help="[roi] 复用已有 mesh，不重新跑 pipeline")
    parser.add_argument("--regions", action="store_true", help="[roi] 执行区域化三角面评估（与现有指标并行）")
    parser.add_argument("--regions-only", action="store_true", help="[roi] 只跑区域评估，跳过现有指标")
    parser.add_argument("--stats", action="store_true", help="[roi] 对全部 GT subject 跑统计，生成阈值文件")
    parser.add_argument("--output", type=str, default=ROI_OUTPUT_DIR, help="[roi] 输出目录")
    args = parser.parse_args()

    if args.domain == "back":
        evaluate_back(recompute=args.recompute)
    elif args.domain == "batch":
        evaluate_batch()
    elif args.domain == "cut":
        evaluate_cut(args.subjects)
    else:
        evaluate_roi(
            subjects=[s.strip() for s in args.subjects.split(",")] if args.subjects else None,
            use_old=args.old,
            skip_run=args.skip_run,
            output_dir=args.output,
            regions=args.regions,
            regions_only=args.regions_only,
            stats=args.stats,
        )


if __name__ == "__main__":
    main()
