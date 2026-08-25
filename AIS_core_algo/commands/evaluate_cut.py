#!/usr/bin/env python3
"""曲线切割评估域 — 曲线切割评估 + 批量参数组合评估。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from utils.logger import logger
from utils.paths import EVAL_CUT_DIR, GROUND_TRUTH_INPUT_DIR

# ── cut / batch 共享 ──
DATA_DIR = GROUND_TRUTH_INPUT_DIR
OUTPUT_DIR = EVAL_CUT_DIR
# mesh 顶点数下限，小于该值视为无效
_MIN_MESH_VERTS = 50
# 曲面点匹配判定距离（mm），小于该距离视为命中
_MATCH_DISTANCE_MM = 5.0
# 覆盖率达标阈值
_COVERAGE_TARGET = 0.95
# 多余率达标阈值
_EXCESS_TARGET = 0.08


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
        from mesh.roi.pipeline import run_roi_pipeline

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

    logger.info("\n" + "=" * 80)
    logger.info("曲线切割评估结果 (pipeline: BFS→erosion→pants_cut)")
    logger.info(f"耗时: {elapsed:.0f}s  subject: {len(ok)} OK / {len(failed)} FAIL")
    logger.info("=" * 80)

    if ok:
        covs = [r["coverage"] for r in ok]
        excs = [r["excess"] for r in ok]
        chs = [r.get("chamfer", 0) for r in ok if "chamfer" in r]
        logger.info(f"覆盖率:  均值={np.mean(covs):.4f}  中位数={np.median(covs):.4f}")
        logger.info(f"多余率:  均值={np.mean(excs):.4f}  中位数={np.median(excs):.4f}")
        if chs:
            logger.info(f"Chamfer: 均值={np.mean(chs):.2f}mm  中位数={np.median(chs):.2f}mm")

        logger.info("\n各 subject:")
        for r in sorted(ok, key=lambda x: -x.get("excess", 0)):
            logger.info(f"  {r['subject']}: cov={r['coverage']:.4f}  exc={r['excess']:.4f}  v={r.get('algo_v', '?')}")

    if failed:
        logger.info(f"\n失败 ({len(failed)}):")
        for r in failed:
            logger.info(f"  {r['subject']}: {r.get('error', '?')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "eval_pipeline.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nSaved: {path}")


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
            logger.info(
                f"[{cfg['name']}] N={len(ok)}  cov={covs.mean():.4f}  exc={excs.mean():.4f}  "
                f"cov>95%={(covs >= _COVERAGE_TARGET).sum()}  exc<8%={(excs < _EXCESS_TARGET).sum()}  "
                f"chamfer={np.median([r.get('chamfer', 0) for r in ok]):.2f}mm  ({elapsed:.0f}s)"
            )
        logger.info(f"Saved: {path}\n")

    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
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
        logger.info(
            f"  {cfg['name']:<10} erosion={cfg['erosion']} pants={cfg['pants_ratio']:.1f}  "
            f"cov={covs.mean():.4f}  exc={excs.mean():.4f}  "
            f"cov5%={p5:.4f}  exc<8%={(excs < _EXCESS_TARGET).sum()}/{len(ok)}"
        )
