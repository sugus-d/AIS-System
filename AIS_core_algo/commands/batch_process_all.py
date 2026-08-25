#!/usr/bin/env python3
"""批量对所有 subject 运行最新 ROI 提取 + Landmark 标注管线。

用法:
    uv run python -m commands.batch_process_all                    # 处理全部未处理的 subject
    uv run python -m commands.batch_process_all --all              # 强制重新处理全部
    uv run python -m commands.batch_process_all --subjects S0119,S0001  # 指定 subject
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import open3d as o3d

from commands.cli_common import find_mesh_path
from landmarks.extract import extract_landmarks
from mesh.roi.pipeline import run_roi_pipeline
from utils.logger import logger
from utils.paths import GROUND_TRUTH_OUTPUT_DIR, MESH_DIR, ROI_DIR

EXPORT_DIR = GROUND_TRUTH_OUTPUT_DIR
OUTPUT_DIR = ROI_DIR

# 左右成对 landmark 的点数（L/R 各一个）
_BILATERAL_POINT_COUNT = 2
# ROI 顶点数下限，小于该值视为提取失败
_MIN_ROI_VERTS = 100


def _landmarks_to_json(landmarks: dict) -> dict:
    """将 extract_landmarks 输出（扁平语义键）转为 ground_truth.json 格式。

    extract_landmarks 已输出扁平 18 键（含 debug 辅助数据），这里过滤纯 landmark 键。
    """
    from landmarks.constants import FLAT_KEYS

    gt: dict = {"_features": {}}
    for key in FLAT_KEYS:
        if key in landmarks:
            gt[key] = np.asarray(landmarks[key]).tolist()
    return gt


def process_subject(subject_id: str) -> dict:
    """对单个 subject 执行最新 ROI 提取 + 标注。"""
    info = {"id": subject_id}
    t0 = time.time()

    # 查找 mesh
    mesh_path = find_mesh_path(subject_id, MESH_DIR)
    if mesh_path is None:
        info["status"] = "SKIP"
        info["reason"] = "no STD_fuse_mesh file"
        return info

    orig_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    v = np.asarray(orig_mesh.vertices, dtype=np.float64)
    t = np.asarray(orig_mesh.triangles, dtype=np.int32)
    info["verts_raw"] = len(v)

    # 创建导出目录
    export_sd = EXPORT_DIR / subject_id
    export_sd.mkdir(parents=True, exist_ok=True)

    # 保存 original.ply（如不存在）
    orig_ply = export_sd / "original.ply"
    if not orig_ply.exists():
        o3d.io.write_triangle_mesh(str(orig_ply), orig_mesh)

    # Step 1: ROI 提取（最新管线）
    roi_v, roi_t = run_roi_pipeline(v, t)
    info["verts_roi"] = len(roi_v)
    logger.info(f"{subject_id}: in={len(v)} → roi={len(roi_v)}")

    if len(roi_v) < _MIN_ROI_VERTS:
        info["status"] = "FAIL"
        info["reason"] = f"ROI too small: {len(roi_v)} verts"
        return info

    # 保存 roi.ply
    roi_mesh = o3d.geometry.TriangleMesh()
    roi_mesh.vertices = o3d.utility.Vector3dVector(roi_v)
    roi_mesh.triangles = o3d.utility.Vector3iVector(roi_t)
    roi_out = OUTPUT_DIR / subject_id
    roi_out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(roi_out / "roi.ply"), roi_mesh)

    # Step 2: Landmark 检测（即使失败也保存 ROI 结果）
    landmarks = None
    try:
        landmarks = extract_landmarks(roi_mesh)
        info["landmark_keys"] = list(k for k in landmarks if not k.endswith("_debug"))
    except Exception as e:
        info["landmark_fail"] = str(e)[:80]
        logger.warning(f"{subject_id}: landmarks failed: {e}")

    # Step 3: 保存 ground_truth.json（如有）
    if landmarks:
        gt = _landmarks_to_json(landmarks)
        gt_dir = GROUND_TRUTH_OUTPUT_DIR / subject_id
        gt_dir.mkdir(parents=True, exist_ok=True)
        (gt_dir / "ground_truth.json").write_text(json.dumps(gt, indent=2, ensure_ascii=False) + "\n")

    info["status"] = "PARTIAL" if "landmark_fail" in info else "OK"
    info["time_s"] = round(time.time() - t0, 1)
    return info


def _find_unprocessed() -> list[str]:
    """找出 data/mesh/ 中尚未导出或导出不完整的 subject。"""
    unprocessed: list[str] = []
    for d in sorted(MESH_DIR.iterdir()):
        if not d.is_dir():
            continue
        sid = d.name
        exp = EXPORT_DIR / sid
        # 检查是否已有完整导出
        if exp.is_dir():
            has_o = (exp / "original.ply").exists()
            has_d = (OUTPUT_DIR / sid / "roi.ply").exists()
            has_lm = (GROUND_TRUTH_OUTPUT_DIR / sid / "ground_truth.json").exists()
            if has_o and has_d and has_lm:
                continue
        unprocessed.append(sid)
    return unprocessed


def main(subjects: str | None = None, force_all: bool = False) -> None:
    if force_all:
        # 所有 subject
        subject_list = sorted(d.name for d in MESH_DIR.iterdir() if d.is_dir())
    elif subjects:
        subject_list = [s.strip() for s in subjects.split(",")]
    else:
        subject_list = _find_unprocessed()

    total = len(subject_list)
    logger.info(f"Processing {total} subjects")

    results = {"OK": 0, "PARTIAL": 0, "SKIP": 0, "FAIL": 0}
    for i, sid in enumerate(subject_list, 1):
        logger.info(f"[{i}/{total}] {sid} ...")
        try:
            info = process_subject(sid)
            if info["status"] == "SKIP":
                results["SKIP"] += 1
                logger.warning(f"⏭  {info.get('reason', '')}")
            elif info["status"] == "PARTIAL":
                results["PARTIAL"] += 1
                logger.warning(f"⚠  {info.get('landmark_fail', '')}")
            elif info["status"] == "FAIL":
                results["FAIL"] += 1
                logger.error(f"❌ {info.get('reason', '')}")
            else:
                results["OK"] += 1
                logger.info(f"✅ {info['time_s']}s (roi={info['verts_roi']})")
        except Exception as e:
            results["FAIL"] += 1
            logger.error(f"❌ {str(e)[:80]}")

    logger.info("=" * 50)
    logger.info(f"Done: {results['OK']} OK | {results['PARTIAL']} Partial | {results['SKIP']} Skip | {results['FAIL']} Fail")
    logger.info(f"Total: {total}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, help="逗号分隔的 subject ID 列表")
    parser.add_argument("--all", action="store_true", help="强制重新处理所有 subject")
    args = parser.parse_args()

    if not args.subjects and not args.all:
        unprocessed = _find_unprocessed()
        if unprocessed:
            logger.info(f"发现 {len(unprocessed)} 个未处理的 subject:")
            for s in unprocessed:
                logger.info(f"  {s}")
            confirm = input("是否处理这些 subject？[Y/n] ").strip().lower()
            if confirm not in ("", "y", "yes"):
                logger.info("已取消")
                sys.exit(0)

    start = time.time()
    main(subjects=args.subjects, force_all=args.all)
    logger.info(f"Wall time: {time.time() - start:.0f}s")
