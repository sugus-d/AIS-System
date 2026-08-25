#!/usr/bin/env python3
"""Batch pre-label all subjects — save no-clothing PLY + algorithm landmark GT.

用法:
    uv run python -m commands.batch_prelabel
    uv run python -m commands.batch_prelabel S0119,S0001
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from commands.cli_common import batch_cli, find_mesh_path
from landmarks.extract import extract_landmarks
from mesh.roi_extract import extract_back_roi

# 左右成对 landmark 的点数（L/R 各一个）
_PAIRED_LANDMARK_COUNT = 2

BASE = Path("projects/AIS/src/core")
MESH_DIR = BASE / "data" / "mesh"  # 原始 mesh 存放目录
CACHE_DIR = BASE / "results" / "cache"  # pipeline 缓存输出
GT_DIR = BASE / "results" / "ground-truth"  # Ground Truth JSON 存放目录
OUT_PLY_DIR = BASE / "results" / "meshes_processed"  # 去衣物网格输出目录


def process_subject(subject_id: str) -> dict[str, Any]:
    """对单个 subject 执行 ROI 提取 + 预标注，返回处理状态字典。

    步骤：
      1. 查找 mesh 文件，不存在则返回 SKIP
      2. 提取背部 ROI（去衣物）
      3. 运行算法 landmark 检测
      4. 保存去衣物网格 + Ground Truth JSON

    Returns:
        dict[str, Any]: 包含 status / time_s / verts_raw / landmark_keys 等字段。
                        非 fatal 错误通过 status="SKIP" 返回，fatal 错误向上抛出。
    """
    info = {"id": subject_id}
    t0 = time.time()

    mesh_path = find_mesh_path(subject_id, MESH_DIR)
    if mesh_path is None:
        # 原始 mesh 文件不存在 → 跳过，不视为错误
        info["status"] = "SKIP"
        info["reason"] = "no STD_fuse_mesh file"
        return info
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    info["verts_raw"] = len(np.asarray(mesh.vertices))
    info["mesh_file"] = mesh_path.name

    # 创建输出目录：缓存、GT、去衣物网格各一份
    extract_dir = CACHE_DIR / subject_id / "extract_roi"
    extract_dir.mkdir(parents=True, exist_ok=True)
    gt_subdir = GT_DIR / subject_id
    gt_subdir.mkdir(parents=True, exist_ok=True)
    OUT_PLY_DIR.mkdir(parents=True, exist_ok=True)

    # 背部 ROI 提取：分离衣物，保留躯干背面
    roi = extract_back_roi(mesh, subject_id=subject_id, angle_threshold_deg=50.0)
    roi_verts = np.asarray(roi.vertices)
    info["verts_roi"] = len(roi_verts)

    # 保存去衣物网格：一份用于缓存重跑，一份用于标注平台加载
    o3d.io.write_triangle_mesh(str(extract_dir / "output.ply"), roi)
    o3d.io.write_triangle_mesh(str(OUT_PLY_DIR / f"{subject_id}_no_clothing.ply"), roi)
    info["ply_saved"] = True

    # 运行算法 landmark 检测，结果同时写入 info 和 GT JSON
    landmarks = extract_landmarks(roi, is_debug=False)
    keys = ["neck_root", "shoulder_transition", "scapular_peaks", "axilla", "waist", "spine_points"]
    lm = {k: np.asarray(landmarks[k]).tolist() for k in keys if k in landmarks}
    info["landmark_keys"] = list(lm.keys())

    # 构建 Ground Truth JSON：spine 点保存为数组，左右成对 landmark 保存为 {"L": ..., "R": ...}
    gt = {"_features": {}}
    for name, pts in lm.items():
        if name == "spine_points":
            gt[name] = pts
        elif isinstance(pts, list) and len(pts) == _PAIRED_LANDMARK_COUNT:
            gt[name] = {"L": pts[0], "R": pts[1]}
    gt_file = gt_subdir / "ground_truth.json"
    gt_file.write_text(json.dumps(gt, indent=2, ensure_ascii=False) + "\n")
    info["gt_saved"] = True
    info["time_s"] = round(time.time() - t0, 1)
    info["status"] = "OK"
    return info


def _discover_all_subjects() -> str:
    """扫描 mesh 目录下所有 subject，返回逗号分隔字符串。"""
    return ",".join(sorted(d.name for d in MESH_DIR.iterdir() if d.is_dir()))


@batch_cli(default_subjects="<auto>")
def main(subjects: str) -> None:
    """对指定 subject（或全部）运行 ROI 提取 + 预标注。

    subjects 为逗号分隔列表，传入 ``<auto>`` 自动扫描所有 subject。
    """
    if subjects == "<auto>":
        # 自动发现所有 subject，避免手动维护列表
        subjects = _discover_all_subjects()

    subject_list = [s.strip() for s in subjects.split(",")]
    total = len(subject_list)
    print(f"Processing {total} subjects\n")

    results = {"OK": 0, "SKIP": 0, "FAIL": 0}
    for i, sid in enumerate(subject_list, 1):
        print(f"[{i}/{total}] {sid} ... ", end="", flush=True)
        try:
            info = process_subject(sid)
            if info["status"] == "SKIP":
                results["SKIP"] += 1
                print(f"⏭  {info.get('reason', '')}")
            else:
                results["OK"] += 1
                print(f"✅ {info['time_s']}s")
        except Exception as e:
            results["FAIL"] += 1
            err_msg = str(e).split("\n")[0][:80]
            print(f"❌ {err_msg}")

    print(f"\n{'=' * 50}")
    print(f"Done: {results['OK']} OK | {results['SKIP']} Skip | {results['FAIL']} Fail")
    print(f"Total: {sum(results.values())}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Wall time: {time.time() - start:.0f}s")
