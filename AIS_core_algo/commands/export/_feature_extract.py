"""导出脚本共享的「人工 ROI → 全量特征」胶水 — 消除 batch/verify 的编排复制。

与 predict 链路同口径：run_pipeline mesh_path 默认人工 ROI
（data/ground_truth/{sid}/roi.ply）、landmarks 用完整 GT json（扁平化后喂参数化）。
reproduce_best_mae / roi_compare 用训练缓存 mesh_cut（不重新参数化）或定制多路对比，
不适用本函数。
"""

from __future__ import annotations

import json
from pathlib import Path

import open3d as o3d
import pandas as pd

from features.extractors.assemble import extract_all
from parameterization.pipeline import run_pipeline


def extract_features_from_roi(
    sid: str,
    clinical_file: str | Path,
    gt_file: str | Path,
    param_out: str | Path,
) -> pd.DataFrame:
    """人工 ROI → 参数化 + extract_all → 单行全量特征 DataFrame。

    Args:
        sid: subject ID（run_pipeline 默认读 data/ground_truth/{sid}/roi.ply）。
        clinical_file: 临床数据 JSON 路径。
        gt_file: ground_truth.json 路径（完整 18 点，扁平化后喂参数化）。
        param_out: 参数化输出目录。

    Returns:
        单行特征 DataFrame（subject_id + 2736 特征列）。
    """
    clinical = json.loads(Path(clinical_file).read_text(encoding="utf-8"))[sid]
    gt = json.loads(Path(gt_file).read_text(encoding="utf-8"))
    landmarks = dict(gt)  # extract_all 直接消费扁平 18 键
    uv = run_pipeline(sid, output_dir=str(param_out))
    mesh = o3d.io.read_triangle_mesh(str(Path(param_out) / sid / "mesh_cut.ply"))
    return extract_all(mesh, sid, {sid: clinical}, landmarks, uv_coords=uv)
