#!/usr/bin/env python3
"""预测核心 — PLY → landmarks / cobb 预测 + 报告图。

CLI 与 HTTP API 共用的唯一核心（channel 不同，契约一致）：
  - CLI:  `python -m prediction.cli`（三模式：landmarks / predict / auto）
  - HTTP: `uvicorn prediction.api:app`（POST /api/landmarks + /api/predict）

三模式（详见 prediction/README.md）：
  landmarks: original.ply → ROI + landmarks（两段式第一段）
  predict:   roi.ply + clinical + landmarks → cobb + 报告图（--landmarks 必填）
  auto:      original.ply + clinical → 自动 landmarks → predict（单步入口）

产物（prediction/outputs/<subject_id>/）：
  ├─ roi.ply + landmarks.json   # auto 落盘；predict 输入已是 ROI+landmarks，不重复产出
  ├─ features.csv / prediction.json
  └─ report/*.png               # 8 张：4 热力图 + landmarks + back + moire + waterfall

模型（--model 字段，缺省 v1.0.0）：
  - v1.0.0（生产）: 人工 ROI → per-class α + Ridge-AI 边界 Ensemble（OOF MF1=0.7364 / MAE=4.38°）
  - v0.1.0（历史）: 算法 ROI → 0.6×CompositeV7 + 0.4×AI-LR（manuscript 复现口径）
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from features.extractors.assemble import extract_all
from landmarks.complete import complete_landmarks_flat
from landmarks.constants import FLAT_KEYS
from landmarks.extract import extract_landmarks
from modeling.model_package import load_model_package
from prediction.feature_pipeline import (
    _compute_indices,
    _predict,
    _run_parameterization,
)
from prediction.measures import _compute_body_params
from prediction.model_registry import _resolve_model_id
from prediction.report import _visualize
from prediction.report_waterfall import _render_waterfall
from utils.logger import logger
from utils.paths import PREDICTION_OUTPUTS_DIR

PREDICT_ROOT = PREDICTION_OUTPUTS_DIR


def _load_mesh(ply_path: str) -> o3d.geometry.TriangleMesh:
    """加载 PLY mesh。"""
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty():
        raise ValueError(f"无法加载 mesh: {ply_path}")
    return mesh


def _extract_roi(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """运行生产 ROI 管线，返回裁剪后的背部网格。"""
    from mesh.roi.pipeline import run_roi_pipeline

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    roi_v, roi_t = run_roi_pipeline(vertices, triangles)
    roi = o3d.geometry.TriangleMesh()
    roi.vertices = o3d.utility.Vector3dVector(roi_v)
    roi.triangles = o3d.utility.Vector3iVector(roi_t)
    return roi


def run_landmarks(ply_path: str, subject_id: str, out_dir: Path) -> Path:
    """PLY → ROI → landmark 检测 → 补全 18 点 → 保存扁平 landmarks.json + roi.ply。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = _load_mesh(ply_path)
    roi = _extract_roi(mesh)
    if roi.is_empty() or not roi.has_triangles():
        raise ValueError(f"ROI 提取结果为空: {subject_id}")
    o3d.io.write_triangle_mesh(str(out_dir / "roi.ply"), roi)

    landmarks = extract_landmarks(roi)
    # extract_landmarks 已输出扁平语义键，过滤出纯 landmark 键
    flat = {key: landmarks[key] for key in FLAT_KEYS if key in landmarks}
    flat = complete_landmarks_flat(flat, roi)
    lm_path = out_dir / "landmarks.json"
    lm_path.write_text(json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"landmarks 已保存: {lm_path}")
    return lm_path


def _load_clinical(clinical_path: str, subject_id: str) -> dict:
    """从 clinical JSON 提取单 subject 的临床数据（兼容全量/单条格式）。"""
    data = json.loads(Path(clinical_path).read_text(encoding="utf-8"))
    entry = data.get(subject_id)
    if isinstance(entry, dict):
        return entry
    # 单条格式：顶层就是字段（含 subject_id）
    if "height_cm" in data or "weight_kg" in data:
        return data
    raise ValueError(f"clinical 数据中找不到 subject {subject_id}: {clinical_path}")


def _predict_flow(
    ply_path: str,
    subject_id: str,
    clinical_path: str,
    landmarks_path: str | None,
    model_path: str,
    out_dir: Path,
    persist_roi_lm: bool = True,
) -> None:
    """predict / auto 共用流程：landmarks → 参数化 → 特征 → 模型 → 报告。

    landmarks 来源：显式传入（新旧格式兼容）→ 否则算法自动检测。
    ROI 输入：传入 landmarks（精确模式）时 ply 已是 ROI 网格直接复用；
    auto 模式 run_landmarks 内部已提取 ROI 写入 out_dir/roi.ply。

    Args:
        persist_roi_lm: 是否把 roi.ply / landmarks.json 作为产物落盘。
            auto 模式 True（返回 landmarks + roi）；predict 模式 False
            （输入已是 ROI + landmarks，不重复产出）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t_prev = time.perf_counter()

    def _checkpoint(label: str) -> None:
        """记录自上次 checkpoint 到当前的耗时（供性能评估）。"""
        nonlocal t_prev
        now = time.perf_counter()
        timings[label] = now - t_prev
        t_prev = now

    # 1. landmarks：显式传入（扁平 18 键）；否则算法自动检测
    if landmarks_path and Path(landmarks_path).exists():
        data = json.loads(Path(landmarks_path).read_text(encoding="utf-8"))
        flat = dict(data)
        logger.info(f"使用传入 landmarks: {landmarks_path}")
    else:
        logger.info("未提供 landmarks，自动检测")
        lm_path = run_landmarks(ply_path, subject_id, out_dir)
        flat = json.loads(lm_path.read_text(encoding="utf-8"))
    _checkpoint("landmarks")

    # 2. 临床数据
    clinical_data = _load_clinical(clinical_path, subject_id)
    clinical_full = {subject_id: clinical_data}

    # 3. ROI：精确模式（传入 landmarks）ply 已是 ROI 网格，直接复用；
    #    auto 模式 run_landmarks 已提取 ROI 写入。predict 模式不持久化 roi.ply，
    #    直接用输入 ply 作为 ROI 网格
    roi_path = out_dir / "roi.ply"
    if persist_roi_lm and not roi_path.exists():
        shutil.copyfile(ply_path, roi_path)
    if not roi_path.exists():
        roi_path = Path(ply_path)  # predict 模式：输入即 ROI
    roi_mesh = o3d.io.read_triangle_mesh(str(roi_path))

    # 4. 补全 18 点（训练集平均 → 相似变换 → mesh 最近顶点），写扁平产物
    flat = complete_landmarks_flat(flat, roi_mesh)
    missing = [key for key in FLAT_KEYS if key not in flat]
    if missing:
        raise ValueError(f"landmarks 不完整且无法补全，缺失: {missing[:5]}")
    landmarks = flat  # extract_all 直接消费扁平 18 键
    gt = flat  # body_params 直接消费扁平 18 键
    if persist_roi_lm:
        lm_path = out_dir / "landmarks.json"
        lm_path.write_text(json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8")
    # 参数化直接消费扁平 18 键（parse_landmarks_json 只认扁平）
    param_lm_path = out_dir / "param" / "landmarks_gt.json"
    param_lm_path.parent.mkdir(parents=True, exist_ok=True)
    param_lm_path.write_text(json.dumps(flat, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5. UV 参数化
    cut_mesh, uv = _run_parameterization(subject_id, roi_path, param_lm_path, out_dir)
    _checkpoint("parameterization")

    # 6. 特征提取 + 预测
    feature_df = extract_all(cut_mesh, subject_id, clinical_full, landmarks, uv_coords=uv)
    feature_df.drop(columns=["subject_id"], errors="ignore").to_csv(out_dir / "features.csv", index=False)
    model_pkg = load_model_package(model_path)
    pred = _predict(feature_df, model_pkg)
    _checkpoint("features_predict")

    # 7. 计算不对称指数 + 体征参数，保存预测结果
    indices = _compute_indices(feature_df, model_pkg)
    body_params = _compute_body_params(gt, subject_id)
    result = {
        "subject_id": subject_id,
        "cobb": pred["cobb"],
        "severity": pred["severity"],
        "model_id": _resolve_model_id(model_path),
        "clinical": clinical_data,
        "indices": indices,
        "body_params": body_params,
    }
    (out_dir / "prediction.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"预测结果: cobb={pred['cobb']:.1f}° ({pred['severity']})")
    _visualize(roi_mesh, flat, out_dir)
    _render_waterfall(feature_df, model_pkg, out_dir, subject_id, pred["severity"], pred["cobb"])
    _checkpoint("report")
    logger.info(
        f"单 subject 总耗时: {sum(timings.values()):.1f}s | "
        + " | ".join(f"{label}={elapsed:.1f}s" for label, elapsed in timings.items())
    )
    logger.info(f"预测完成，输出目录: {out_dir}")
