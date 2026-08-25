"""主入口：从 mesh + clinical data 提取全部特征，返回单行 DataFrame。

特征组成与训练特征提取（v0.1.0）一致：
  basic(5 临床) + morph(31) + region candidate(2700) = 2736。
region 特征来自 :mod:`features.extractors.asymmetry` 的 candidate + pairwise
（225 候选区 × 6 测量 × 2 差异模式），列名与 results/extraction/features_extraction
/v0.1.0/region_asymmetry.csv 完全一致，供 api.predict 按特征方案选列。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.extractors.asymmetry import (
    compute_candidate_asymmetry,
    compute_candidate_asymmetry_pairwise,
)
from features.extractors.asymmetry.measures import (
    compute_gauss_curvature,
    compute_mean_curvature,
)
from features.extractors.basic.clinical import extract_basic  # 与训练 basic.csv 一致的临床特征
from features.extractors.morphology import extract_morphology
from mesh.roi.bfs import compute_mesh_roughness, scatter_face_values_to_vertices


def _vertex_roughness(mesh) -> np.ndarray:
    """per-face 粗糙度 → 顶点（散射取均值，复用 mesh.roi.bfs 共享实现）。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    rough_f = compute_mesh_roughness(mesh)
    return scatter_face_values_to_vertices(faces, rough_f, len(vertices))


def _asymmetry_row(
    row: dict,
    uv_coords: np.ndarray,
    vertices: np.ndarray,
    heights: np.ndarray,
    curv_mean: np.ndarray,
    curv_gauss: np.ndarray,
    mesh,
) -> None:
    """计算 candidate + pairwise 不对称特征并写入 row（原地）。"""
    rough = _vertex_roughness(mesh)
    mesh.compute_vertex_normals()
    vn = np.asarray(mesh.vertex_normals)
    normal_angle = np.degrees(np.arccos(np.clip(np.abs(vn[:, 1]), 0, 1)))

    cand, cand_names = compute_candidate_asymmetry(
        uv_coords, heights, curv_mean, curv_gauss, rough, normal_angle, vn
    )
    pw, pw_names = compute_candidate_asymmetry_pairwise(
        uv_coords, heights, curv_mean, curv_gauss, rough, normal_angle, vn,
        vertices=vertices,
    )
    # 特征矩阵 (R, Q) 行优先展平，与 names（每候选区按测量序）一一对应
    for i, name in enumerate(cand_names):
        row[name] = float(cand.ravel()[i])
    for i, name in enumerate(pw_names):
        row[name] = float(pw.ravel()[i])


def extract_all(
    mesh,
    subject_id: str,
    clinical_data: dict,
    landmarks: dict | None = None,
    *,
    uv_coords: np.ndarray | None = None,
    heights: np.ndarray | None = None,
) -> pd.DataFrame:
    """从 mesh + clinical data 提取全部特征，返回单行 DataFrame。

    流程：
    1. extract_basic —— 从临床数据提取身高/体重/BMI/性别等。
    2. extract_morphology —— 从 landmarks 提取形态学测量值。
    3. candidate + pairwise —— 从 UV 参数化结果提取区域不对称特征（2700）。

    Args:
        mesh:          open3d TriangleMesh 对象（与 uv_coords 顶点对齐）。
        subject_id:    受试者 ID。
        clinical_data: 临床数据字典（{sid: {...}} 或含 height_cm 等字段）。
        landmarks:     extract_landmarks() 输出的 landmark 字典，可选。
        uv_coords:     (N, 2) UV 参数化坐标，可选。若不提供则跳过不对称特征。
        heights:       (N,) 顶点高度，可选。若不提供则从 mesh Z 坐标计算。

    Returns:
        单行 pd.DataFrame，含 subject_id + basic + morph + region 特征列。
    """
    # 1. 基本临床特征（extract_basic 返回 {sid: {...}}，取当前 subject 条目）
    basic_all = extract_basic(clinical_data)
    row = dict(basic_all.get(subject_id, {}))
    row["subject_id"] = subject_id

    # 2. 形态学特征
    if landmarks is not None:
        morph = extract_morphology(landmarks)
        row.update(morph)

    # 3. 区域不对称特征（需要 UV 参数化结果）
    if uv_coords is not None:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)

        if heights is None:
            heights = vertices[:, 2].copy()

        # 与训练特征提取（v0.1.0 measures.py）一致：曲率经 1%-99% 百分位截断
        curv_mean = compute_mean_curvature(mesh)
        curv_gauss = compute_gauss_curvature(mesh)

        if curv_mean is not None and curv_gauss is not None:
            _asymmetry_row(row, uv_coords, vertices, heights, curv_mean, curv_gauss, mesh)

    return pd.DataFrame([row])
