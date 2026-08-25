"""度量计算 — cobb 分级、顶点级表面测量、体征参数。

顶点测量（曲率/粗糙度/法向角）由热力图与体征共用；体征参数复用
analysis.body_params.compute_cosmetic（论文表 2 口径），此处只做
标签与单位包装，不重复实现几何计算。
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from mesh.curvature import calculate_curvature
from mesh.roi.bfs import compute_mesh_roughness, scatter_face_values_to_vertices

# 体征参数对外标签（论文表 2 键 → 可读名 + 单位）
_BODY_PARAM_LABELS = {
    "Sh.IB": ("左右肩垂直高度差", "mm"),
    "Sh.A": ("肩线倾角", "deg"),
    "Sca.IB": ("左右肩胛垂直高度差", "mm"),
    "Sca.A": ("肩胛线倾角", "deg"),
    "ASIS.A": ("腰线倾角", "deg"),
    "Trunk.L": ("躯干长度", "mm"),
    "Sh.W": ("肩宽", "mm"),
    "Sh.AI": ("肩不对称指数", ""),
    "Pe.AI": ("骨盆不对称指数", ""),
}


def _compute_measures(cut_mesh: o3d.geometry.TriangleMesh) -> dict:
    """计算顶点级表面测量（高度/曲率/粗糙度/法向角），供指数与热力图共用。"""
    vertices = np.asarray(cut_mesh.vertices, dtype=np.float64)
    faces = np.asarray(cut_mesh.triangles, dtype=np.int64)
    heights = vertices[:, 2]
    curv_mean = calculate_curvature(cut_mesh, "mean")
    curv_gauss = calculate_curvature(cut_mesh, "gaussian")
    rough_f = compute_mesh_roughness(cut_mesh)
    # 面粗糙度 → 顶点（散射取均值，复用 mesh.roi.bfs 共享实现）
    rough_v = scatter_face_values_to_vertices(faces, rough_f, len(vertices))
    cut_mesh.compute_vertex_normals()
    vn = np.asarray(cut_mesh.vertex_normals)
    normal_angle = np.degrees(np.arccos(np.clip(np.abs(vn[:, 1]), 0, 1)))
    return {
        "vertices": vertices,
        "faces": faces,
        "heights": heights,
        "curv_mean": curv_mean,
        "curv_gauss": curv_gauss,
        "roughness": rough_v,
        "normal_angle": normal_angle,
        "normals": vn,  # 单位法向量（光照渲染 back.png 用）
    }


def _compute_body_params(gt: dict, subject_id: str) -> dict:
    """计算 9 个体征参数（论文表2），键为 {info（医学意义带单位）, value}。"""
    from analysis.body_params import compute_cosmetic, gt_to_csv_row

    row = gt_to_csv_row(gt, subject_id)
    params = compute_cosmetic(row)
    out: dict = {}
    for key, value in params.items():
        name, unit = _BODY_PARAM_LABELS.get(key, (key, ""))
        info = f"{name}({unit})" if unit else name
        out[key] = {"info": info, "value": float(value)}
    return out
