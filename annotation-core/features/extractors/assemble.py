"""主入口：从 mesh + clinical data 提取全部特征，返回单行 DataFrame。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.extractors.asymmetry import extract_asymmetry
from features.extractors.basic import extract_basic
from features.extractors.morphology import extract_morphology
from mesh.curvature import calculate_curvature


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
    1. extract_basic —— 从临床数据提取年龄、性别、BMI。
    2. extract_morphology —— 从 landmarks 提取形态学测量值。
    3. extract_asymmetry —— 从 UV 参数化结果提取不对称特征。

    Args:
        mesh:          open3d TriangleMesh 对象。
        subject_id:    受试者 ID。
        clinical_data: 临床数据字典（age / sex / bmi 等）。
        landmarks:     extract_landmarks() 输出的 landmark 字典，可选。
        uv_coords:     (N, 2) UV 参数化坐标，可选。若不提供则跳过不对称特征。
        heights:       (N,) 顶点高度，可选。若不提供则从 mesh Z 坐标计算。

    Returns:
        单行 pd.DataFrame，含 subject_id + 全部特征列。
    """
    # 1. 基本临床特征
    row = extract_basic(clinical_data)
    row["subject_id"] = subject_id

    # 2. 形态学特征
    if landmarks is not None:
        morph = extract_morphology(landmarks)
        row.update(morph)

    # 3. 不对称特征（需要 UV 参数化结果）
    if uv_coords is not None:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)

        if heights is None:
            heights = vertices[:, 2].copy()

        curv_mean = calculate_curvature(mesh, "mean")
        curv_gauss = calculate_curvature(mesh, "gaussian")

        if curv_mean is not None and curv_gauss is not None:
            asym = extract_asymmetry(
                uv_coords=uv_coords,
                heights=heights,
                curv_mean=curv_mean,
                curv_gauss=curv_gauss,
            )
            row.update(asym)

    return pd.DataFrame([row])
