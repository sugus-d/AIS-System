"""编排层：landmark 连线对比图渲染。

接收 CLI 参数，调用渲染模块生成 landmark 图，复用 plot_shared._save_figure。
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from commands.plot_shared import _save_figure
from utils.logger import logger


def render_landmarks(
    subject: str,
    cache_dir: str,
    output_dir: str,
    skip_run: bool,
) -> None:
    """渲染带解剖标注的曲率图（2 面板）。

    上方面板：曲率图 + 所有解剖标注点（neck_root、shoulder_transition、axilla、waist、scapular_peaks、spine），
    下方面板：waist 调试信息（仅在存在时显示）。

    waist 定位依赖轮廓窄茎最细处，结果易受噪声影响，
    因此需要独立的调试面板来验证 waist 是否定位在正确的 Y 层级。
    """
    from utils.io import load_landmarks
    from utils.mesh import load_cached_mesh
    from visualization._data_utils import load_cached_numpy
    from visualization.landmarks_panels import render_curvature_landmarks_panel

    # 加载缓存
    lmks = load_landmarks(subject, cache_dir)
    mesh = load_cached_mesh(subject, cache_dir)
    curv = load_cached_numpy(cache_dir, subject, "curvature", "mean_curvature.npy")

    missing: list[str] = []
    if lmks is None:
        missing.append("landmarks.pkl")
    if mesh is None:
        missing.append("cached mesh")
    if curv is None:
        missing.append("curvature/mean_curvature.npy")

    if missing:
        logger.error(f"Missing cached inputs for {subject}: {', '.join(missing)}")
        return

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        logger.error(f"Mesh has no triangles for {subject}")
        return

    out_dir = os.path.join(output_dir, subject)
    os.makedirs(out_dir, exist_ok=True)

    # 单面板：曲率图 + 各解剖标记点叠层
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor="black")
    render_curvature_landmarks_panel(ax, vertices, triangles, curv, lmks, subject)
    _save_figure(fig, os.path.join(out_dir, "landmarks.png"), facecolor="black")
