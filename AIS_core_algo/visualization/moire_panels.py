"""莫尔条纹渲染面板 — 纯 matplotlib 渲染（接收 ndarray + Axes）。

从 `moire/moire.py` 抽出（三层分离）：算法 `compute_moire_distances` 留在 moire/，
渲染函数 `render_moire` 归入可视化层。仅被 prediction 报告图编排调用。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import TriMesh
from matplotlib.tri import Triangulation

__all__ = ["render_moire"]


def render_moire(
    ax: Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    distances: np.ndarray,
    num_levels: int = 100,
    colors: np.ndarray | None = None,
) -> None:
    """在传入 Axes 上渲染莫尔条纹。

    无 `colors`：经典黑白交替等高线带 + 白色边界轮廓。
    有 `colors`（如 `back_panels.compute_phong_colors` 输出）：以光照渲染为底图，
    奇数条带压暗成黑纹、偶数条带露出光照底图——条纹随曲面起伏有明暗体积感。

    Args:
        ax: 目标坐标轴。
        vertices: (N, 3) 物理空间顶点坐标（x/y 投影）。
        faces: (M, 3) 三角面索引。
        distances: (N,) 每顶点到参考平面距离（compute_moire_distances 输出）。
        num_levels: 等高线数量，默认 100。
        colors: (N, 3) 逐顶点 RGB 底图颜色；None 时黑白交替纯条纹。
    """
    d_min, d_max = float(distances.min()), float(distances.max())
    levels = np.linspace(d_min, d_max, num_levels)
    n_bands = len(levels) - 1

    if len(faces) > 0:
        triang = Triangulation(vertices[:, 0], vertices[:, 1], faces)
        if colors is None:
            band_colors = ["black" if i % 2 == 0 else "white" for i in range(n_bands)]
            ax.tricontourf(triang, distances, levels=levels, colors=band_colors)
            # 条纹之间加黑色细线，增强清晰度
            ax.tricontour(triang, distances, levels=levels, colors="black", linewidths=0.3, alpha=0.4)
        else:
            # 光照底图：偶数条带露底，奇数条带压暗成黑纹
            ax.add_collection(TriMesh(triang, facecolors=colors))
            ax.autoscale_view()
            for i in range(0, n_bands, 2):
                ax.tricontourf(
                    triang, distances, levels=[levels[i], levels[i + 1]],
                    colors="black", alpha=0.55,
                )
            ax.tricontour(triang, distances, levels=levels, colors="white", linewidths=0.5, alpha=0.5)
        # 白色粗线绘制 ROI 网格外边界
        for edge in _get_boundary_edges(faces):
            ax.plot(
                vertices[edge, 0],
                vertices[edge, 1],
                color="white",
                linewidth=2.0,
                solid_capstyle="round",
            )
    else:
        # 无三角面时退化为散点分层着色
        band_colors = ["black" if i % 2 == 0 else "white" for i in range(n_bands)]
        for i in range(n_bands):
            mask = (distances >= levels[i]) & (distances < levels[i + 1])
            ax.scatter(vertices[mask, 0], vertices[mask, 1], c=band_colors[i], s=1, alpha=0.8)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Moire", fontsize=11)


def _get_boundary_edges(triangles: np.ndarray) -> list[list[int]]:
    """提取网格边界边（只出现在一个三角形中的边），供轮廓高亮。"""
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    edge_verts: dict[tuple[int, int], list[int]] = {}
    for tri in triangles:
        for i in range(3):
            v0, v1 = int(tri[i]), int(tri[(i + 1) % 3])
            key = (min(v0, v1), max(v0, v1))
            edge_count[key] += 1
            edge_verts[key] = [v0, v1]
    return [v for key, v in edge_verts.items() if edge_count[key] == 1]
