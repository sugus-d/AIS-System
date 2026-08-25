"""背部原始图像渲染：Blinn-Phong 光照渲染（白底，中国肤色/珍珠白配色）。

渲染层——无 I/O、无计算、无 open3d。接收编排层算好的顶点/面/顶点法向量，
用 Blinn-Phong 光照模型（环境光 + 漫反射 + 镜面高光 + 弱补光）在白底上
画出人体背部：凸起部位亮、脊沟/凹陷深。

- :func:`render_back`：纯底图（无 landmarks）——对应 back.png。
- :func:`render_back_landmarks`：底图 + 标注平台配色 landmarks——对应 landmarks.png。
- :func:`compute_phong_colors`：逐顶点光照颜色（供 moire 等复用）。
- :data:`_PALETTES`：可选配色（中国肤色三档 + pearl 白底）。
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import TriMesh
from matplotlib.tri import Triangulation

from visualization.heatmap_panels import _overlay_landmarks

# 定向光方向（左上前，与 backImage.png「左亮右暗、上亮下暗」一致）
_LIGHT_DIR = np.array([-0.35, 0.42, 0.84])
# 弱补光（右下前）：照亮背光侧，避免暗部死黑丢细节
_FILL_DIR = np.array([0.5, -0.15, 0.75])
# 视图方向（正交投影）
_VIEW_DIR = np.array([0.0, 0.0, 1.0])
# 光照强度（环境/漫反射/补光）
_AMBIENT = 0.15
_DIFFUSE = 0.7
_FILL = 0.35
# 皮肤渲染参数（依据 GPU Gems 3「高级实时皮肤渲染」）：
# wrap lighting 让明暗过渡柔和（皮肤没有硬阴影线）；
# SSS 近似让暗部泛血色——光线进入皮肤被血液散射后以红色出射。
_WRAP = 0.6
_SSS_STRENGTH = 0.6
_BLOOD = np.array([0.6, 0.2, 0.16])
# 默认配色名（中国肤色）
_DEFAULT_PALETTE = "cn_skin"

# 配色方案：基色 + 高光色 + 光泽度 + 镜面强度 + 是否启用 SSS 泛红。
# 中国肤色 = 暖黄基底 + 粉调（R 略高、G-B 拉开）；哑光低光泽柔高光；SSS 开。
# pearl 仅作 moire 白底（SSS 关，保持近白）。
_PALETTES: dict[str, tuple[np.ndarray, np.ndarray, float, float, bool]] = {
    "cn_light": (np.array([0.86, 0.71, 0.58]), np.array([1.0, 0.96, 0.92]), 12.0, 0.25, True),
    "cn_skin": (np.array([0.82, 0.64, 0.52]), np.array([1.0, 0.95, 0.9]), 15.0, 0.25, True),
    "cn_deep": (np.array([0.7, 0.52, 0.4]), np.array([1.0, 0.93, 0.87]), 12.0, 0.25, True),
    "pearl": (np.array([0.88, 0.86, 0.84]), np.array([1.0, 0.99, 0.98]), 25.0, 0.35, False),
}


def compute_phong_colors(normals: np.ndarray, palette: str = _DEFAULT_PALETTE) -> np.ndarray:
    """Blinn-Phong 逐顶点颜色（环境 + 漫反射 + 补光 + 镜面高光）。

    Args:
        normals: (N, 3) 单位顶点法向量。
        palette: `_PALETTES` 中的配色名。

    Returns:
        (N, 3) RGB 颜色（0-1）。
    """
    base_color, spec_color, shininess, specular, use_sss = _PALETTES[palette]
    light = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)
    fill = _FILL_DIR / np.linalg.norm(_FILL_DIR)
    view = _VIEW_DIR / np.linalg.norm(_VIEW_DIR)
    half = (light + view) / np.linalg.norm(light + view)
    # wrap lighting：dot∈[-1,1] 重映射到 [wrap/(1+wrap), 1]，背光面不全黑、过渡柔和
    wrapped = np.clip((normals @ light + _WRAP) / (1 + _WRAP), 0, 1)
    fill_term = _FILL * np.clip((normals @ fill + 1) / 2, 0, 1)
    specular_term = np.clip(normals @ half, 0, 1) ** shininess
    diffuse_term = _AMBIENT + _DIFFUSE * wrapped + fill_term
    # SSS 近似：越暗处肤色越向血色混合（光线进入皮肤被血液散射后以红色出射）
    if use_sss:
        mix_t = _SSS_STRENGTH * (1 - wrapped)
        shaded_color = (1 - mix_t[:, None]) * base_color + mix_t[:, None] * _BLOOD
    else:
        shaded_color = base_color
    color = diffuse_term[:, None] * shaded_color + specular * specular_term[:, None] * spec_color
    return np.clip(color, 0, 1)


def render_back(
    ax: Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    palette: str = _DEFAULT_PALETTE,
) -> None:
    """光照底图（无 landmarks），白底。

    Args:
        ax: 目标坐标轴。
        vertices: (N, 3) 物理空间顶点坐标（x/y 作投影轴）。
        faces: (M, 3) 三角面索引。
        normals: (N, 3) 顶点法向量（单位向量）。
        palette: 配色名（`_PALETTES`）。
    """
    colors = compute_phong_colors(normals, palette)
    triangulation = Triangulation(vertices[:, 0], vertices[:, 1], faces)
    ax.add_collection(TriMesh(triangulation, facecolors=colors))
    ax.autoscale_view()
    _style_ax(ax, "Back")


def render_back_landmarks(
    ax: Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    flat: dict,
    palette: str = _DEFAULT_PALETTE,
) -> None:
    """光照底图 + landmarks 直线连接与地标点（标注平台配色）。

    Args:
        ax: 目标坐标轴。
        vertices: (N, 3) 物理空间顶点坐标（x/y 作投影轴）。
        faces: (M, 3) 三角面索引。
        normals: (N, 3) 顶点法向量（单位向量）。
        flat: 扁平 landmarks（18 键，物理坐标）。
        palette: 配色名（`_PALETTES`）。
    """
    render_back(ax, vertices, faces, normals, palette)
    ax.set_title("Landmarks", fontsize=11)
    _overlay_landmarks(ax, flat)


def _style_ax(ax: Axes, title: str) -> None:
    """白底黑字 + 等比例无坐标轴。"""
    ax.set_facecolor("white")
    ax.figure.patch.set_facecolor("white")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
