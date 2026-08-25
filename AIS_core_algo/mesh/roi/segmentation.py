"""ROI 模板分割 — 将背部 ROI 网格按模板分区并分配左右侧。

segment_template 按模板 landmark 分区，_assign_sides 沿脊柱中线分配左右，
供特征提取的区域特征与可视化使用。
"""

import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

# Segment ids
SEG_SHOULDER = 2
SEG_THORACIC = 0
SEG_LUMBAR = 1
SEG_PELVIC = 3

# Y-fraction boundaries (fraction of normalised height 0-1)
_SHOULDER_THRESH = 0.78  # > 78 % -> shoulder
_LUMBAR_THRESH = 0.42  # 18-42 % -> lumbar
_PELVIC_THRESH = 0.18  # < 18 % -> pelvic


def segment_template(
    template: o3d.geometry.TriangleMesh,
    spine_midline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment the back mesh into anatomical regions.

    Uses Y-coordinate ranges (relative to mesh bounding box) for the
    four anatomical segments and the spine midline X-coordinates to
    split each segment into left / right sub-regions.

    Segments:
      0 = thoracic  (T1-T12, upper back below shoulder line)
      1 = lumbar    (L1-L5, mid-lower back)
      2 = shoulder  (above thoracic)
      3 = pelvic    (hip / sacral region)

    Args:
        template: Open3D TriangleMesh (registered or aligned back mesh).
        spine_midline: np.ndarray (M, 3) -- spinal midline points.

    Returns:
        labels: np.ndarray (N,) int -- segment id per vertex.
        sides:  np.ndarray (N,) int -- 0 = left, 1 = right.
    """
    vertices = np.asarray(template.vertices, dtype=np.float64)
    N = len(vertices)

    # 将 Y 坐标归一化到 [0, 1]，基于 mesh 包围盒高度
    y = vertices[:, 1]
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min if y_max > y_min else 1.0
    y_norm = (y - y_min) / y_range  # 0..1

    # 按 Y 归一化阈值划分解剖区域（从高到低：肩→胸→腰→骨盆）
    labels = np.full(N, SEG_THORACIC, dtype=np.int32)
    labels[y_norm > _SHOULDER_THRESH] = SEG_SHOULDER
    labels[y_norm < _PELVIC_THRESH] = SEG_PELVIC
    labels[(y_norm >= _PELVIC_THRESH) & (y_norm <= _LUMBAR_THRESH)] = SEG_LUMBAR

    # 通过脊柱中线将每个区域分为左右两侧
    sides = _assign_sides(vertices, spine_midline)

    return labels, sides


def _assign_sides(vertices: np.ndarray, spine_midline: np.ndarray) -> np.ndarray:
    """For each vertex, determine left (0) or right (1) relative to the spine.

    We project each vertex onto the Y axis, find the nearest spine point
    by Y coordinate, then compare vertex X to spine X at that Y.
    """
    # 脊柱线为空时退化为全局 X 均值分割
    if spine_midline is None or len(spine_midline) == 0:
        mid_x = vertices[:, 0].mean()
        return (vertices[:, 0] > mid_x).astype(np.int32)

    # 用 KDTree 按 Y 坐标查找每个顶点对应的脊柱点
    # 取脊柱线 Y 坐标做 1D KDTree，对每个顶点 Y 找最近的脊柱点
    tree = KDTree(spine_midline[:, 1:2])  # 1D KDTree on Y
    _, idx = tree.query(vertices[:, 1:2])
    spine_x_at_y = spine_midline[idx, 0]

    # 惯例：从背后观察，顶点 X < 脊柱 X 为左侧（0），反之为右侧（1）
    sides = (vertices[:, 0] > spine_x_at_y).astype(np.int32)
    return sides
