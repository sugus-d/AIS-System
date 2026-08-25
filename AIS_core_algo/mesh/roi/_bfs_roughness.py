"""BFS roughness pre-computation — fully vectorized per-triangle roughness."""

import numpy as np
import open3d as o3d

from utils.logger import logger

_NORM_EPSILON = 1e-10      # 平均法线长度小于该值视为退化（不归一化）
_MIN_FACE_COUNT = 3        # 顶点至少关联 3 个三角面（否则粗糙度置零）


def scatter_face_values_to_vertices(
    faces: np.ndarray,
    face_values: np.ndarray,
    n_vertices: int,
) -> np.ndarray:
    """per-face 标量 → 顶点（散射累加取均值）。

    每个顶点取其关联三角面值的均值（面值 scatter 到 3 个角点后除以关联面数），
    未关联任何面的顶点保持 0。纯 numpy 向量化，N>100k 无 Python 循环。
    特征提取（assemble）与报告热力图（prediction.measures）共用。

    Args:
        faces: (F, 3) 三角面顶点索引。
        face_values: (F,) 每面一个标量。
        n_vertices: 顶点总数。

    Returns:
        (n_vertices,) 顶点标量（关联面的均值）。
    """
    vertex_sum = np.zeros(n_vertices, dtype=np.float64)
    np.add.at(vertex_sum, faces.ravel(), np.repeat(face_values, 3))
    face_count = np.zeros(n_vertices, dtype=np.float64)
    np.add.at(face_count, faces.ravel(), 1)
    return vertex_sum / np.maximum(face_count, 1)


def _compute_face_normals(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Fully vectorized face normals — no open3d clone."""
    v0, v1, v2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    return fn


def compute_mesh_roughness(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Pre‑compute per‑triangle roughness, fully vectorised.

    Strategy (vertex roughness → max‑pool to triangles):
      1. Compute face normals via numpy cross‑product (vectorised).
      2. Per **vertex**: scatter‑add face normals → mean normal →
         scatter‑max angular deviation of incident faces.
         (Fully vectorised — no Python loop over 850k vertices.)
      3. Per **triangle**: max of its 3 vertex roughnesses (numpy, 0‑copy).

    Returns
    -------
    roughness : ndarray, shape (n_tri,) — radians.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    t = np.asarray(mesh.triangles)
    n_vert = len(v)

    # ── 1. Face normals (fully vectorised) ──
    fn = _compute_face_normals(v, t)

    # ── 2. Per‑vertex mean normal (scatter‑add, fully vectorised) ──
    fn_sum = np.zeros((n_vert, 3), dtype=np.float64)
    for c in range(3):
        np.add.at(fn_sum[:, c], t[:, 0], fn[:, c])
        np.add.at(fn_sum[:, c], t[:, 1], fn[:, c])
        np.add.at(fn_sum[:, c], t[:, 2], fn[:, c])

    face_count = np.bincount(t.ravel(), minlength=n_vert)
    mean_norm = fn_sum / np.maximum(face_count[:, None], 1)
    norm = np.linalg.norm(mean_norm, axis=1)
    good = norm > _NORM_EPSILON
    if good.any():
        mean_norm[good] /= norm[good, None]

    # ── 3. Per‑vertex roughness (scatter‑max, fully vectorised) ──
    dot0 = np.sum(fn * mean_norm[t[:, 0]], axis=1)
    dot1 = np.sum(fn * mean_norm[t[:, 1]], axis=1)
    dot2 = np.sum(fn * mean_norm[t[:, 2]], axis=1)

    np.clip(dot0, -1.0, 1.0, out=dot0)
    np.clip(dot1, -1.0, 1.0, out=dot1)
    np.clip(dot2, -1.0, 1.0, out=dot2)

    vert_rough = np.zeros(n_vert, dtype=np.float64)
    np.maximum.at(vert_rough, t[:, 0], np.arccos(dot0))
    np.maximum.at(vert_rough, t[:, 1], np.arccos(dot1))
    np.maximum.at(vert_rough, t[:, 2], np.arccos(dot2))
    vert_rough[face_count < _MIN_FACE_COUNT] = 0.0

    # ── 4. Triangle‑level → max of its 3 vertices (0‑copy) ──
    roughness = np.maximum(vert_rough[t[:, 0]], vert_rough[t[:, 1]])
    roughness = np.maximum(roughness, vert_rough[t[:, 2]])

    logger.info(
        "Roughness(vert‑max): range=[%.4f, %.4f] rad, median=%.4f",
        float(roughness.min()),
        float(roughness.max()),
        float(np.median(roughness)),
    )
    return roughness
