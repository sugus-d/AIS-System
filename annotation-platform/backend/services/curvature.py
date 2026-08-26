"""曲率热力图渲染为 PNG + 像素→mm 映射。PCA 投影 + edited mesh 支持。"""

import io
import json
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.spatial import KDTree

from ..constants import CACHE_DIR, MESH_DIR, MESH_PROCESSED_DIR, ROI_DIR
from ._paths import _find_algorithm_dir, _get_latest_edited

CURV_IMG_DIR: Path = CACHE_DIR / "curvature_images"


def _get_processed_path(subject_id: str) -> str | None:
    """找到 body mesh，搜索顺序同 converter._get_processed_path。"""
    edited = _get_latest_edited(subject_id)
    if edited:
        return edited
    # AIS 算法输出 roi.ply（prediction-outputs/<sid>-*/）
    algo_dir = _find_algorithm_dir(subject_id)
    if algo_dir:
        roi = algo_dir / "roi.ply"
        if roi.exists():
            return str(roi)
    p = ROI_DIR / subject_id / "roi.ply"
    if p.exists():
        return str(p)
    p = MESH_PROCESSED_DIR / f"{subject_id}_no_clothing.ply"
    if p.exists():
        return str(p)
    # Fallback: original mesh（优先 STD_fuse_mesh → STD*_fuse_mesh → *_fuse_mesh → *.ply）
    d = MESH_DIR / subject_id
    if not d.exists():
        return None
    ply = (sorted(d.glob("STD_fuse_mesh_*.ply"))
           or sorted(d.glob("STD*_fuse_mesh_*.ply"))
           or sorted(d.glob("*_fuse_mesh_*.ply"))
           or sorted(d.glob("*.ply")))
    return str(ply[0]) if ply else None


def _load_mesh_curvature(subject_id: str) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
    """加载 mesh + 曲率，优先 edited 版本。

    无缓存时自动计算曲率并保存，避免重复计算。
    """
    curv_path = CACHE_DIR / subject_id / "curvature" / "mean_curvature.npy"

    def _compute_and_save(path: str) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        from mesh.curvature import compute_mean_curvature
        mesh = o3d.io.read_triangle_mesh(path)
        v = np.asarray(mesh.vertices, dtype=np.float64)
        t = np.asarray(mesh.triangles, dtype=np.int32)
        curvature = compute_mean_curvature(v, t)
        curv_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(curv_path), curvature)
        return mesh, curvature

    if not curv_path.exists():
        pp = _get_latest_edited(subject_id) or _get_processed_path(subject_id)
        if not pp:
            raise FileNotFoundError(f"No processed mesh for {subject_id}")
        return _compute_and_save(pp)

    orig_curvature = np.load(str(curv_path))

    def _save_curvature(curvature: np.ndarray) -> None:
        curv_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(curv_path), curvature)

    def _load_mesh(path: str) -> tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        mesh = o3d.io.read_triangle_mesh(path)
        v = np.asarray(mesh.vertices, dtype=np.float32)
        if len(v) == len(orig_curvature):
            return mesh, orig_curvature
        steps = ("load_mesh", "unknown")
        for step in steps:
            ref_p = CACHE_DIR / subject_id / step / "output.ply"
            if ref_p.exists():
                ref_mesh = o3d.io.read_triangle_mesh(str(ref_p))
                ref_v = np.asarray(ref_mesh.vertices, dtype=np.float32)
                if len(ref_v) == len(orig_curvature):
                    tree = KDTree(ref_v)
                    _, idx = tree.query(v, k=1)
                    mapped = orig_curvature[idx]
                    _save_curvature(mapped)
                    return mesh, mapped
        from mesh.curvature import compute_mean_curvature

        new_curv = compute_mean_curvature(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.triangles, dtype=np.int32),
        )
        _save_curvature(new_curv)
        return mesh, new_curv

    edited = _get_latest_edited(subject_id)
    if edited:
        mesh, curvature = _load_mesh(edited)
    else:
        pp = _get_processed_path(subject_id)
        if pp is None:
            raise FileNotFoundError(f"No processed mesh for {subject_id}")
        mesh, curvature = _load_mesh(pp)
    return mesh, curvature


# ── 统一的 PCA 参数 —— 所有投影（curvature image / landmarks / contours）共享同一个旋转矩阵 ──


def _get_pca_params(subject_id: str) -> dict:
    """加载或计算 PCA 参数（mean, Vt，含方向修正）。

    优先从缓存 mapping.json 读取。如果 edited mesh 比缓存新，自动触发重建。
    """
    mapping_path = CURV_IMG_DIR / subject_id / "mapping.json"
    # 检查 edited mesh 是否比缓存新（与 render_curvature_image 的 stale 检测一致）
    if mapping_path.exists():
        edited = _get_latest_edited(subject_id)
        if edited and mapping_path.stat().st_mtime < Path(edited).stat().st_mtime:
            mapping_path.unlink(missing_ok=True)
    # 未缓存或已过期 → 先渲染 image（会生成 mapping.json）
    if not mapping_path.exists():
        render_curvature_image(subject_id)
    with open(mapping_path) as f:
        return json.load(f)


def _compute_pca_params(vertices: np.ndarray) -> dict[str, list[list[float]]]:
    """计算 PCA 参数（mean, Vt），含 PC1/PC2 方向修正。返回适合存 mapping 的 dict。"""
    mean = vertices.mean(axis=0)
    centered = vertices - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    # 修正 PC1 方向：确保 Y 分量向上（头朝上、脚朝下）
    if Vt[0, 1] < 0:
        Vt[0] = -Vt[0]
    # 修正 PC2 方向：确保 X 分量向右（人体右侧为正 X）
    if Vt[1, 0] < 0:
        Vt[1] = -Vt[1]
    return {"pca_mean": mean.tolist(), "pca_Vt": Vt.tolist()}


def _pca_transform(vertices: np.ndarray, pca_params: dict) -> np.ndarray:
    """用缓存的 PCA 参数旋转顶点，返回 (N, 3) 旋转后坐标。"""
    mean = np.array(pca_params["pca_mean"], dtype=np.float64)
    Vt = np.array(pca_params["pca_Vt"], dtype=np.float64)
    return (vertices - mean) @ Vt.T


# ── Curvature image ──────────────────────────────────────


def render_curvature_image(subject_id: str) -> tuple[bytes, dict]:
    """渲染曲率热力图（PCA 旋转）到 PNG bytes + mapping dict。"""
    img_dir = CURV_IMG_DIR / subject_id
    png_path = img_dir / "curvature.png"
    mapping_path = img_dir / "mapping.json"

    # 如果 edited mesh 比缓存新，删除旧缓存重建
    if png_path.exists() and mapping_path.exists():
        edited = _get_latest_edited(subject_id)
        if edited and png_path.stat().st_mtime < Path(edited).stat().st_mtime:
            png_path.unlink(missing_ok=True)
            mapping_path.unlink(missing_ok=True)

    # 直接读取已缓存的圖像
    if png_path.exists() and mapping_path.exists():
        with open(png_path, "rb") as f:
            png_bytes = f.read()
        with open(mapping_path) as f:
            mapping = json.load(f)
        return png_bytes, mapping

    mesh, curvature = _load_mesh_curvature(subject_id)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles)

    pca_params = _compute_pca_params(vertices)
    rotated = _pca_transform(vertices, pca_params)
    # PC1→Y（竖直），PC2→X（水平）
    x_vals, y_vals = rotated[:, 1], rotated[:, 0]

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    pad = 0.05
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_lo, x_hi = x_min - pad * x_range, x_max + pad * x_range
    y_lo, y_hi = y_min - pad * y_range, y_max + pad * y_range

    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    triang = mtri.Triangulation(x_vals, y_vals, triangles)
    ax.tripcolor(
        triang,
        curvature,
        cmap="jet",
        shading="gouraud",
        vmin=-0.03,
        vmax=0.03,
        alpha=0.6,
    )
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="#1a1a1a", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    mapping = {
        "x_data_range": [float(x_lo), float(x_hi)],
        "y_data_range": [float(y_lo), float(y_hi)],
        **pca_params,
    }

    img_dir.mkdir(parents=True, exist_ok=True)
    with open(png_path, "wb") as f:
        f.write(png_bytes)
    # 先写临时文件再 replace（原子覆盖，Windows 下 rename 遇已存在文件会抛 FileExistsError）
    tmp = mapping_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(mapping, f)
    tmp.replace(mapping_path)
    return png_bytes, mapping


def _pca_project(subject_id: str, vertices: np.ndarray) -> np.ndarray:
    """用统一的 PCA 参数旋转顶点，取 PC2→X、PC1→Y 的 2D 坐标。"""
    pca_params = _get_pca_params(subject_id)
    rotated = _pca_transform(vertices, pca_params)
    return rotated[:, [1, 0]]


def get_contours(subject_id: str) -> dict:
    """提取左右轮廓线 2D 坐标（统一 PCA 投影），供前端显示。"""
    from landmarks.lateral_profile import extract_split_contours

    mesh, _ = _load_mesh_curvature(subject_id)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    pca_2d = _pca_project(subject_id, vertices)

    left_c, right_c = extract_split_contours(pca_2d)
    return {
        "left": np.asarray(left_c)[:, :2].tolist() if left_c is not None and len(left_c) > 0 else [],
        "right": np.asarray(right_c)[:, :2].tolist() if right_c is not None and len(right_c) > 0 else [],
    }
