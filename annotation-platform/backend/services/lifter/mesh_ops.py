"""Mesh 加载、PCA 投影与顶点吸附 —— lifter 几何操作层。"""

from pathlib import Path

import numpy as np

from ...constants import MESH_DIR, MESH_PROCESSED_DIR, ROI_DIR
from .._paths import _get_latest_edited
from .constants import COORD_DIM_3D, MIN_PROJECT_COORDS, PAIR_SIDES, SNAP_DIST_MM


def _load_subject_mesh(subject_id: str) -> tuple[object, np.ndarray]:
    """加载 subject 的 body mesh，返回 (open3d_mesh, vertices_ndarray)。

    搜索顺序：edited PLY → roi.ply → meshes_processed → 原始 mesh。
    """
    import open3d as o3d

    def _read(path: str) -> tuple:
        m = o3d.io.read_triangle_mesh(path)
        return m, np.asarray(m.vertices, dtype=np.float64)

    edited = _get_latest_edited(subject_id)
    if edited and Path(edited).exists():
        return _read(edited)
    p = ROI_DIR / subject_id / "roi.ply"
    if p.exists():
        return _read(str(p))
    p = MESH_PROCESSED_DIR / f"{subject_id}_no_clothing.ply"
    if p.exists():
        return _read(str(p))
    d = MESH_DIR / subject_id
    if d.exists():
        ply = (sorted(d.glob("STD_fuse_mesh_*.ply"))
               or sorted(d.glob("STD*_fuse_mesh_*.ply"))
               or sorted(d.glob("*_fuse_mesh_*.ply"))
               or sorted(d.glob("*.ply")))
        if ply:
            return _read(str(ply[0]))
    raise FileNotFoundError(f"找不到 subject {subject_id} 的 body mesh")


def _landmarks_to_pca(landmarks: dict, subject_id: str) -> dict:
    """将 3D 坐标 landmark 投影到 PCA 2D 空间（与 curvature image 共享同一旋转矩阵）。
    返回格式: {group_name: [[pc2_x, pc1_y, z], ...], ...}
    """
    from ..curvature import _get_pca_params, _pca_transform

    try:
        pca_params = _get_pca_params(subject_id)
    except Exception:
        return landmarks
    result: dict = {}
    for name, pts in landmarks.items():
        projected = []
        for pt in pts:
            if not isinstance(pt, (list, np.ndarray)) or len(pt) < MIN_PROJECT_COORDS:
                projected.append(pt)
                continue
            p3d = np.array([pt[0], pt[1], pt[2]] if len(pt) >= COORD_DIM_3D else [pt[0], pt[1], 0.0])
            rotated = _pca_transform(p3d.reshape(1, 3), pca_params)[0]
            # 与 curvature image 一致：PC2→X, PC1→Y
            projected.append([float(rotated[1]), float(rotated[0]), float(p3d[2])])
        result[name] = projected
    return result


def _validate_landmarks_on_mesh(subject_id: str, landmarks: dict) -> dict[str, list]:
    """校验 landmark 点是否在 body mesh 顶点上。

    如果某点距最近顶点超过 100mm，标记为未放置（设为 None），否则吸附到最近顶点。
    """
    from scipy.spatial import KDTree

    try:
        _, verts = _load_subject_mesh(subject_id)
    except FileNotFoundError:
        return landmarks
    tree = KDTree(verts)

    for _name, pts in landmarks.items():
        for i, pt in enumerate(pts):
            if pt is None:
                continue
            if not isinstance(pt, (list, np.ndarray)) or len(pt) < COORD_DIM_3D:
                pts[i] = None
                continue
            dist, idx = tree.query([float(pt[0]), float(pt[1]), float(pt[2])], k=1)
            if dist <= SNAP_DIST_MM:
                pts[i] = verts[idx].tolist()  # 吸附到最近顶点
            else:
                pts[i] = None  # 距表面太远，标记未放置
    return landmarks


def _auto_detect_waist_lower(subject_id: str, landmarks: dict) -> list | None:
    """将 body mesh 投影到 PCA 空间，在腰部下方找最左/最右顶点作为腰下缘 L/R。"""
    try:
        _, verts = _load_subject_mesh(subject_id)
    except FileNotFoundError:
        return None

    # PCA 投影到 2D
    try:
        from ..curvature import _get_pca_params

        pca_params = _get_pca_params(subject_id)
    except Exception:
        return None
    mean = np.array(pca_params["pca_mean"], dtype=np.float64)
    Vt = np.array(pca_params["pca_Vt"], dtype=np.float64)
    rotated = (verts - mean) @ Vt.T  # (N,3): PC1≈Y, PC2≈X
    pca_x = rotated[:, 1]  # PC2→X
    pca_y = rotated[:, 0]  # PC1→Y

    # 用 waist 作参考
    waist = landmarks.get("waist", [])
    if len(waist) >= PAIR_SIDES and waist[0] and waist[1]:
        waist_center = [
            (waist[0][0] + waist[1][0]) / 2,
            (waist[0][1] + waist[1][1]) / 2,
            (waist[0][2] + waist[1][2]) / 2,
        ]
        w_rot = (np.array(waist_center, dtype=np.float64) - mean) @ Vt.T
        waist_y = w_rot[0]  # PC1
    else:
        waist_y = float(np.percentile(pca_y, 30))  # fallback

    min_y = float(pca_y.min())
    # 从 waist 往下 30%～60% 区间搜索
    lower_bound = waist_y + (min_y - waist_y) * 0.3
    upper_bound = waist_y + (min_y - waist_y) * 0.6
    mask = (pca_y >= lower_bound) & (pca_y <= upper_bound)
    candidates = verts[mask]
    cand_x = pca_x[mask]

    if len(candidates) == 0:
        return None

    # 选取最左 / 最右顶点作为腰下缘 L/R
    left_idx = np.argmin(cand_x)
    right_idx = np.argmax(cand_x)
    return [candidates[left_idx].tolist(), candidates[right_idx].tolist()]
