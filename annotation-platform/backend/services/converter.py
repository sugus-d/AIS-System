"""PLY → GLB 转换，笔刷擦除/恢复（body/cloth 顶点分类）。"""

import tempfile
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import KDTree

from ..constants import MESH_DIR, MESH_PROCESSED_DIR, PLATFORM_CACHE_DIR, ROI_DIR
from ._paths import _get_latest_edited

MIN_FACES = 3  # 构成有效网格所需最少面片数
EXPAND_MM = 3.0  # 恢复顶点 3mm 自动扩散半径

# ── Path helpers ─────────────────────────────────────


def _get_processed_path(subject_id: str) -> str | None:
    """找到 body mesh，优先级: edited → roi.ply → meshes_processed → 原始 mesh"""
    # 1) 最新 edited
    edited = _get_latest_edited(subject_id)
    if edited:
        return edited
    # 2) roi.ply
    p = ROI_DIR / subject_id / "roi.ply"
    if p.exists():
        return str(p)
    # 3) meshes_processed
    p = MESH_PROCESSED_DIR / f"{subject_id}_no_clothing.ply"
    if p.exists():
        return str(p)
    # 4) Fallback: original mesh
    return _get_clothed_path(subject_id)


def _get_clothed_path(subject_id: str) -> str | None:
    """找到有衣 mesh 路径（优先 STD_fuse_mesh → STD*_fuse_mesh → *_fuse_mesh → *.ply）。"""
    d = MESH_DIR / subject_id
    if not d.exists():
        return None
    ply = (sorted(d.glob("STD_fuse_mesh_*.ply"))
           or sorted(d.glob("STD*_fuse_mesh_*.ply"))
           or sorted(d.glob("*_fuse_mesh_*.ply"))
           or sorted(d.glob("*.ply")))
    return str(ply[0]) if ply else None


def _load(path: str) -> tuple[np.ndarray, np.ndarray]:
    """加载 PLY 文件，返回 (vertices, triangles)。"""
    m = o3d.io.read_triangle_mesh(str(path))
    return np.asarray(m.vertices, dtype=np.float32), np.asarray(m.triangles, dtype=np.uint32)


def _save_edited(tm: trimesh.Trimesh, subject_id: str) -> Path | None:
    """保存为新的 roi_edited_{ts}.ply（不覆盖旧文件）。"""
    d = PLATFORM_CACHE_DIR / subject_id / "extract_roi"
    d.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    out = d / f"roi_edited_{ts}.ply"
    tm.export(str(out), file_type="ply")
    return out


# ── Single-mesh GLB ──────────────────────────────────


def ply_to_glb_bytes(subject_id: str, clothed: bool = False) -> bytes:
    """加载 PLY mesh → 导出为 GLB bytes（供前端 3D 显示）。

    优先使用 edited 版本（笔刷编辑后的 mesh），
    clothed=True 时返回原始有衣 mesh。
    """
    if not clothed:
        edited = _get_latest_edited(subject_id)
        if edited:
            tm = trimesh.load(str(edited))
            if isinstance(tm, trimesh.Scene):
                meshes = [g for g in tm.geometry.values() if isinstance(g, trimesh.Trimesh)]
                tm = trimesh.util.concatenate(meshes) if meshes else tm
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=True) as tmp:
                tm.export(tmp.name, file_type="glb")
                return Path(tmp.name).read_bytes()
    p = _get_clothed_path(subject_id) if clothed else _get_processed_path(subject_id)
    if not p:
        raise FileNotFoundError(f"No mesh for {subject_id}")
    v, t = _load(p)
    tm = trimesh.Trimesh(vertices=v, faces=t)
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=True) as tmp:
        tm.export(tmp.name, file_type="glb")
        return Path(tmp.name).read_bytes()


# ── Cloth overlay (face-based, computed from classification) ──


def overlay_cloth_data(subject_id: str) -> dict:
    """返回布料顶点/面，用于前端红色 overlay 显示。

    Cloth = 原始 mesh 中不在当前 body 集合的顶点，
    且面片中所有 3 个顶点都在布料中。
    Body = 最新 edited.ply（或原始 output.ply）。
    """
    body_path = _get_latest_edited(subject_id) or _get_processed_path(subject_id)
    if not body_path:
        return {"error": "missing mesh"}
    body_v, _ = _load(body_path)

    cloth_path = _get_clothed_path(subject_id)
    if not cloth_path:
        return {"error": "missing cloth mesh"}
    orig_v, orig_t = _load(cloth_path)

    # 判断原始顶点中哪些在 body 集合中
    tree = KDTree(orig_v)
    _, body_orig_idx = tree.query(body_v, k=1)
    is_body = np.zeros(len(orig_v), dtype=bool)
    is_body[body_orig_idx] = True

    # 所有 3 个顶点都不是 body 的面片 = 布料面片
    cloth_face_mask = np.all(~is_body[orig_t], axis=1)
    cloth_t = orig_t[cloth_face_mask]

    if len(cloth_t) < MIN_FACES:
        return {"extra_points": [], "extra_indices": [], "faces": []}

    used_verts = np.unique(cloth_t)
    old2new = {int(old): new for new, old in enumerate(used_verts)}
    cloth_v_subset = orig_v[used_verts]
    cloth_t_subset = np.array([[old2new[int(v)] for v in face] for face in cloth_t])

    return {
        "extra_points": cloth_v_subset.tolist(),
        "extra_indices": [int(v) for v in used_verts],
        "faces": cloth_t_subset.tolist(),
    }


# ── Brush commit (vertex classification) ─────────────


def brush_commit(
    subject_id: str,
    points: list[list[float]] | None = None,
    cloth_indices: list[int] | None = None,
) -> dict:
    """笔刷擦除 + 恢复，通过 body/cloth 顶点分类实现。

    **擦除**：笔刷标记的顶点 → 从 body 集合移除。
    **恢复**：cloth_indices → 加入 body 集合 + 3mm 邻居扩散。
    两者都基于原始 mesh 三角化重建新的 body.ply。

    Args:
        subject_id: subject ID
        points: body 顶点 3D 坐标 [[x,y,z], ...]，标记为擦除（可选）
        cloth_indices: 原始 mesh 顶点索引，标记为恢复（可选）
    """
    if not points and not cloth_indices:
        return {"error": "no points or cloth indices provided"}

    # 1. 加载当前 body mesh（包含之前的所有编辑）
    body_path = _get_latest_edited(subject_id) or _get_processed_path(subject_id)
    if not body_path:
        return {"error": "no mesh"}
    body_v, _ = _load(body_path)

    # 2. 加载原始完整 mesh（STD_fuse_mesh）
    cloth_path = _get_clothed_path(subject_id)
    if not cloth_path:
        return {"error": "no cloth mesh"}
    orig_v, orig_t = _load(cloth_path)

    # 3. 将 body 顶点映射到原始 mesh 索引
    tree = KDTree(orig_v)
    _, body_orig_idx = tree.query(body_v, k=1)
    is_body = np.zeros(len(orig_v), dtype=bool)
    is_body[body_orig_idx] = True

    # 4. 擦除：标记笔刷覆盖的顶点为 false
    if points:
        tree_body = KDTree(body_v)
        hit_count = 0
        for pt in points:
            dist, bidx = tree_body.query(np.array(pt, dtype=np.float32))
            if dist <= 1.0:
                oidx = int(body_orig_idx[bidx])
                if is_body[oidx]:
                    is_body[oidx] = False
                    hit_count += 1
        if hit_count == 0:
            return {"error": "no vertices matched — coordinates may be stale"}

    # 5. 恢复：布料顶点加入 body + 3mm 自动扩散
    if cloth_indices:
        for idx in cloth_indices:
            if idx < len(orig_v):
                is_body[int(idx)] = True
        # 3mm 自动扩散：距恢复点 3mm 内的顶点也标记为 body
        restore_verts = orig_v[list(cloth_indices)]
        tree_restore = KDTree(restore_verts)
        dists, _ = tree_restore.query(orig_v, k=1)
        expansion = dists <= EXPAND_MM
        is_body[expansion] = True

    # 6. 从 is_body 掩码重建 body mesh
    all_body_faces = orig_t[np.all(is_body[orig_t], axis=1)]
    if len(all_body_faces) < MIN_FACES:
        return {"error": "mesh destroyed beyond recovery"}

    used_verts = np.unique(all_body_faces)
    old2new = {int(old): new for new, old in enumerate(used_verts)}
    body_v_out = orig_v[used_verts]
    body_t_out = np.array([[old2new[int(v)] for v in face] for face in all_body_faces])

    result_tm = trimesh.Trimesh(vertices=body_v_out, faces=body_t_out)
    result_tm = trimesh.Trimesh(
        vertices=np.asarray(result_tm.vertices, dtype=np.float32),
        faces=np.asarray(result_tm.faces, dtype=np.uint32),
    )

    # 7. 仅保留最大连通分量（擦除可能切出多个孤立块）
    try:
        components = result_tm.split(only_watertight=False)
        if len(components) > 1:
            largest = max(components, key=lambda m: len(m.vertices))
            result_tm = trimesh.Trimesh(
                vertices=np.asarray(largest.vertices, dtype=np.float32),
                faces=np.asarray(largest.faces, dtype=np.uint32),
            )
    except Exception:
        pass  # split 失败时保留原样

    # 8. 保存
    out = _save_edited(result_tm, subject_id)
    if out is None:
        return {"error": "failed to save mesh"}
    return {"status": "ok", "file": str(out), "vert_count": len(result_tm.vertices)}

