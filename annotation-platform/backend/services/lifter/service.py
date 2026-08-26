"""Landmark CRUD、2D→3D 提升与提交校验 —— lifter 服务接口层。"""

import json
import pickle

import numpy as np

from ...constants import BILATERAL_LANDMARKS, CACHE_DIR, GT_DIR, SPINE_POINT_COUNT
from ...utils.logger import logger
from .._paths import load_algorithm_landmarks
from .constants import COORD_DIM_3D, PAIR_SIDES, SNAP_DIST_MM, SPINE_SOURCE_PAIRS
from .mesh_ops import _auto_detect_waist_lower, _load_subject_mesh, _validate_landmarks_on_mesh


def load_landmarks(subject_id: str) -> dict:
    """优先返回 GT 标注，无则返回算法检测值。

    格式: {"neck_root": [[Lx,Ly,Lz], [Rx,Ry,Rz]], ...}
    自动补全缺失的 waist_lower（定位到 body mesh 左下角/右下角）和 spine_points。
    """
    # 尝试读取 GT 文件
    gt_file = GT_DIR / subject_id / "ground_truth.json"
    if gt_file.exists():
        gt = json.loads(gt_file.read_text())
        result = {k: v for k, v in gt.items() if not k.startswith("_")}
        out: dict = {}
        for name, val in result.items():
            if isinstance(val, dict):
                # bilateral 格式: {"L": ..., "R": ...}，可能只有一侧
                out[name] = [val.get("L"), val.get("R")]
            else:
                out[name] = val
    else:
        # 从算法缓存加载（旧 pkl 布局）
        lm_file = CACHE_DIR / subject_id / "landmarks" / "landmarks.pkl"
        if lm_file.exists():
            data = pickle.loads(lm_file.read_bytes())
            keys = [
                "neck_root",
                "shoulder_transition",
                "scapular_peaks",
                "axilla",
                "waist",
                "waist_lower",
                "spine_points",
            ]
            out = {}
            for k in keys:
                if k in data:
                    arr = np.asarray(data[k])
                    out[k] = arr.tolist()
        else:
            # 从 AIS 算法输出加载（新布局 landmarks.json）
            out = load_algorithm_landmarks(subject_id)

    # 校验所有 landmark：吸附到 body mesh 最近顶点，远距点标记为未放置
    try:
        out = _validate_landmarks_on_mesh(subject_id, out)
    except Exception as exc:
        logger.warning("landmarks validation failed for {}: {}", subject_id, exc)

    # 自动补全 waist_lower
    if "waist_lower" not in out or not out["waist_lower"]:
        try:
            wl = _auto_detect_waist_lower(subject_id, out)
            if wl:
                out["waist_lower"] = wl
        except Exception as exc:
            logger.warning("waist_lower auto-detect failed for {}: {}", subject_id, exc)

    # 计算脊柱中点：从双边 landmark 补全 None 的 P 点（不覆盖已有值）
    spine = out.get("spine_points", [])
    while len(spine) < SPINE_POINT_COUNT:
        spine.append(None)
    for i, pair_name in enumerate(SPINE_SOURCE_PAIRS):
        if i >= len(spine):
            break
        if spine[i] is None:
            pair = out.get(pair_name)
            if pair and len(pair) >= PAIR_SIDES and pair[0] and pair[1]:
                spine[i] = [(pair[0][0] + pair[1][0]) / 2, (pair[0][1] + pair[1][1]) / 2, (pair[0][2] + pair[1][2]) / 2]
    # 未填充的保持 None
    spine = [s if s is not None else None for s in spine]
    out["spine_points"] = spine
    return out


def save_landmarks(subject_id: str, landmarks: dict) -> None:
    """保存 GT JSON，保留已有 _features，自动计算标注状态。

    支持部分 bilateral（L-only / R-only 也可保存，不会因 len<2 静默丢弃）。
    只写入一次 GT 文件（构建完整 dict 后再序列化）。
    """
    gt_dir = GT_DIR / subject_id
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_file = gt_dir / "ground_truth.json"
    existing = {}
    if gt_file.exists():
        existing = json.loads(gt_file.read_text())

    # 构建完整 gt dict（保留旧 _features，覆盖 landmark 数据）
    gt = {}
    for k, v in existing.items():
        if not k.startswith("_"):
            continue
        gt[k] = v

    for name, pts in landmarks.items():
        if name.startswith("_"):
            continue
        if name == "spine_points":
            gt[name] = pts
        elif isinstance(pts, list):
            pair: dict[str, list[float]] = {}
            if len(pts) >= 1 and pts[0] is not None:
                pair["L"] = pts[0]  # type: ignore[assignment]
            if len(pts) >= PAIR_SIDES and pts[1] is not None:
                pair["R"] = pts[1]  # type: ignore[assignment]
            if pair:
                gt[name] = pair

    # 计算标注状态
    total_expected = len(BILATERAL_LANDMARKS) * 2 + SPINE_POINT_COUNT
    completed = 0
    for name in BILATERAL_LANDMARKS:
        pair = gt.get(name, {})
        if isinstance(pair, dict):
            if pair.get("L") is not None:
                completed += 1
            if pair.get("R") is not None:
                completed += 1
    spine = gt.get("spine_points", [])
    completed += sum(1 for pt in spine if pt is not None)
    if completed >= total_expected:
        features = {"labeling_status": "labeled"}
    elif completed > 0:
        features = {"labeling_status": "prelabeled"}
    else:
        features = {"labeling_status": "unlabeled"}
    # 合并新的标注状态到已有 _features（保留 arms、body_asymmetry 等字段）
    existing_features = gt.get("_features", {})
    existing_features["labeling_status"] = features["labeling_status"]
    gt["_features"] = existing_features

    # 单次写入
    gt_file.write_text(json.dumps(gt, indent=2, ensure_ascii=False) + "\n")


def reset_landmarks(subject_id: str) -> dict:
    """删除 GT 文件，返回算法自动检测的坐标。"""
    gt_file = GT_DIR / subject_id / "ground_truth.json"
    if gt_file.exists():
        gt_file.unlink()
    return load_landmarks(subject_id)


def lift_2d_to_3d(subject_id: str, x_2d: float, y_2d: float) -> list:
    """2D 数据坐标 → 沿 PCA 垂线（PC3 方向）与 mesh 表面相交 → 最近顶点。

    原始 2D 最近邻搜索丢弃了 PC3（深度），导致前后表面混叠。
    改为沿 PC3 方向做 ray casting，取最近交点后再吸附到最近顶点，
    保证吸附前后 2D 投影坐标基本不变。

    如果点击在图片轮廓之外（ray casting 交点到远侧），
    回退到 PCA 2D 空间最近邻搜索，保证 ROI 外的点击就近吸附。
    """
    import open3d as o3d
    from scipy.spatial import KDTree

    # 优先加载 edited mesh（与 curvature image / 3D 显示一致）
    mesh, vertices = _load_subject_mesh(subject_id)

    # 从 curvature service 获取 PCA 参数（含 staleness 检查）
    from ..curvature import _get_pca_params

    try:
        mapping = _get_pca_params(subject_id)
        pca_mean = mapping.get("pca_mean")
        pca_Vt = mapping.get("pca_Vt")
    except Exception:
        pca_mean = pca_Vt = None

    if pca_mean is not None and pca_Vt is not None:
        mean = np.array(pca_mean, dtype=np.float64)
        Vt = np.array(pca_Vt, dtype=np.float64)  # (3, 3)

        # ── 方案 1：沿 PC3 射线求交（点击在轮廓内时最精确） ──
        # 构造射线：过 (PC2=x_2d, PC1=y_2d, PC3=0) 沿 PC3 轴方向
        # 射线起点由 PCA 均值沿 PC1 与 PC2 方向偏移得到
        ray_origin = mean + y_2d * Vt[0] + x_2d * Vt[1]
        ray_dir = Vt[2]  # PC3 轴方向（单位向量，Vt 正交）

        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(t_mesh)

        hit_points_3d = []
        for direction in (ray_dir, -ray_dir):
            rays = np.array(
                [[ray_origin[0], ray_origin[1], ray_origin[2], direction[0], direction[1], direction[2]]],
                dtype=np.float32,
            )
            result = scene.cast_rays(o3d.core.Tensor(rays))
            t_hit = float(result["t_hit"].item())
            if np.isfinite(t_hit):
                hit_points_3d.append(ray_origin + direction * t_hit)

        if hit_points_3d:
            best_hit = min(hit_points_3d, key=lambda p: np.linalg.norm(p - ray_origin))
            tree = KDTree(vertices)
            _, idx = tree.query(best_hit, k=1)
            lifted = vertices[idx]

            # 验证：将 lift 结果投影回 PCA 2D，距离原始点击 < 100mm 才接受
            projected = (lifted - mean) @ Vt.T  # (3,): PC1≈Y, PC2≈X, PC3≈Z
            proj_dx = projected[1] - x_2d  # PC2→X
            proj_dy = projected[0] - y_2d  # PC1→Y
            if np.sqrt(proj_dx * proj_dx + proj_dy * proj_dy) < SNAP_DIST_MM:
                return lifted.tolist()

        # ── 方案 2：PCA 2D 最近邻（点击在轮廓外时就近吸附） ──
        rotated = (vertices - mean) @ Vt.T  # (N, 3): PC1≈Y, PC2≈X, PC3≈Z
        pca_2d = np.column_stack([rotated[:, 1], rotated[:, 0]])  # (N, 2): [PC2, PC1]
        tree = KDTree(pca_2d)
        _, idx = tree.query([[x_2d, y_2d]], k=1)
        return vertices[idx[0]].tolist()

    # ── 方案 3：终极回退 — 原始 XY 平面最近邻 ──
    from ...utils.mesh_utils import lift_2d_to_vertex

    pt_2d = np.array([[x_2d, y_2d]])
    result = lift_2d_to_vertex(vertices, pt_2d)
    return result[0].tolist()


def validate_landmarks(subject_id: str, landmarks: dict) -> dict:
    """commit 后校验所有 landmark：吸附到最新 mesh 最近顶点 + 重新计算脊椎中点。

    要求 landmarks 为 list 格式（[[Lx,Ly,Lz], [Rx,Ry,Rz], ...]），非 GT dict 格式（{"L":..., "R":...}）。
    """
    from scipy.spatial import KDTree

    # 格式校验：检测是否误传了 GT dict 格式
    sample = next((v for v in landmarks.values() if isinstance(v, dict) and "L" in v), None)
    if sample is not None:
        raise TypeError("validate_landmarks 需要 list 格式，收到 GT dict 格式（含 'L'/'R' key）")

    try:
        mesh, verts = _load_subject_mesh(subject_id)
    except FileNotFoundError:
        return landmarks
    tree = KDTree(verts)

    result: dict = {}
    for name in BILATERAL_LANDMARKS:
        pts = landmarks.get(name, [])
        snapped = []
        for pt in pts:
            if not isinstance(pt, (list, np.ndarray)) or len(pt) < COORD_DIM_3D:
                snapped.append(pt)
                continue
            _, idx = tree.query([float(pt[0]), float(pt[1]), float(pt[2])], k=1)
            snapped.append(verts[idx].tolist())
        result[name] = snapped

    spine: list = []
    for name in SPINE_SOURCE_PAIRS:
        idx = len(spine)
        existing_pt = landmarks.get("spine_points", [])[idx] if idx < len(landmarks.get("spine_points", [])) else None
        if existing_pt is not None and isinstance(existing_pt, (list, np.ndarray)) and len(existing_pt) >= COORD_DIM_3D:
            # 保留用户拖拽的 Pn 位置（在 L-R 连线上的比例），只做 mesh 吸附
            spine.append(existing_pt)
        else:
            pair = result.get(name, [])
            if len(pair) >= PAIR_SIDES and pair[0] is not None and pair[1] is not None:
                spine.append([(pair[0][0] + pair[1][0]) / 2, (pair[0][1] + pair[1][1]) / 2, (pair[0][2] + pair[1][2]) / 2])
            else:
                spine.append(None)

    existing_spine = landmarks.get("spine_points", [])
    for i in range(len(spine), len(existing_spine)):
        spine.append(existing_spine[i])
    # 脊柱点吸附到最近 mesh 顶点（非 None 的点）
    for i, pt in enumerate(spine):
        if pt is None:
            continue
        _, idx = tree.query([float(pt[0]), float(pt[1]), float(pt[2])], k=1)
        spine[i] = verts[idx].tolist()
    result["spine_points"] = spine
    return result
