"""指标计算 + 校验 API。"""

from pathlib import Path

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from ..constants import BILATERAL_LANDMARKS, CACHE_DIR, DATA_ROOT
from ..services.lifter import load_landmarks
from ..services.subject_loader import discover_subjects, get_algorithm_features, get_gt_features

router = APIRouter(prefix="/api", tags=["metrics"])

PAIR_SIDES = 2  # bilateral 成对 landmark 的 L/R 数量


def compute_metrics(
    landmarks: dict,
    contours: dict | None = None,
    features: dict | None = None,
) -> dict[str, object]:
    """计算全部指标：左右偏差、对称度、颈/腋/腰宽比、脊柱序列。"""
    m: dict = {}
    for name in BILATERAL_LANDMARKS:
        pts = landmarks.get(name)
        if pts is None or len(pts) < PAIR_SIDES or pts[0] is None or pts[1] is None:
            continue
        L, R = np.array(pts[0]), np.array(pts[1])
        dx = float(abs(R[0] - L[0]))
        dy = float(abs(R[1] - L[1]))
        dz = float(abs(R[2] - L[2]))
        angle = float(np.degrees(np.arctan2(dy, dx))) if dx > 1 else 0.0
        m[name] = {
            "L": [float(L[0]), float(L[1]), float(L[2])],
            "R": [float(R[0]), float(R[1]), float(R[2])],
            "dx": round(dx, 1),
            "dy": round(dy, 1),
            "dz": round(dz, 1),
            "angle_deg": round(angle, 1),
        }

    # 颈/腋/腰宽比
    nr_dx = m.get("neck_root", {}).get("dx", 0)
    ax_dx = m.get("axilla", {}).get("dx", 0)
    wa_dx = m.get("waist", {}).get("dx", 0)
    if nr_dx and ax_dx:
        m["neck_axilla_ratio"] = round(nr_dx / ax_dx * 100, 1)
    if nr_dx and wa_dx:
        m["neck_waist_ratio"] = round(nr_dx / wa_dx * 100, 1)

    # 脊柱（跳过 None 的未放置点）
    sp = landmarks.get("spine_points", [])
    if sp:
        m["spine"] = []
        for i, pt in enumerate(sp):
            if pt is None:
                continue
            m["spine"].append({"index": i, "x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2])})
    return m


@router.get("/subjects/{subject_id}/metrics")
def get_metrics(subject_id: str) -> dict:
    """获取指定 subject 的指标计算结果，包含对称度和宽比。"""
    lm = load_landmarks(subject_id)
    features = get_gt_features(subject_id)
    if not features:
        features = get_algorithm_features(subject_id)
    metrics = compute_metrics(lm)
    return {
        "subject_id": subject_id,
        "is_relaxed": features.get("arms") == "none" or features.get("body_asymmetry", False),
        "metrics": metrics,
    }


@router.get("/subjects/{subject_id}/validate")
def validate(subject_id: str) -> dict:
    """校验 landmark 是否符合标准阈值（左右对称度等）。"""
    lm = load_landmarks(subject_id)
    features = get_gt_features(subject_id)
    if not features:
        features = get_algorithm_features(subject_id)
    is_relaxed = features.get("arms") == "none" or features.get("body_asymmetry", False)
    metrics = compute_metrics(lm)

    th: dict = {
        "neck_root": {"dy": 40 if is_relaxed else 20},
        "scapular_peaks": {"dy": 20 if is_relaxed else 10},
        "waist": {"contour": 5},
    }

    checks: dict = {}
    nr = metrics.get("neck_root", {})
    if nr:
        t = th["neck_root"]["dy"]
        checks["neck_root_dy"] = {"value": nr["dy"], "threshold": t, "pass": nr["dy"] <= t}
    sp = metrics.get("scapular_peaks", {})
    if sp:
        t = th["scapular_peaks"]["dy"]
        checks["scapular_peaks_dy"] = {"value": sp["dy"], "threshold": t, "pass": sp["dy"] <= t}
    return {"is_relaxed": is_relaxed, "checks": checks}


class GenerateResponse(BaseModel):
    status: str
    count: int


@router.post("/batch/generate")
def batch_generate() -> dict:
    """对所有未标注 subject 运行算法检测，生成 landmark 坐标并保存为 GT。"""
    import open3d as o3d

    from landmarks.extract import extract_landmarks

    subjects = discover_subjects()
    count = 0
    for s in subjects:
        if s.has_gt:
            continue
        if not s.has_cache:
            continue
        for subdir in ("extract_roi", "align"):
            p = Path(s.mesh_file or "")
            if not p.is_absolute():
                p = DATA_ROOT / (s.mesh_file or "")
            cache_p = CACHE_DIR / s.id / subdir / "output.ply"
            if cache_p.exists():
                mesh = o3d.io.read_triangle_mesh(str(cache_p))
                break
        else:
            continue
        data = extract_landmarks(mesh, is_debug=False)
        keys = ["neck_root", "shoulder_transition", "scapular_peaks", "axilla", "waist", "spine_points"]
        lm = {k: np.asarray(data[k]).tolist() for k in keys if k in data}
        from ..services.lifter import save_landmarks

        save_landmarks(s.id, lm)
        count += 1
    return {"status": "done", "count": count}
