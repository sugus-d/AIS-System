"""Subject API 路由。"""

import json
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel

from ..constants import DATA_ROOT
from ..services.converter import (
    brush_commit,
    overlay_cloth_data,
    ply_to_glb_bytes,
)
from ..services.curvature import get_contours, render_curvature_image
from ..services.subject_loader import (
    _find_mesh_file,
    _load_clinical,
    compute_labeling_status,
    discover_subjects,
    get_algorithm_features,
    get_gt_features,
    set_manual_labeling_status,
)

CLINICAL_JSON: Path = DATA_ROOT / "form" / "clinical_data.json"


def _load_clinical_data() -> dict:
    """读取 clinical_data.json（由 convert_clinical.py 从 Excel 生成）。"""
    if not CLINICAL_JSON.exists():
        return {}
    return json.loads(CLINICAL_JSON.read_text())


router = APIRouter(prefix="/api/subjects", tags=["subjects"])

# Module-level cache for list_subjects (NAS 文件访问慢)
_subjects_cache: list[dict] | None = None
_subjects_cache_time: float = 0
_LIST_CACHE_TTL: int = 120  # 秒


@router.get("")
def list_subjects() -> list[dict]:
    """列出全部 subject（仅 id + status，详情延迟加载）。"""
    global _subjects_cache, _subjects_cache_time
    now = time.time()
    if _subjects_cache and now - _subjects_cache_time < _LIST_CACHE_TTL:
        return _subjects_cache
    result = [
        {
            "id": s.id,
            "labeling_status": compute_labeling_status(s.id),
            "has_cache": s.has_cache,
        }
        for s in discover_subjects()
    ]
    _subjects_cache = result
    _subjects_cache_time = now
    return result


@router.get("/{subject_id}")
def get_subject(subject_id: str) -> dict:
    """获取单个 subject 详情（按需加载临床数据 + mesh 路径）。"""
    subs = discover_subjects()
    s = next((x for x in subs if x.id == subject_id), None)
    if s is None:
        return {"error": "not found"}
    age, sex, bmi = _load_clinical(subject_id)
    gt_feat = get_gt_features(subject_id)
    algo_feat = get_algorithm_features(subject_id)
    return {
        "id": s.id,
        "has_gt": s.has_gt,
        "has_cache": s.has_cache,
        "labeling_status": compute_labeling_status(subject_id),
        "mesh_file": _find_mesh_file(subject_id),
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "arms": gt_feat.get("arms") or algo_feat.get("arms", "unknown"),
        "body_asymmetry": gt_feat.get("body_asymmetry", False),
        "notes": gt_feat.get("notes", ""),
    }


@router.get("/{subject_id}/curvature-image")
def curvature_image(subject_id: str) -> Response:
    """返回曲率热力图 PNG（用于前端 2D 视图底图）。"""
    try:
        png_bytes, mapping = render_curvature_image(subject_id)
    except FileNotFoundError:
        return Response(status_code=404, content=b"")
    return Response(content=png_bytes, media_type="image/png")


@router.get("/{subject_id}/curvature-mapping")
def curvature_mapping(subject_id: str) -> dict:
    """返回曲率图的坐标映射参数（像素→mm PCA 变换矩阵）。"""
    try:
        _, mapping = render_curvature_image(subject_id)
    except FileNotFoundError:
        return Response(status_code=404, content=b"")
    return mapping


@router.get("/{subject_id}/contours")
def subject_contours(subject_id: str) -> dict:
    """返回 PCA 投影后的人体左右轮廓 2D 坐标（供前端叠加显示）。"""
    return get_contours(subject_id)


@router.get("/{subject_id}/mesh")
def subject_mesh(subject_id: str, clothed: bool = False) -> Response:
    """返回 subject 的 3D mesh（GLB 格式），clothed=True 为原始有衣版本。"""
    try:
        glb = ply_to_glb_bytes(subject_id, clothed=clothed)
    except FileNotFoundError:
        return Response(status_code=404, content=b"")
    return Response(content=glb, media_type="model/gltf-binary")


# ── Overlay (clothed → extra regions in red) ──
@router.get("/{subject_id}/mesh/overlay-cloth")
def overlay_cloth(subject_id: str) -> dict:
    """返回有衣 mesh 中未被 body 覆盖的布料顶点/面（红色 overlay 显示用）。"""
    return overlay_cloth_data(subject_id)


class LabelingStatusBody(BaseModel):
    status: str


@router.put("/{subject_id}/labeling-status")
def put_labeling_status(subject_id: str, body: LabelingStatusBody) -> dict:
    """手动设置标注状态（覆盖自动计算）。"""
    if body.status not in ("unlabeled", "prelabeled", "labeled"):
        return {"error": f"无效状态: {body.status}"}
    set_manual_labeling_status(subject_id, body.status)
    return {"labeling_status": body.status}


class CommitBody(BaseModel):
    points: list[list[float]] | None = None
    cloth_indices: list[int] | None = None


# ── 无状态笔刷提交 ──────────────────────────────────
@router.post("/{subject_id}/brush/commit")
def brush_commit_route(subject_id: str, body: CommitBody) -> dict:
    """笔刷擦除/恢复：points=擦除 body 顶点，cloth_indices=恢复布料顶点。"""
    return brush_commit(subject_id, points=body.points, cloth_indices=body.cloth_indices)
