"""Landmark CRUD + lift API。"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.lifter import (
    _validate_coordinate_order,
    lift_2d_to_3d,
    load_landmarks,
    reset_landmarks,
    save_landmarks,
    validate_landmarks,
)

router = APIRouter(prefix="/api/subjects/{subject_id}/landmarks", tags=["landmarks"])


@router.get("")
def get_landmarks(subject_id: str) -> dict:
    """获取指定 subject 的所有 landmark 标注数据（3D 坐标）。"""
    lm = load_landmarks(subject_id)
    return lm


class LandmarksSaveBody(BaseModel):
    landmarks: dict
    bypass_validation: bool = False


@router.put("")
def put_landmarks(subject_id: str, body: LandmarksSaveBody) -> dict:
    """保存 landmark 标注到 GT JSON，返回计算后的标注状态。

    error 级别的校验问题（如左右反了）会阻止保存，除非传 bypass_validation=true。
    """
    from ..services.subject_loader import compute_labeling_status

    # 先校验坐标（仅检查 landmarks 数据本身，不依赖 mesh）
    issues = _validate_coordinate_order(body.landmarks)
    errors = [i for i in issues if i.get("severity") == "error"]
    if errors and not body.bypass_validation:
        raise HTTPException(
            status_code=400,
            detail={"status": "validation_error", "issues": errors},
        )

    save_landmarks(subject_id, body.landmarks)
    # 保存成功后，也把 warning 级别的问题带回去
    warnings = [i for i in issues if i.get("severity") == "warning"]
    result: dict = {"status": "saved", "labeling_status": compute_labeling_status(subject_id)}
    if warnings:
        result["issues"] = warnings
    return result


@router.post("/reset")
def post_reset(subject_id: str) -> dict:
    """删除 GT 标注，回退到算法自动检测值。"""
    return reset_landmarks(subject_id)


class LiftBody(BaseModel):
    x: float
    y: float


@router.post("/lift")
def post_lift(subject_id: str, body: LiftBody) -> dict:
    """2D 像素坐标 → 沿 PCA 方向 ray casting 到 mesh 表面 → 3D 顶点坐标。"""
    result = lift_2d_to_3d(subject_id, body.x, body.y)
    return {"x": result[0], "y": result[1], "z": result[2]}


@router.post("/validate")
def post_validate(subject_id: str, body: LandmarksSaveBody) -> dict:
    """校验所有 landmark：吸附到最新 body mesh 最近顶点 + 坐标顺序检查。"""
    result = validate_landmarks(subject_id, body.landmarks)
    issues = _validate_coordinate_order(result)
    return {"landmarks": result, "issues": issues}
