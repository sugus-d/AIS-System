"""API 请求校验与文件落盘辅助 — 与路由解耦，供 FastAPI 路由与单测复用。

从 api.py 拆出（P2-4）：subject_id 防目录逃逸、landmarks/clinical 字段校验、
上传分块落盘、JSON 安全读取、landmarks 响应组装。依赖仅 fastapi/http/landmarks.constants。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, UploadFile

from landmarks.constants import FLAT_KEYS

# 上传 mesh 大小上限（原始背部 PLY 一般几十 MB）
_MAX_PLY_BYTES = 100 * 1024 * 1024
# landmark 坐标维度（x/y/z）
_COORD_COUNT = 3


def _sanitize_subject_id(subject_id: str | None, file_stem: str) -> str:
    """校验/生成 subject_id：拒绝路径分隔符/`.`/`..`（防目录逃逸）；缺省用文件名_stem_时间戳。"""
    if subject_id is not None and subject_id.strip():
        raw = subject_id.strip()
        candidate = Path(raw).name
        if candidate in ("", ".", "..") or "/" in raw or "\\" in raw:
            raise HTTPException(status_code=400, detail="subject_id 非法")
        return candidate
    return f"{file_stem}_{datetime.now():%m%d%H%M%S}"


def _validate_landmarks(data: dict) -> list[str]:
    """校验 landmarks 完整性（18 扁平键），返回缺失键列表。

    API 状态 2 要求完整无缺：缺任何键直接拒绝，不靠 predict 内部补全。
    """
    missing: list[str] = []
    for key in FLAT_KEYS:
        coordinate = data.get(key)
        if not (isinstance(coordinate, list) and len(coordinate) == _COORD_COUNT):
            missing.append(key)
    return missing


def _validate_clinical(data: dict) -> list[str]:
    """校验临床数据必填字段（模型 basic 特征消费 height/weight/gender），返回缺失列表。"""
    missing: list[str] = []
    height = data.get("height_cm")
    weight = data.get("weight_kg")
    gender = data.get("gender")
    if height is None or not isinstance(height, (int, float)) or height <= 0:
        missing.append("height_cm")
    if weight is None or not isinstance(weight, (int, float)) or weight <= 0:
        missing.append("weight_kg")
    if gender is None or (isinstance(gender, str) and not gender.strip()):
        missing.append("gender")
    return missing


def _save_upload(upload: UploadFile, path: Path) -> None:
    """上传文件分块落盘（含大小上限，防大文件占满磁盘）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with path.open("wb") as file_handle:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > _MAX_PLY_BYTES:
                raise HTTPException(status_code=422, detail="文件超过大小上限")
            file_handle.write(chunk)


def _read_json_file(path: Path) -> dict:
    """读取 JSON 文件；非法内容抛 400（安全：只用 json.loads，不 eval/pickle）。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"JSON 解析失败: {path.name}") from exc


def _build_landmarks_response(out_dir: Path, subject_id: str) -> dict:
    """landmarks 接口响应：landmarks + ROI 产物路径。"""
    roi_path = out_dir / "roi.ply"
    return {
        "subject_id": subject_id,
        "landmarks": _read_json_file(out_dir / "landmarks.json"),
        "outputs": {
            "roi": str(roi_path),
        },
    }
