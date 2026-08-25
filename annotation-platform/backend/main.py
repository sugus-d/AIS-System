"""Landmark 标注平台 — FastAPI backend（纯 API，前端由 Vite dev server 提供）"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.api.export import router as export_router
from backend.api.landmarks import router as landmarks_router
from backend.api.metrics import router as metrics_router
from backend.api.subjects import router as subjects_router
from backend.api.annotation_session import router as annotation_session_router

app = FastAPI(title="Landmark Labeling Platform")

_allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ANNOTATION_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
if not _allowed_origins:
    _allowed_origins = ["http://127.0.0.1", "http://localhost"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):\d+",
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(subjects_router)
app.include_router(landmarks_router)
app.include_router(metrics_router)
app.include_router(export_router)
app.include_router(annotation_session_router)

_frontend = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _frontend.is_dir():
    app.mount("/assets", StaticFiles(directory=_frontend / "assets"), name="annotation-assets")


@app.get("/api/health")
def health() -> dict:
    """健康检查端点。"""
    return {"status": "ok"}


@app.get("/api/clinical-data")
def clinical_data() -> dict:
    """返回所有 subject 的临床数据（来自 clinical_data.json）。"""
    from backend.api.subjects import _load_clinical_data

    return _load_clinical_data()


@app.get("/{path:path}", include_in_schema=False)
def annotation_spa(path: str):
    """Serve the packaged React workbench, including deep-linked subject routes."""
    index = _frontend / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"detail": "Annotation frontend is not installed."}
# REV backend-main-new
