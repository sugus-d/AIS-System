"""AIS 预测 API 服务 — FastAPI。

两个对外接口：
  POST /api/landmarks — PLY → ROI + landmarks（预处理）
  POST /api/predict   — PLY + clinical [+ landmarks] → cobb + indices + body_params + 报告图
                        不传 landmarks = 状态1（auto，自动检测）；
                        传 landmarks = 状态2（精确，要求完整 18 点，缺失即拒绝）

启动: uvicorn prediction.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from prediction._validation import (
    _build_landmarks_response,
    _read_json_file,
    _sanitize_subject_id,
    _save_upload,
    _validate_clinical,
    _validate_landmarks,
)
from prediction.model_registry import resolve_registered_model
from prediction.predict import (
    _predict_flow,
    PREDICT_ROOT,
    run_landmarks,
)
from prediction.schemas import (
    _APP_DESCRIPTION,
    _LANDMARKS_DESCRIPTION,
    _PREDICT_DESCRIPTION,
    _TAGS_METADATA,
    ErrorResponse,
    LandmarksResponse,
    PredictResponse,
)
from utils.logger import logger

# 报告图片文件名（对齐 predict.py _visualize + _render_waterfall）
_REPORT_FILES = [
    "curvature_mean.png",
    "curvature_gauss.png",
    "roughness.png",
    "normal_angle.png",
    "landmarks.png",
    "back.png",
    "moire.png",
    "waterfall.png",
]

app = FastAPI(
    title="AIS 脊柱侧弯预测 API",
    version="2.0.0",
    description=_APP_DESCRIPTION,
    openapi_tags=_TAGS_METADATA,
)
_SERVICE_TOKEN = os.environ.get("AIS_SERVICE_TOKEN")


@app.middleware("http")
async def require_local_service_token(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    if _SERVICE_TOKEN and request.headers.get("x-ais-service-token") != _SERVICE_TOKEN:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized local service request."})
    return await call_next(request)


@app.get("/health", include_in_schema=False)
def health() -> dict:
    return {"status": "ok"}

# 报告图静态服务：/reports/<subject_id>/report/*.png（只读挂载，指向 prediction/outputs/）
PREDICT_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(PREDICT_ROOT)), name="reports")

# 统一错误响应声明（Swagger 错误码文档）
_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "参数非法（clinical 非 JSON / subject_id 含路径分隔符）"},
    422: {
        "model": ErrorResponse,
        "description": "校验失败（非 .ply / clinical 缺字段 / landmarks 不完整 / PLY 无法解析）",
    },
    500: {"model": ErrorResponse, "description": "管线内部异常"},
}


@app.post(
    "/api/landmarks",
    tags=["landmarks"],
    summary="预处理：PLY → ROI + landmark 检测",
    description=_LANDMARKS_DESCRIPTION,
    response_model=LandmarksResponse,
    responses=_ERROR_RESPONSES,
)
def landmarks_route(
    file: Annotated[UploadFile, File(description="背部 3D 扫描 PLY 文件（原始网格，一般几十 MB 以内）")],
    subject_id: Annotated[
        str | None,
        Form(
            description="[可选] subject ID（输出目录名）。缺省用文件名 stem + 时间戳；"
            "训练集 sid（data/ground_truth/<sid>/ 存在）在 predict 阶段自动用人工 ROI",
        ),
    ] = None,
) -> dict:
    """预处理：PLY → ROI + landmarks（landmarks.json + roi.ply 落盘到 prediction/outputs/）。"""
    if not (file.filename or "").lower().endswith(".ply"):
        raise HTTPException(status_code=422, detail="仅支持 .ply 文件")
    subject = _sanitize_subject_id(subject_id, Path(file.filename or "upload").stem)
    out_dir = PREDICT_ROOT / subject
    ply_path = out_dir / "input" / "original.ply"
    _save_upload(file, ply_path)

    try:
        run_landmarks(str(ply_path), subject, out_dir)
    except ValueError as exc:
        # 如 PLY 无法解析 / ROI 为空——客户端可修复后重试
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"landmarks 管线异常 {subject}: {exc}")
        raise HTTPException(status_code=500, detail="landmarks 提取失败") from exc

    logger.info("API /api/landmarks %s: 检测完成", subject)
    return _build_landmarks_response(out_dir, subject)


@app.post(
    "/api/predict",
    tags=["predict"],
    summary="预测：PLY + clinical → Cobb 角 + 报告",
    description=_PREDICT_DESCRIPTION,
    response_model=PredictResponse,
    response_model_exclude_none=True,
    responses=_ERROR_RESPONSES,
)
def predict_route(
    file: Annotated[
        UploadFile,
        File(description="PLY 网格：状态 1（auto）为原始背部扫描；状态 2（predict）为 ROI 网格（roi.ply）"),
    ],
    clinical: Annotated[
        str,
        Form(
            description="临床数据 JSON 字符串（必填 height_cm/weight_kg/gender），例如 "
            '{"gender":"Female","height_cm":150.5,"weight_kg":38.7}',
        ),
    ],
    landmarks: Annotated[
        str | None,
        Form(
            description="[可选] 完整 landmarks JSON 字符串（18 键扁平格式，如 "
            '{"neck_root_L":[x,y,z],...}；也兼容旧嵌套格式 ground_truth.json）。'
            "不传 = 状态 1（auto）自动检测；传 = 状态 2（predict），缺失任何键即 422 拒绝",
        ),
    ] = None,
    subject_id: Annotated[
        str | None,
        Form(
            description="[可选] subject ID（输出目录名）。缺省用文件名 stem + 时间戳。"
            "状态 2 时输入为 ROI 网格 + 完整 landmarks（不使用本地人工数据）",
        ),
    ] = None,
    model: Annotated[
        str | None,
        Form(
            description="[可选] 模型版本：v1.0.0（缺省，生产）/ v0.1.0（历史）；别名 production/beta",
        ),
    ] = None,
) -> dict:
    """预测：状态1（不传 landmarks）auto 自动检测；状态2（传 landmarks）要求完整 18 点。"""
    if not (file.filename or "").lower().endswith(".ply"):
        raise HTTPException(status_code=422, detail="仅支持 .ply 文件")

    # ── clinical 先校验（不落盘 PLY，校验失败不产生输出目录）────────────
    try:
        clinical_data = json.loads(clinical)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="clinical 不是合法 JSON") from exc
    missing_clinical = _validate_clinical(clinical_data)
    if missing_clinical:
        raise HTTPException(status_code=422, detail=f"clinical 缺少必填字段: {missing_clinical}")

    # ── landmarks：可选 JSON 字符串。提供 → 状态2（必须完整，新旧格式兼容）；否则状态1 ──
    manual_mode = landmarks is not None and bool(landmarks.strip())
    landmarks_data: dict | None = None
    landmarks_path: Path | None = None
    if manual_mode:
        try:
            landmarks_data = json.loads(landmarks or "")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="landmarks 不是合法 JSON") from exc
        missing_landmarks = _validate_landmarks(landmarks_data)
        if missing_landmarks:
            raise HTTPException(status_code=422, detail=f"landmarks 不完整，缺少: {missing_landmarks}")

    subject = _sanitize_subject_id(subject_id, Path(file.filename or "upload").stem)
    out_dir = PREDICT_ROOT / subject
    input_dir = out_dir / "input"
    ply_name = "roi.ply" if manual_mode else "original.ply"
    ply_path = input_dir / ply_name
    _save_upload(file, ply_path)

    clinical_path = input_dir / "clinical.json"
    clinical_path.write_text(json.dumps(clinical_data, ensure_ascii=False), encoding="utf-8")

    if manual_mode:
        assert landmarks_data is not None  # manual_mode 分支内已 json.loads 成功
        landmarks_path = out_dir / "landmarks.json"
        landmarks_path.write_text(json.dumps(landmarks_data, ensure_ascii=False), encoding="utf-8")

    try:
        _predict_flow(
            str(ply_path),
            subject,
            str(clinical_path),
            str(landmarks_path) if landmarks_path else None,
            resolve_registered_model(model or "v1.0.0"),
            out_dir,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"predict 管线异常 {subject}: {exc}")
        raise HTTPException(status_code=500, detail="预测失败") from exc

    prediction = _read_json_file(out_dir / "prediction.json")
    # outputs：auto 含 roi.ply 路径；报告图 URL（合并入 outputs，展开为图名键）
    outputs: dict[str, str] = {}
    if not manual_mode:
        outputs["roi"] = str(out_dir / "roi.ply")
    for name in _REPORT_FILES:
        if (out_dir / "report" / name).exists():
            outputs[name[:-4]] = f"/reports/{subject}/report/{name}"
    # 仅 auto 模式返回 landmarks 与 roi；predict 模式输入已是 ROI+landmarks，不回显
    result = {**prediction, "outputs": outputs}
    if not manual_mode:
        result["landmarks"] = _read_json_file(out_dir / "landmarks.json")
    return result
