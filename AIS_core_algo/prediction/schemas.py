"""API 响应模型 — FastAPI `response_model`，让 `/docs` 交互文档展示完整的请求/响应结构。

OpenAPI 文档描述常量（app 级 / tag 级 / 路由级）在 `prediction/_descriptions.py`，
本模块 re-export（api.py 的 import 不变）。模型与实际返回 JSON 逐字段对齐
（产物来自 `prediction/predict.py`，路由组装在 `prediction/api.py`）。
"""

# ruff: noqa: N815 — 扁平 landmarks 键名（neck_root_L 等）是 API 契约，非 Python 变量命名

from __future__ import annotations

from pydantic import BaseModel, Field

from prediction._descriptions import (  # noqa: F401 — re-export 给 api.py，本模块不直接使用
    _APP_DESCRIPTION,
    _LANDMARKS_DESCRIPTION,
    _PREDICT_DESCRIPTION,
    _TAGS_METADATA,
)

# ══════════════════════════ landmarks 响应结构 ══════════════════════════


class LandmarksBody(BaseModel):
    """扁平 landmarks 全集：18 键（6 对 bilateral _L/_R + 6 个 spine 点），每键一个三维坐标。

    说明：算法自动检测结果缺 `waist_lower`、spine 不足 6 点，predict 内部用
    训练集归一化平均 + mesh 最近顶点补全为完整 18 点后返回。
    """

    neck_root_L: list[float] = Field(description="颈根点左侧 [x, y, z]（mm）")
    neck_root_R: list[float] = Field(description="颈根点右侧 [x, y, z]（mm）")
    shoulder_transition_L: list[float] = Field(description="肩臂转点左侧 [x, y, z]（mm）")
    shoulder_transition_R: list[float] = Field(description="肩臂转点右侧 [x, y, z]（mm）")
    scapular_peaks_L: list[float] = Field(description="肩胛最高点左侧 [x, y, z]（mm）")
    scapular_peaks_R: list[float] = Field(description="肩胛最高点右侧 [x, y, z]（mm）")
    axilla_L: list[float] = Field(description="腋窝点左侧 [x, y, z]（mm）")
    axilla_R: list[float] = Field(description="腋窝点右侧 [x, y, z]（mm）")
    waist_L: list[float] = Field(description="腰点左侧 [x, y, z]（mm）")
    waist_R: list[float] = Field(description="腰点右侧 [x, y, z]（mm）")
    waist_lower_L: list[float] = Field(description="腰下点左侧 [x, y, z]（mm）")
    waist_lower_R: list[float] = Field(description="腰下点右侧 [x, y, z]（mm）")
    neck_root_spine_point: list[float] = Field(description="脊柱 P0 点（颈根水平）[x, y, z]（mm）")
    scapular_spine_point: list[float] = Field(description="脊柱 P1 点（肩胛水平）[x, y, z]（mm）")
    axilla_spine_point: list[float] = Field(description="脊柱 P2 点（腋窝水平）[x, y, z]（mm）")
    waist_spine_point: list[float] = Field(description="脊柱 P3 点（腰水平）[x, y, z]（mm）")
    waist_lower_spine_point: list[float] = Field(description="脊柱 P4 点（腰下水平）[x, y, z]（mm）")
    thoracic_spine_point: list[float] = Field(description="脊柱 P5 点（胸腰段，P2/P3 之间）[x, y, z]（mm）")


class LandmarksOutputs(BaseModel):
    """landmarks 接口产物路径。"""

    roi: str = Field(description="ROI 网格文件（prediction/outputs/<sid>/roi.ply）")


class LandmarksResponse(BaseModel):
    """POST /api/landmarks 响应。"""

    subject_id: str = Field(description="subject ID（请求传入或文件名_stem_时间戳自动生成）")
    landmarks: LandmarksBody = Field(description="检测到的 landmarks（ground_truth 兼容格式）")
    outputs: LandmarksOutputs = Field(description="产物文件路径（roi.ply）")


# ══════════════════════════ predict 响应结构 ══════════════════════════


class BodyParam(BaseModel):
    """单个体征参数（论文表2）：info 医学意义带单位 + value。"""

    info: str = Field(description="参数医学意义（带单位，如『左右肩垂直高度差(mm)』）")
    value: float = Field(description="参数数值")


class Indices(BaseModel):
    """5 个不对称指数（公式来自训练时保存的模型包参数，对应论文表3）。"""

    curvature_index: float = Field(description="曲率不对称指数")
    height_index: float = Field(description="表面高度不对称指数")
    normal_angle_index: float = Field(description="法向角不对称指数")
    roughness_index: float = Field(description="粗糙度不对称指数")
    asymmetric_index: float = Field(description="综合不对称指数")


class PredictOutputs(BaseModel):
    """predict 接口产物：roi.ply 路径（仅 auto）+ 8 张报告图 URL（按图名展开）。"""

    roi: str | None = Field(default=None, description="ROI 网格文件路径（仅 auto 模式返回）")
    curvature_mean: str | None = Field(default=None, description="平均曲率热力图 URL")
    curvature_gauss: str | None = Field(default=None, description="高斯曲率热力图 URL")
    roughness: str | None = Field(default=None, description="粗糙度热力图 URL")
    normal_angle: str | None = Field(default=None, description="法向角热力图 URL")
    landmarks: str | None = Field(default=None, description="landmark 连线图 URL")
    back: str | None = Field(default=None, description="背部原始渲染图 URL（中国肤色光照）")
    moire: str | None = Field(default=None, description="莫尔条纹图 URL（珍珠白底 + 光照）")
    waterfall: str | None = Field(default=None, description="SHAP 特征贡献瀑布图 URL")


class PredictResponse(BaseModel):
    """POST /api/predict 响应。"""

    subject_id: str = Field(description="subject ID")
    cobb: float = Field(description="预测 Cobb 角（度，钳制 0-90）")
    severity: str = Field(description="严重度分级：Normal(<10°) / Mild(10-20°) / Moderate(20-40°) / Severe(≥40°)")
    model_id: str = Field(
        description="模型版本编号（v1.0.0 = 生产，人工 ROI per-class α + Ridge-AI 边界 Ensemble；v0.1.0 = 历史 manuscript 复现）"
    )
    clinical: dict = Field(description="临床数据（原样回传）")
    indices: Indices = Field(description="5 个不对称指数")
    body_params: dict[str, BodyParam] = Field(description="9 个体征参数（键为简写如 Sh.IB，值为 {info, value}）")
    landmarks: LandmarksBody | None = Field(default=None, description="完整 landmarks（仅 auto 模式返回，含补全）")
    outputs: PredictOutputs = Field(description="报告图 URL（auto 模式含 roi 路径）")


class ErrorResponse(BaseModel):
    """统一错误响应（400 / 422 / 500）。"""

    detail: str = Field(description="错误描述（中文，含具体失败原因）")
