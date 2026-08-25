"""OpenAPI 文档描述常量 — app 级 / tag 级 / 路由级描述，集中管理避免 api.py 过长。

从 schemas.py 拆出（P2-5）：纯字符串常量，无任何依赖；schemas.py 经本模块
re-export（api.py 的 import 不变）。
"""

from __future__ import annotations

# ══════════════════════════ OpenAPI 文档描述 ══════════════════════════

_APP_DESCRIPTION = """AIS 青少年特发性脊柱侧弯无创筛查系统的 **Cobb 角预测服务**。

基于 3D 背部扫描（PLY）的完整推理管线（与 `prediction/predict.py` 脚本共用）：

1. **ROI 提取** — `run_roi_pipeline` 裁剪背部区域
2. **Landmark 检测** — 6 类成对解剖点（颈根/肩臂转点/肩胛/腋窝/腰/腰下）+ 脊柱点
3. **调和 UV 参数化** — 拟共形映射，标准化跨主体坐标系
4. **特征提取** — basic 临床(5) + morph(31) + region 候选(2700) + CI 合成(6)
5. **模型预测** — Ridge-AI 边界 Ensemble（缺省 `v1.0.0`）→ Cobb 角 + 严重度
6. **报告生成** — 4 张热力图（曲率均值/高斯/粗糙度/法向角）+ landmark 图 + 背部渲染 + 莫尔条纹 + SHAP 瀑布图

## 接口一览

| 接口 | 说明 |
|------|------|
| `POST /api/landmarks` | 预处理：原始扫描 PLY → ROI + landmarks（roi.ply + landmarks.json） |
| `POST /api/predict` | 预测：PLY + clinical [+ landmarks] → cobb + 分级 + 指数 + 体征参数 + 报告图 |

## 双状态预测

- **状态 1（auto）**：`file`（原始扫描）+ `clinical` → landmarks 自动检测，
  响应返回**完整 18 键 landmarks + roi 路径** + 报告图
- **状态 2（predict）**：`file`（ROI 网格）+ `clinical` + `landmarks`(JSON 字符串，
  必须完整 18 点，缺失 422 拒绝)，响应仅预测字段 + 报告图，**不回显 landmarks/roi**

## 模型版本（`model` 字段选择，缺省 v1.0.0）

| model | 说明 | 指标 |
|-------|------|------|
| `v1.0.0`（缺省） | 生产：人工 ROI → per-class α + Ridge-AI 边界 Ensemble | OOF MF1=0.7364 / MAE=4.38° |
| `v0.1.0`（历史） | manuscript 复现：算法 ROI → 0.6×CompositeV7 + 0.4×AI-LR | OOF MF1=0.7242 / MAE=4.53° |

别名 beta（v0.1.0）可用。权重按模型文件 mtime **自动重载**（模型重训后无需重启服务）。

## 输出位置

预测结果落盘 **`prediction/outputs/<subject_id>/`**：auto 模式产出 roi.ply + landmarks.json +
prediction.json + report/*.png（8 张报告图）；predict 模式仅 prediction.json + report/*.png
（输入已是 ROI + landmarks，不重复产出）。报告图经 `/reports/<sid>/report/*.png` 静态服务。

## 错误码

| 码 | 含义 |
|----|------|
| `400` | 参数非法（clinical 非 JSON、subject_id 含路径分隔符） |
| `422` | 校验失败（非 .ply、clinical 缺字段、landmarks 不完整、PLY 无法解析） |
| `500` | 管线内部异常 |

错误响应统一 `{"detail": "..."}`。

## 快速示例

```bash
# 预处理：PLY → landmarks
curl -X POST http://localhost:8000/api/landmarks \\
    -F "file=@data/ground_truth/23-10673/original.ply" \\
    -F "subject_id=my_subject"

# 预测（状态 1，auto 自动检测，缺省 v1.0.0 模型）
curl -X POST http://localhost:8000/api/predict \\
    -F "file=@data/ground_truth/23-10673/original.ply" \\
    -F 'clinical={"gender":"Female","height_cm":150.5,"weight_kg":38.7}'

# 预测（状态 2，ROI + landmarks，v0.1.0 模型）
curl -X POST http://localhost:8000/api/predict \\
    -F "file=@prediction/outputs/sid/roi.ply" \\
    -F 'clinical={"gender":"Female","height_cm":150.5,"weight_kg":38.7}' \\
    -F 'landmarks={"neck_root_L":[...],"neck_root_R":[...],...}' \\
    -F "model=v0.1.0"
```

详细文档：`prediction/README.md`（CLI 三模式 + HTTP 双接口 + 完整调用/返回示例 + 特征链路 + 已知限制）。
"""

_TAGS_METADATA = [
    {
        "name": "landmarks",
        "description": (
            "**预处理**：PLY → ROI + landmark 检测，产出 landmarks.json + roi.ply。"
            "供 predict 状态 1（auto）前置调用，或人工修正后作为状态 2 输入。"
        ),
    },
    {
        "name": "predict",
        "description": (
            "**预测**：PLY + clinical → Cobb 角 + 严重度 + 5 不对称指数 + 9 体征参数 + 报告图。"
            "双状态：auto（原始扫描自动检测，返回 landmarks+roi）/ "
            "predict（ROI 网格 + 完整 landmarks JSON，不回显 landmarks/roi）。"
        ),
    },
]

_LANDMARKS_DESCRIPTION = """PLY（原始扫描）→ ROI + landmark 检测，产出 roi.ply + landmarks.json，落盘 `prediction/outputs/<subject_id>/`。

## 流程

1. 保存上传 PLY 到 `prediction/outputs/<sid>/input/original.ply`
2. `run_roi_pipeline` 算法提取背部 ROI（保存 `roi.ply`）
3. `extract_landmarks` 检测 6 类成对 landmark + 脊柱点，缺失点用训练集平均补全
   为完整 18 键（保存 `landmarks.json`）
4. 返回 landmarks + ROI 统计 + 产物路径

产出 `roi.ply` + `landmarks.json` 可直接作为 predict 状态 2（predict）输入。

## 示例

```bash
curl -X POST http://localhost:8000/api/landmarks \\
    -F "file=@data/ground_truth/23-10673/original.ply" \\
    -F "subject_id=my_subject"
```

## 错误码

| 码 | 场景 |
|----|------|
| `400` | subject_id 含路径分隔符（防目录逃逸） |
| `422` | 非 .ply 文件 / PLY 无法解析 / ROI 为空 |
| `500` | 管线内部异常 |
"""

_PREDICT_DESCRIPTION = """PLY + clinical [+ landmarks] → Cobb 角 + 严重度 + 5 不对称指数 + 9 体征参数 + 报告图。

## 双状态

- **状态 1（auto）**：`file`（原始扫描 PLY）+ `clinical`(JSON) → landmarks 自动检测，
  响应返回完整 18 键 landmarks + roi 路径 + 报告图
- **状态 2（predict）**：`file`（ROI 网格 roi.ply）+ `clinical`(JSON) + `landmarks`(JSON 字符串)，
  **必须完整 18 扁平键**（也兼容旧嵌套格式），缺失直接 422 拒绝；响应仅预测字段 + 报告图，
  不回显 landmarks/roi

## 模型选择

`model`（可选 Form 字段）：`v1.0.0`（缺省，生产）/ `v0.1.0`（历史 manuscript 复现，0.6×CompositeV7 + 0.4×AI-LR）。别名 beta（v0.1.0）可用。

## 示例

```bash
# 状态 1（auto，原始扫描 → 自动检测）
curl -X POST http://localhost:8000/api/predict \\
    -F "file=@data/ground_truth/23-10673/original.ply" \\
    -F 'clinical={"gender":"Female","height_cm":150.5,"weight_kg":38.7}'

# 状态 2（ROI 网格 + 完整 landmarks JSON，v0.1.0 模型）
curl -X POST http://localhost:8000/api/predict \\
    -F "file=@prediction/outputs/sid/roi.ply" \\
    -F 'clinical={"gender":"Female","height_cm":150.5,"weight_kg":38.7}' \\
    -F 'landmarks={"neck_root_L":[x,y,z],...}' \\
    -F "model=v0.1.0"
```

## 响应字段

| 字段 | 说明 |
|------|------|
| `cobb` / `severity` | 预测 Cobb 角 + 分级 |
| `model_id` | 模型版本编号（`v1.0.0` / `v0.1.0`） |
| `indices` | 5 个不对称指数（curvature/height/normal_angle/roughness/asymmetric） |
| `body_params` | 9 个体征参数（可读名 + 单位） |
| `landmarks` | 完整 18 键（仅 auto 模式返回） |
| `outputs` | 8 张报告图 URL（auto 额外含 `roi` 路径） |

## 错误码

| 码 | 场景 |
|----|------|
| `400` | clinical / landmarks 不是合法 JSON |
| `422` | 非 .ply / clinical 缺字段 / landmarks 不完整 / PLY 无法解析 |
| `500` | 管线内部异常 |
"""
