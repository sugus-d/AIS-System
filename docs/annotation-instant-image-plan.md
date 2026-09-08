# 标注即时图像方案（annotation-instant-image）

> 目标：用户标注完成、回到受检者报告页时，**即时显示标注后的连线图**；重新分析继续使用新标注文件。
> 原则：**纯增量**，不删 auto、不改 `/api/landmarks`、不改 `/api/predict` 现有语义，不新增 `prediction-outputs/<caseId>-*` 目录，渲染失败不阻塞标注完成。

---

## 0. 决策记录

| 项 | 决定 |
|---|---|
| 即时渲染范围 | 仅 `landmarks.png`（标注连线图） |
| 渲染接口命名 | `POST /api/render` |
| 版本号存储 | `resultJson.annotationImageVersion`（免 Prisma 迁移） |
| 渲染落盘 | **原地覆盖**当前 `artifactDirectory/report/landmarks.png` |

---

## 1. 现状与问题（代码依据）

1. 报告页四张算法图（`landmarks.png` / `curvature_mean.png` / `moire.png` / `normal_angle.png`）**全部**在预测管线末尾 `_visualize` 一次性渲染（`AIS_core_algo/prediction/predict.py:201`）。这条管线包含参数化 + 特征 + 模型推理，秒级到几十秒。
2. 前端 `client/pages/AnalysisReport.tsx:324-338`：标注回来后 `completeAnnotation()` → `pollTaskForReport()` **等整条重分析跑完**才刷新，所以标注图要等预测结束才可见。
3. 重分析用新标注文件**已经实现**：`annotation.ts:60-71` 读 `ground_truth.json` + `roi_edited_*.ply` 建 `annotation_reanalysis`，`analysis-runner.ts:35-36` 传 `manualRoiPath + landmarks`，`/api/predict` 状态 2 用传入 landmarks 渲染 `landmarks.png`。

**缺口**：标注图渲染与预测耦合，缺一条「秒级渲染标注图」的快速通道。

---

## 2. 方案总览

```
标注完成 (ground_truth.json + roi_edited_*.ply 已落盘)
        │
        ▼
completed 接口
  ├─(新) POST /api/render  同步覆盖 artifactDirectory/report/landmarks.png
  │        写 resultJson.annotationImageVersion = 时间戳
  │
  ├─(旧) 建 annotation_reanalysis 任务 → enqueueAnalysisTask（不改）
  └─ 返回 reanalysisTaskId
        │
        ▼
前端 AnalysisReport.tsx
  ① completeAnnotation 后立即 api.getReport → 标注图秒级显示（URL 带新 &av= 触发重取）
  ② 继续 pollTaskForReport → 完成后刷新 Cobb/热力图
```

关键约束（防止污染标注平台扫描）：标注平台 `_paths.py:40-47` 会扫描 `prediction-outputs/<caseId>-*` 取 mtime 最新目录读 `roi.ply`/`landmarks.json`。因此**不能**为标注图新开 `<caseId>-*` 目录，必须**原地覆盖当前报告目录里的 `landmarks.png`**。

---

## 3. 逐文件改动

### 3.1 `AIS_core_algo/prediction/render.py`（新增）

轻量渲染模块，**只**做「ROI mesh + landmarks → landmarks.png」，复用现有渲染函数保证与预测报告图**逐像素一致**。

```python
# prediction/render.py
from __future__ import annotations
from pathlib import Path
import open3d as o3d
from prediction.measures import _compute_measures
from landmarks.complete import complete_landmarks_flat
from landmarks.constants import FLAT_KEYS

def render_landmarks_image(ply_path: str, landmarks_flat: dict, out_dir: Path) -> Path:
    """渲染标注连线图 landmarks.png（不加载模型/特征/参数化）。

    Args:
        ply_path: ROI PLY 路径（编辑后 ROI 或算法 ROI）。
        landmarks_flat: 扁平 18 键 landmarks（与 flattenGroundTruth 输出一致）。
        out_dir: 输出根（landmarks.png 写到 out_dir/report/）。

    Returns:
        输出 PNG 路径（out_dir/report/landmarks.png）。

    Raises:
        ValueError: mesh 为空 / landmarks 不完整（缺 FLAT_KEYS）。
    """
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty():
        raise ValueError("无法加载 ROI mesh")

    missing = [k for k in FLAT_KEYS if k not in landmarks_flat]
    if missing:
        raise ValueError(f"landmarks 不完整，缺少: {missing[:5]}")
    flat = complete_landmarks_flat(landmarks_flat, mesh)

    measures = _compute_measures(mesh)  # 曲率/粗糙度/法向，仅 mesh 级计算

    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from visualization._render_utils import save_img
    from visualization._style import ACADEMIC_STYLE
    from visualization.back_panels import render_back_landmarks

    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(ACADEMIC_STYLE):
        fig, ax = plt.subplots(figsize=(6, 8))
        render_back_landmarks(ax, measures["vertices"], measures["faces"],
                              measures["normals"], flat)
        fig.tight_layout()
        out_path = report_dir / "landmarks.png"
        save_img(fig, str(out_path), dpi=150)
    return out_path
```

> 说明：渲染参数（`figsize=(6,8)`、`ACADEMIC_STYLE`、`save_img dpi=150`）与 `prediction/report.py:111-116` 的 `landmarks.png` 完全一致，保证标注图与预测报告图同一风格。

### 3.2 `AIS_core_algo/prediction/api.py`（新增路由，纯增量）

在 `api.py` 追加 `POST /api/render`，复用 `_save_upload` / `_sanitize_subject_id` / `_validate_landmarks` / `_read_json_file`。**不 import 模型、feature_pipeline、report_waterfall**。

```python
# prediction/api.py 追加
from prediction.render import render_landmarks_image

@app.post("/api/render", tags=["render"],
          summary="轻量渲染：ROI + landmarks → landmarks.png（不加载模型/特征）")
def render_route(
    file: Annotated[UploadFile, File(description="ROI 网格 PLY（编辑后 ROI 或算法 roi.ply）")],
    landmarks: Annotated[str, Form(description="完整扁平 18 键 landmarks JSON 字符串")],
    subject_id: Annotated[str | None, Form(description="[可选] subject ID（输出目录名，用于原地覆盖）")] = None,
) -> dict:
    if not (file.filename or "").lower().endswith(".ply"):
        raise HTTPException(status_code=422, detail="仅支持 .ply 文件")

    try:
        landmarks_data = json.loads(landmarks)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="landmarks 不是合法 JSON") from exc
    missing = _validate_landmarks(landmarks_data)
    if missing:
        raise HTTPException(status_code=422, detail=f"landmarks 不完整，缺少: {missing}")

    subject = _sanitize_subject_id(subject_id, Path(file.filename or "upload").stem)
    out_dir = PREDICT_ROOT / subject
    ply_path = out_dir / "input" / "roi.ply"
    _save_upload(file, ply_path)

    try:
        out_path = render_landmarks_image(str(ply_path), landmarks_data, out_dir)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"render 管线异常 {subject}: {exc}")
        raise HTTPException(status_code=500, detail="标注图渲染失败") from exc

    return {
        "subject_id": subject,
        "outputs": {"landmarks": f"/reports/{subject}/report/{out_path.name}"},
    }
```

> `render_route` 复用 `/api/predict` 的中间件（`x-ais-service-token`）与 `_MAX_PLY_BYTES` 上限，无需新增鉴权逻辑。

### 3.3 `server/services/algorithm.ts`（新增 `renderImages`）

仿照 `predict()`（`algorithm.ts:6-21`）新增：

```ts
export async function renderImages(filePath: string, subjectId: string, landmarks: Record<string, unknown>) {
  const bytes = await readFile(filePath);
  const body = new FormData();
  body.append("file", new Blob([bytes]), path.basename(filePath));
  body.append("subject_id", subjectId);
  body.append("landmarks", JSON.stringify(landmarks));
  const serviceToken = process.env.AIS_SERVICE_TOKEN;
  let response: globalThis.Response;
  try {
    response = await fetch(`${algorithmUrl}/api/render`, {
      method: "POST", body,
      headers: serviceToken ? { "x-ais-service-token": serviceToken } : undefined,
      signal: AbortSignal.timeout(Number(process.env.AIS_ALGORITHM_TIMEOUT_MS || 300000)),
    });
  } catch { throw new Error("AIS 核心算法服务不可用"); }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || "AIS 标注图渲染失败");
  return payload;
}
```

### 3.4 `server/routes/annotation.ts`（`completed` 插入即时渲染）

在 `POST /reports/:reportId/completed`（`annotation.ts:60-71`）里，**建完 task、返回之前**插入（`landmarks` / `manualRoiPath` 已在 65-66 行算好）：

```ts
// —— 新增：即时渲染标注图（原地覆盖当前报告的 landmarks.png）——
try {
  const renderSubjectId = path.basename(report.artifactDirectory); // <caseId>-<oldTaskId>
  if (renderSubjectId) {
    await renderImages(manualRoiPath, renderSubjectId, landmarks);
    const result = JSON.parse(report.resultJson);
    result.annotationImageVersion = new Date().toISOString();
    await db.report.update({ where: { id: report.id }, data: { resultJson: JSON.stringify(result) } });
  }
} catch (error) {
  // 渲染失败静默降级：前端仍会等重分析，行为与今天一致
  console.warn("Annotation image render failed, falling back to reanalysis:", error);
}
// —— 以下为原有逻辑，不改 ——
```

需要顶部补 import：`import { renderImages } from "../services/algorithm";`（`path` 已在文件顶部导入）。

### 3.5 `server/routes/reports.ts`（标注图加版本号）

`present()`（`reports.ts:13`）里给 `annotatedImage` 追加 `&av=` 版本参数。将第 32 行的 `imageUrl` 使用处改为：

```ts
const av = result.annotationImageVersion; // 从 resultJson 读
// present() 返回对象里：
annotatedImage: av ? `${imageUrl("landmarks")}&av=${encodeURIComponent(av)}` : imageUrl("landmarks"),
```

> 原理：`completed` 原地覆盖了 `artifactDirectory/report/landmarks.png`，但 `?v=<目录名>` 未变，浏览器会命中缓存；`&av=<时间戳>` 让 URL 变化从而强制重取。重分析完成后 `artifactDirectory` 换新目录、`?v` 自然变化，无需额外处理。

### 3.6 `client/pages/AnalysisReport.tsx`（前端时序拆分）

将 `annotationUpdated` 的 effect（`AnalysisReport.tsx:324-338`）改为「先刷新图、再等预测」：

```ts
const data = await api.completeAnnotation(reportIdParam);
if (cancelled) return;
// ① 标注图已在服务端同步渲染好：立即重拉报告，秒级显示
try {
  const fresh = await api.getReport(reportIdParam);
  if (!cancelled) setReport(adaptReportData(fresh));
} catch { /* 忽略：拉取失败则等待重分析结果覆盖 */ }
// ② 继续等待重分析（Cobb/热力图）
const taskId = data?.reanalysisTaskId;
if (taskId) await pollTaskForReport(taskId, reportIdParam);
```

> `pollTaskForReport` 成功后已会 `api.getReport` + `setReport` + `navigate`，所以热力图/Cobb 仍由它刷新，逻辑不变。

---

## 4. 接口契约

### `POST /api/render`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | multipart file | 是 | ROI PLY（编辑后 ROI 或算法 roi.ply），≤100MB |
| `landmarks` | string(JSON) | 是 | 扁平 18 键 landmarks |
| `subject_id` | string | 否 | 输出目录名，用于原地覆盖 |

| 状态码 | 场景 |
|---|---|
| 200 | 成功，返回 `{ subject_id, outputs: { landmarks } }` |
| 400 | landmarks 非 JSON / subject_id 含路径分隔符 |
| 422 | 非 .ply / landmarks 缺键 / mesh 为空或无法解析 |
| 500 | 渲染管线内部异常 |

### 错误处理约定（Express 侧）

- `renderImages` 抛错 → `completed` catch 静默降级，**不改变** `completed` 对外返回（仍返回 `reanalysisTaskId`）。
- 无 `artifactDirectory`（异常报告）→ 跳过即时渲染，走原有重分析路径。

---

## 5. 路径契约（实施前提）

| 项 | 值 |
|---|---|
| 共享产物根 | `AIS_RESULTS_ROOT`；Python `paths.py:44` 与 Express `analysis-runner.ts:40` 都落在 `…/prediction-outputs/<subject_id>/` |
| 渲染 `subject_id` | **必须取 `basename(report.artifactDirectory)`**（=`<caseId>-<oldTaskId>`），实现原地覆盖 |
| landmarks 格式 | Express `flattenGroundTruth` 产出扁平 18 键，与 `render_back_landmarks`（`visualization/back_panels.py:105`）及 `complete_landmarks_flat` 一致 |
| 前置校验 | 部署时 `AIS_RESULTS_ROOT` 三个进程（Express / 标注平台 / 算法服务）指向同一根（现网预测流程已依赖，非新风险） |

---

## 6. 边界与降级

| 场景 | 行为 |
|---|---|
| 只改 landmark（没动 ROI） | `landmarks.png` 即时更新；热力图/moire 不变，等重分析刷新 |
| 改了 ROI（去衣） | `landmarks.png` 即时反映新体形；热力图/moire 仍旧（旧 ROI），重分析完成后统一刷新（Phase 1 接受） |
| 渲染失败（算法服务挂/ROI 异常） | `completed` catch 后照常返回，前端仍轮询重分析，行为与今天一致 |
| 报告无 `artifactDirectory` | 跳过即时渲染，空值守卫 |
| 从没标注过的报告 | `annotationImageVersion` 为空 → `annotatedImage` 仍走原有 `landmarks.png` URL，无回归 |

---

## 7. 测试用例清单

### Python 单测（`tests/prediction/test_render.py`，新增）
1. `render_landmarks_image` 正常路径：合法 ROI + 完整 landmarks → 生成 `landmarks.png`。
2. landmarks 缺键 → `ValueError`（缺键列表含缺失项）。
3. 空 mesh / 非法 PLY → `ValueError`。
4. 渲染产物与 `report.py` 的 `landmarks.png` 风格一致（尺寸 6×8、dpi 150、无坐标轴）。
5. **不加载模型/特征**：断言 `render.py` 无 `model_registry` / `feature_pipeline` / `report_waterfall` 导入。

### 接口测试（`tests/prediction/test_api.py` 追加）
6. `POST /api/render` 正常：200 + `outputs.landmarks` URL。
7. landmarks 缺键 → 422；非 JSON → 400；非 .ply → 422。
8. `subject_id` 含 `/` 或 `..` → 400（防目录逃逸）。

### Express / 端到端
9. 标注完成 → 报告页 `annotatedImage` 立即变化（`&av=` 变化触发重取）。
10. 重分析完成后 Cobb / 热力图刷新，`annotatedImage` 指向新目录。
11. 渲染失败（停算法服务）→ `completed` 仍 202 返回，前端仍能等重分析完成。
12. 标注平台再次打开同一 subject → `_find_algorithm_dir` 行为不变（未被污染）。

---

## 8. 不动的部分（非目标）

- `/api/landmarks`、`/api/predict`、`auto`、CLI `auto`：**全不动**。
- `analysis-runner.ts` 重分析逻辑（已用 `manualRoiPath + landmarks`）：**不动**。
- `outputs.roi` 契约（auto 模式仍返回，`annotation.ts` 回退逻辑继续有效）：**不动**。
- 报告页其余三张图（heatmap/moire/normalAngle）的展示与刷新：**不动**。
- Prisma schema / 迁移：**不动**（版本号存 `resultJson`）。

---

## 9. 回滚方式

- 本方案为纯增量；如需回滚，删除 `prediction/render.py`、`api.py` 中 `render_route` 与 `renderImages`，恢复 `annotation.ts` / `reports.ts` / `AnalysisReport.tsx` 的对应 diff 即可，无数据迁移、无目录结构变更。
