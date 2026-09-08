# 算法相关 API 报错清单

> 链路：前端 API → node 服务（analysis.ts / analysis-runner.ts / algorithm.ts）→ Python 算法服务 `POST {AIS_ALGORITHM_URL}/api/predict`（FastAPI，超时 300s）

## 一、触发入口（前端 API 调用）
| 调用 | 端点 | 用途 |
|---|---|---|
| `api.analyzeSingle` | POST /api/analysis/single | 对某受检者（默认最新 PLY）发起分析 |
| `api.analyzeBatch` | POST /api/analysis/batch | 批量分析 |
| `api.checkAnalysis` | GET /api/analysis/check/:caseId | 检查是否有基础信息+PLY |
| `api.getTask` | GET /api/tasks/:id | 轮询任务状态（前端 1s/次）|
| `api.completeAnnotation` | POST /api/annotation/reports/:id/completed | 标注完成后触发重分析 |
| `api.reanalyzeReport` | POST /api/annotation/reports/:id/reanalyze | 手动重新分析 |

## 二、入口请求本身的报错（尚未进入算法）
| 错误码 | 消息 | 触发场景 |
|---|---|---|
| 404 | 「Case not found」 | 受检者不存在/无权限 |
| 400 | 「Upload a PLY scan before analysis.」 | 该受检者没有任何 PLY 文件 |
| 409 | 「该文件已有分析任务进行中，请稍后再试。」 | 同文件已有 pending/running 任务（多端同时操作）|
| 202 | 任务创建成功（含 `tasks`/`skipped`） | batch 部分失败入 `skipped`（「Access denied」「No PLY scan」「Case not found」）|
| 403 | 「无权限执行此操作」 | 角色不符 |

## 三、算法服务调用错误（`algorithm.ts predict()`）
| 场景 | 抛出的错误（→ 任务 failed 的 failureReason）|
|---|---|
| **算法服务未启动/连接失败**（端口未起、地址错误）| 「AIS 核心算法服务不可用」|
| **算法执行超时**（>300s，`AbortSignal.timeout`）| 同样抛「AIS 核心算法服务不可用」（⚠️ 超时被误报为"不可用"，未区分）|
| **Python 返回 HTTP 4xx/5xx** | `payload.detail`（FastAPI 错误详情）或「AIS 核心算法分析失败」|
| 响应体非 JSON | 「AIS 核心算法分析失败」|

## 四、任务执行（`runAnalysisTask`）的失败来源（→ failureReason）
| 阶段 | 失败原因 |
|---|---|
| 读取任务/文件 | 任务不存在/已取消 → 静默返回（不报错）|
| 输入路径 | `manualRoiPath`（标注编辑后的 ROI）缺失/不可读 → predict 读文件抛错 |
| predict 调用 | 见第三节 |
| **结果字段缺失** | Python 返回缺 `subject_id`/`cobb`/`severity` → `path.join(undefined)` 抛 TypeError 或 `Number(undefined)=NaN`（⚠️ 可能产生 NaN 报告值）|
| 事务写入报告 | 数据库错误（罕见）|
| 兜底 | 「AIS prediction failed.」|

## 五、报告产物（算法生成的图片）相关
| 场景 | 错误 |
|---|---|
| 结果目录 `prediction-outputs/{subject_id}/report/*.png` 未生成（Python 未产出/目录被清）| 报告页图片 404「Report image has not been generated」|
| 图片名非法 | 404「Report image not found」|

## 六、标注重分析链路的算法交互
| 场景 | 错误 |
|---|---|
| 无标注真值（`ground-truth/{caseId}/ground_truth.json` 缺失）| completed → 409「No saved annotation ground truth was found」|
| 编辑后 ROI 文件缺失（`labeling/cache/{caseId}/extract_roi/`）| completed → 409「The source ROI artifact is unavailable for manual reanalysis.」|
| 标注数据无效（landmarks 校验失败）| completed → 422 数据无效 |
| 有进行中任务再点重新分析 | reanalyze → 202 幂等（返回进行中任务 id，前端等待）|
| 算法服务不可用/失败 | 任务 failed（见三、四节）|

## 七、前端对算法失败的展示点
| 位置 | 行为 |
|---|---|
| 受检者详情「开始分析/重新分析」 | 轮询任务，failed → `alert(failureReason)`（展示真实算法错误）|
| 报告详情「重新分析」 | 轮询任务，failed → 仅 `alert('重新分析失败，请稍后重试')`（⚠️ 不展示 failureReason，丢失真实原因）|
| 标注完成自动重分析 | 轮询 failed → 静默保留当前数据（用户可手动重试）|
| 任务列表/统计 | 任务 `failed` 状态展示；统计成功率按 failed 计数 |

## 八、潜在优化点（待确认）
1. **区分算法超时与不可用**：超时（300s）目前抛「AIS 核心算法服务不可用」，应改为「算法分析超时，请重试」（可把 AbortError 单独捕获）。
2. **报告页重分析失败提示丢失原因**：`pollTaskForReport` 失败仅提示"重新分析失败"，建议把 `failureReason` 一并展示。
3. **Python 结果字段缺失校验**：`result.cobb`/`subject_id` 缺失时目前可能产生 NaN 报告，建议入库前校验并置为失败。
