# 当前系统报错清单（全量）

> 覆盖：后端 API 错误分支（65 路由/58 错误分支）、前端 UI 错误提示、系统级/网络错误、标注平台入口。
> 标注：`[前端提示]`= 前端本地文案；`[后端]`= 后端返回 message；`[系统]`= 桌面进程级。

## 一、全局层（所有请求）
| 类型 | 提示/错误码 | 触发场景 |
|---|---|---|
| [系统] 网络异常 | 「无法连接本地服务，请确认应用已正常启动后重试。」 | node 服务未启动/端口未就绪，`fetch` 抛 TypeError（已统一友好化，简/繁双语）|
| [后端] 401 | 「请先登录」「登录已失效」 | 未带/无效 token、用户被禁用（重启客户端后 sessionStorage 清空需重新登录）|
| [后端] 403 | 「无权限执行此操作」 | 角色不满足（operator 访问用户管理/备份等）|
| [后端] 500 | 数据库异常信息 | SQLite 损坏/磁盘异常（罕见）|
| [前端] | 「请求失败」 | HTTP 非 2xx 且响应体无 message |

## 二、认证 auth
| 端点 | 报错 |
|---|---|
| POST /login | 400「请输入用户名和密码」；401「用户名或密码错误」（含账号被禁用）|
| GET /me | 401「登录已失效」|
| POST /password | 400「原密码错误」；400「新密码至少需要 12 位」|

## 三、受检者 cases
| 端点 | 报错 |
|---|---|
| GET /cases | 分页参数非数字时异常分页（低风险）|
| GET /cases/:id | 404「Case not found」|
| POST /cases | 400 创建失败（caseNumber 冲突/字段校验）|
| PUT /cases/:id | 404；400 更新异常 |
| DELETE /cases/:id | 404；文件删除失败静默忽略 |
| POST /cases/batch | 部分失败入 `skipped`（不报错）|

## 四、文件 files
| 端点 | 报错 |
|---|---|
| POST /files | 400「A file is required」；422「仅支持 .ply 扫描文件或 .jpg/.png/.webp 影像」；404「Case not found」；500 存储失败（磁盘满/权限）|
| GET /files/:id/download | 404「File not found」「File content missing」|
| DELETE /files/:id | 404「File not found」|

## 五、分析 analysis
| 端点 | 报错 |
|---|---|
| POST /analysis/single | 404「Case not found」；400「Upload a PLY scan before analysis」；409「该文件已有分析任务进行中」|
| GET /analysis/check/:caseId | 404 |
| 任务结果 | 任务 `failed` → 轮询看到 `failureReason`（真实算法失败：PLY 无效/算法异常等）→ 前端「真实算法分析失败」|

## 六、任务 tasks
| 端点 | 报错 |
|---|---|
| GET /tasks/:id | 404「Task not found」|
| POST /tasks/:id/cancel | 404；400「Task cannot be cancelled」（非 pending/running）|
| POST /tasks/:id/retry | 404；400「Task cannot be retried」（非 failed/cancelled）|

## 七、报告 reports
| 端点 | 报错 |
|---|---|
| GET /reports/:id | 404「Report not found」|
| GET /reports/:id/images/:image | 404「Report image not found」「Report not found」「Report image has not been generated」（分析未完成/结果目录被清）|
| PUT /reports/:id/diagnosis | 404 |
| GET /:id/versions、/export | 404 |

## 八、标注 annotation
| 端点 | 报错 |
|---|---|
| GET /annotation/subjects | 403（仅系统管理员）|
| POST /annotation/sessions | 404「Report not found」|
| POST /reports/:id/completed | 404；409「No saved annotation ground truth was found」；409「The source ROI artifact is unavailable」；422 标注数据无效 |
| POST /reports/:id/reanalyze | 404；202 幂等（有进行中任务返回其 id，前端等待并刷新）|
| [前端] | 「无法打开标注工具」（创建会话失败）；标注服务未启动时 sessionUrl 打不开 |

## 九、审核 reviews / 统计 statistics
- reviews：403（非管理员）；404「Report not found」
- statistics：无错误分支；time-series 遇 createdAt 为空可能 500（低概率）

## 十、用户 users
| 端点 | 报错 |
|---|---|
| GET /users、/:id | 403；404「User not found」|
| POST /users | 400 用户名格式非法/缺机构/密码<12/「无效的角色」；409「用户名已存在，请换一个。」|
| PUT /users/:id | 404；400 用户名/密码校验；409 用户名冲突 |
| DELETE /users/:id | 400「Cannot delete the current user」「该用户名下还有受检者档案…」；404 |
| POST /toggle-status | 404 |

## 十一、设置/个人/备份/帮助
- settings：PUT 403（仅系统管理员）
- profile：GET 401；PUT password 400「Current password is invalid」「New password must match and contain at least 12 characters」
- backups：POST 500「Backup failed」；restore 422「Restore failed」
- help：GET /:id 404「文档不存在」；POST feedback 400「标题和反馈内容不能为空」、500「反馈保存失败」

## 十二、前端页面 UI 错误提示（catch 分支）
| 页面 | 提示 |
|---|---|
| 登录页 | 「请输入账号和密码。」「登录失败。」+ 后端 message |
| 受检者详情 | 「加载失败」「病例不存在」「分析失败」「删除失败」「请选择 PLY 格式的筛查文件。」「请选择 JPG / PNG / WebP 格式的 X 光影像。」「文件上传失败。」|
| 报告详情 | 「获取报告失败」「未找到相关报告」「保存失败，请重试」「重新分析失败，请稍后重试」「审核操作失败，请重试」「PDF 导出失败，请重试。」「无法打开标注工具」|
| 受检者管理 | 「加载失败」|
| 建档/编辑 | 「请填写姓名、性别、生日、身高和体重」「保存失败，请重试」|
| 数据统计 | 「统计数据加载失败」|
| 个人设置 | 「个人资料加载失败。」「姓名和科室为必填项。」「请完整填写密码信息。」「新密码至少需要 12 位。」「两次输入的新密码不一致。」「个人资料保存失败。」「密码修改失败。」|
| 用户管理 | 「用户列表加载失败」「操作失败，请重试。」「用户名需为 2-32 位字母、数字或 . _ -…」「请填写姓名。」「新密码至少需要 12 位。」「创建用户失败。」「保存失败。」|
| 系统设置 | error.message 直接显示（含「Backup failed」「Restore failed」）|
| 3D 查看 | 「3D 模型加载失败」|
| 通用组件 | 「加载中...」「暂无数据」「共 N 条记录，第 x/y 页」|

## 十三、标注平台（annotation-platform 独立后端，入口级）
- 打开标注工具：会话创建成功但标注服务未启动 → 页面打不开（浏览器层错误）
- 标注平台内部 API（/lift、/landmarks、/validate、/subjects/:id/mesh、curvature 等）：mesh/ROI/缓存缺失 → 4xx/5xx，平台内 toast 展示
