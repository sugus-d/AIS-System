# AIS 系统前端

青少年特发性脊柱侧弯（AIS）无创筛查系统的 Web 管理端。面向医院/筛查机构提供受检者建档、背部扫描文件上传、AI 分析、报告生成、统计与任务管理的完整工作流，并与核心算法仓库（`AIS_core_algo`）的分析结果对接。

---

## 功能特性

| 模块 | 说明 |
|------|------|
| 登录 / 绑定 | 用户认证与设备绑定 |
| 工作台 | 数据总览（受检者、分析、任务状态） |
| 病例记录 | 单个 / 批量建档 |
| 病例管理 / 详情 | 受检者档案与三维网格查看 |
| 文件上传 | 扫描文件上传（单文件 / 批量） |
| 分析 | AI 分析任务发起与进度 |
| 分析报告 | 报告详情（背部影像 / 热力图 / Moiré 等） |
| 报告 / 统计 | 结果报表与数据统计 |
| 任务管理 | 分析 / 处理任务列表与详情 |
| 帮助 / 个人设置 | 使用帮助与账号设置 |
| 管理后台 | 用户管理、API 配置、系统设置 |

---

## 技术栈

- **前端**：React 18 + Vite + TypeScript + TanStack Query + Tailwind CSS + shadcn/ui
- **服务端**：Express（内置 mock API，路径前缀 `/aisapi`，默认端口 `8080`）
- **脚手架**：基于 Builder.io 生成（含 `.builder/` 规则与 `VITE_PUBLIC_BUILDER_KEY`）

---

## 目录结构

```
client/                    React 前端
  pages/                    页面组件（登录、病例、分析、报告、统计、任务、管理后台…）
  components/               通用组件
  lib/                      API 客户端与工具
  hooks/                    React Hooks
server/                    Express 服务端
  routes/                   REST 路由（auth/users/cases/files/analysis/reports/…）
  data/mock.ts              内置 mock 数据（22 个受检者）
  node-build.ts             生产静态托管（服务 /ais/ 前缀 SPA）
shared/                    前后端共享类型
docs/                      产品文档与 API 约定（01-产品概述、核心业务、API文档、截图）
public/                    静态资源（报告示例图等）
```

---

## 快速开始

```bash
# 1. 安装依赖
npm install

# 2. 启动开发服务器（Vite dev server）
npm run dev

# 3. 构建 + 生产启动（Express 托管，默认端口 8080）
npm run build
npm start                  # 或 PORT=9000 npm start
```

> mock 数据自带完整业务流（登录 / 建档 / 上传 / 分析 / 报告），无需连接外部服务即可体验全部页面。

---

## 环境变量

`.env` 文件（已被 git 追踪的占位值，非真实密钥）：

| 变量 | 说明 |
|------|------|
| `VITE_PUBLIC_BUILDER_KEY` | Builder.io 公共 key（占位 `__BUILDER_PUBLIC_KEY__`） |
| `PING_MESSAGE` | 健康检查返回消息 |

生产环境如需真实后端，可在 `server/routes/*.ts` 中将 mock 替换为对算法服务的 HTTP 调用（见 `docs/API文档/`）。

---

## 测试

```bash
npm test                    # vitest 单元测试
npm run typecheck           # TypeScript 类型检查
```

---

## 部署

- **生产构建**：`npm run build` 产出 `dist/spa/`（静态）与 `dist/server/`（Express）。
- **启动**：`node dist/server/node-build.mjs` 托管 SPA（服务路径前缀 `/ais/`）。
- **Serverless**：`netlify/functions/` 提供 Netlify Functions 适配。

---

## 文档

详细产品说明见 `docs/`：

- `01-产品概述与文档导航.md`
- `核心业务/` — 受检者管理、文件上传、算法调用、报告分析
- `其他页面/` — 登录、工作台、任务管理、数据统计、用户管理
- `API文档/` — 算法 API 约定
- `screenshots/` — 各页面截图
