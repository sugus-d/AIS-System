# AIS Frontend

本文件的规则会在每次 Claude Code 会话开始时自动加载。

一个全栈 React 应用，用于案件管理和分析系统。前端使用 React Router 6 SPA 模式，后端集成 Express 服务器。

## 端口和 Nginx 配置（勿轻易修改）

### 内网环境
- 前端：http://192.168.1.6:8080/ais/
- 后端：http://192.168.1.6:8080/aisapi/

### 外网环境
- 前端：https://home.xiaokubao.space/ais/
- 后端：https://home.xiaokubao.space/aisapi/

> ⚠️ 重要：禁止修改前后端端口和 nginx 配置

## Tech Stack

- **PNPM**: 使用 pnpm 作为包管理器
- **Frontend**: React 18 + React Router 6 (SPA) + TypeScript + Vite + TailwindCSS 3
- **Backend**: Express 服务器集成 Vite 开发服务器
- **Testing**: Vitest
- **UI**: Radix UI + TailwindCSS 3 + Lucide React icons

## Project Structure

```
client/                   # React SPA frontend
├── pages/                # 路由页面组件
│   └── admin/            # 管理页面 (Users, ApiConfig, Settings)
├── components/ui/        # UI 组件库
├── lib/                  # 工具函数和 API 客户端
├── App.tsx               # 应用入口和 SPA 路由配置
└── global.css            # TailwindCSS 主题和全局样式

server/                   # Express API backend
├── index.ts              # 主服务器配置
├── routes/               # API 路由处理器
└── data/                 # Mock 数据
```

## 开发命令

```bash
pnpm dev        # 启动开发服务器 (client + server)
pnpm build      # 生产构建（开发阶段不要使用，用户自己构建）
pnpm start      # 启动生产服务器
pnpm typecheck  # TypeScript 类型检查
pnpm test       # 运行 Vitest 测试
```

开发阶段禁止使用 `pnpm build`，用户会自己构建

## 路由系统

### 前端路由 (basename: `/ais`)

**认证路由:**
- `/login` - 登录页面
- `/bind` - 绑定页面

**受保护路由:**
- `/dashboard` - 仪表盘
- `/case-record` - 案件录入
- `/batch-case-record` - 批量案件录入
- `/cases` - 案件管理
- `/case-detail/:caseId` - 案件详情
- `/file-upload` - 文件上传
- `/analysis` - 分析页面
- `/analysis-report` - 分析报告
- `/reports` - 报告管理
- `/statistics` - 数据统计
- `/tasks` - 任务管理
- `/help` - 帮助中心
- `/settings` - 个人设置

**管理路由:**
- `/admin/users` - 用户管理
- `/admin/api-config` - API 配置
- `/admin/settings` - 系统设置

### API 路由 (prefix: `/aisapi`)

- `GET /aisapi/ping` - 健康检查
- `/aisapi/auth` - 认证相关
- `/aisapi/users` - 用户管理
- `/aisapi/cases` - 案件管理
- `/aisapi/files` - 文件管理
- `/aisapi/analysis` - 分析相关
- `/aisapi/reports` - 报告相关
- `/aisapi/tasks` - 任务相关
- `/aisapi/statistics` - 统计数据
- `/aisapi/settings` - 系统设置
- `/aisapi/profile` - 个人资料
- `/aisapi/help` - 帮助相关

## Styling

- **TailwindCSS 3**: 使用 utility classes
- **主题配置**: `client/global.css`
- **UI 组件**: `client/components/ui/`
- **`cn()` 工具函数**: 结合 `clsx` + `tailwind-merge`

## 测试要求

所有代码必须覆盖测试：

1. **前后端单独测试**：使用 vitest/jest 对前端和后端分别进行单元测试
2. **真实访问页面测试**：
   - 使用 curl 读取页面标题、返回值、文档结构
   - 如果 curl 读取失败，改用 browser (playwright)
3. **截图测试**：调用 playwright 截图并识别页面内容是否有效

> 完成代码后，请自行安排并执行上述所有测试。
