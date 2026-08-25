# AIS Landmark 标注平台

面向 AIS 核心算法管线的 3D 网格人工标注工具。对核心仓库（`AIS_core_algo`）产出的**预标 landmark** 与 **ROI 区域**进行人工修正，修正结果作为最终 Ground Truth 回传核心仓库，用于参数化、特征提取与模型训练。

---

## 功能

- **受检者列表**：浏览数据目录下的所有受检者及标注状态
- **3D 网格查看**：Open3D 渲染背部网格（含穿衣/去衣覆盖层）
- **Landmark 人工标注**：颈根、肩臂转点、腋窝、腰、肩胛峰、骨盆线等解剖特征点
- **ROI 标注**：背部区域（去衣/去裤）人工修正
- **曲率可视化**：曲率影像 / 曲率映射覆盖
- **指标计算**：标注质量度量
- **数据导出**：导出标注结果为 GT 数据

---

## 架构

```
┌─────────────┐   HTTP (/api/*)   ┌──────────────────┐
│  前端        │ ───────────────▶ │  后端 FastAPI     │
│  React/Vite │                  │  Open3D + Trimesh │
│  :5173      │ ◀─────────────── │  :8765            │
└─────────────┘                  └──────────────────┘
        │                                │
        └────────── 读取 mesh / 标注 ────┘
                          │
                   data/  +  results/（可 env 配置）
```

- **后端**：FastAPI 纯 API（`backend/main.py`），`open3d`/`trimesh` 处理网格，`scipy`/`matplotlib` 计算曲率。
- **前端**：React + Vite（`frontend/src/`），`zustand` 状态管理，Three.js 网格渲染。
- **数据路径**：通过环境变量 `AIS_DATA_ROOT` / `AIS_RESULTS_ROOT` 指向数据与结果目录（默认指向同级 core 仓库的 `data/` 与 `results/`）。

---

## 快速开始

```bash
# 一键启动后端 + 前端
bash launch.sh start

# 其他子命令
bash launch.sh stop                # 停止全部
bash launch.sh restart             # 重启全部
bash launch.sh status              # 查看运行状态 + 健康检查
bash launch.sh health              # 健康检查
bash launch.sh logs backend        # 查看后端日志
bash launch.sh logs frontend       # 查看前端日志
```

启动后访问 **http://localhost:5173**（前端）/ **http://localhost:8765**（后端 API）。

---

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `AIS_DATA_ROOT` | 核心数据目录（mesh、标注） | `../core/data` |
| `AIS_RESULTS_ROOT` | 结果输出目录 | `../core/results` |

---

## 后端 API

| 端点 | 说明 |
|------|------|
| `GET /api/health` | 健康检查 |
| `GET /api/subjects` | 受检者列表 |
| `GET /api/subjects/{id}` | 受检者详情与标注状态 |
| `GET /api/subjects/{id}/curvature-image` | 曲率影像 |
| `GET /api/subjects/{id}/curvature-mapping` | 曲率映射 |
| `GET /api/subjects/{id}/contours` | 轮廓线 |
| `GET /api/subjects/{id}/mesh` | 网格数据（支持 `clothed` 参数） |
| `GET /api/subjects/{id}/mesh/overlay-cloth` | 穿衣覆盖层 |
| `GET/PUT /api/subjects/{id}/landmarks` | Landmark 标注读取 / 保存 |
| `POST /api/subjects/{id}/landmarks/reset` | 重置标注为预标 |
| `POST /api/subjects/{id}/landmarks/lift` | 标注提升（对齐） |
| `POST /api/subjects/{id}/landmarks/validate` | 标注校验 |
| `GET /api/subjects/{id}/metrics` | 标注质量指标 |
| `GET /api/subjects/{id}/validate` | 受检者校验 |
| `POST /api/batch/generate` | 批量生成预标 |
| `GET /api/export/csv` | 导出标注为 CSV |
| `POST /api/export/csv-batch` | 批量导出 CSV |
| `POST /api/export/data-export` | 发起数据导出任务 |
| `GET /api/export/data-export/{task_id}` | 查询导出任务结果 |

---

## 目录结构

```
backend/                  FastAPI 后端
  api/                     REST 路由（subjects/landmarks/metrics/export）
  services/                领域服务（converter、curvature、subject_loader、lifter/）
  utils/                   mesh 工具、日志
  main.py                  FastAPI 入口
frontend/                 React 前端
  src/
    api/                   API 客户端
    components/            组件（3D 查看器、标注工具栏、面板…）
    pages/                 页面
    stores/                zustand 状态
    types/                 类型定义
scripts/                  数据整理脚本
launch.sh                  一键启动/停止/状态脚本
results/labeling/          运行时日志目录
```

---

## 依赖安装

```bash
# 后端（Python ≥ 3.11）
uv sync                    # 或 pip install -e .

# 前端
cd frontend && npm install
```

---

## 与核心仓库的数据流

```
AIS_core_algo 预标
      │
      ▼
┌─ 标注平台（人工修正 landmark + ROI）──┐
      │
      ▼
Ground Truth 导出（data/ground_truth/）
      │
      ▼
参数化 → 特征提取 → 模型训练（AIS_core_algo）
```
