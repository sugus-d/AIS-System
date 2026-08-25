> **⚠️ HISTORICAL DOCUMENT** — Last updated 2026-05. Directory structure
> and pipeline design have changed significantly. For current state see
> `docs/refactor/2026-07-28-surgical-cleanup-refactor-design.md` and `README.md`.

# AIS — 青少年特发性脊柱侧弯非侵入性筛查系统

注意：为了提高可维护性，仓库核心代码的公开函数已补充中文 Google 风格 docstring 与关键算法的 WHY 注释；在阅读本 PRD 的配置/参数细节时，建议同时参考各模块源码中的 docstring 获取更精确的参数含义与使用限制。

## 产品需求文档 (PRD)

> **版本**: 1.0  
> **日期**: 2026-05-04  
> **状态**: 草稿  
> **论文参考**: *A Non-Invasive Radiation-Free Screening System for Adolescent Idiopathic Scoliosis: Integration of Digital Moiré, Geometric Analysis and Machine Learning*（投稿至 EClinicalMedicine, 2026 年 3 月）

---

## 1. 项目概述

### 1.1 目标

基于 3D 背部表面扫描，构建一套无创、无辐射的青少年特发性脊柱侧弯（AIS）筛查系统。系统从 3D 扫描生成数字 Moiré 影像，提取几何特征，计算不对称指标，并利用机器学习模型进行严重程度预测。

### 1.2 输入 / 输出

| | |
|---|---|
| **输入** | 3D 背部表面网格（PLY 格式，来自结构光或深度相机扫描） |
| **输出** | 不对称指标（AI, Z Index, Moiré Number）+ 预测严重程度评分 |
| **目标用户** | 临床研究人员、骨科筛查项目 |

### 1.3 成功标准

- 单条命令即可端到端处理一个受试者
- 全部五个论文模块（M1–M5）产出可复现的结果
- 特征提取结果与论文公式（Eq. 8–11）一致
- ML 模型达到临床有意义的严重程度预测效果

---

## 2. 系统架构

### 2.1 Pipeline 概览

```
[PLY Mesh] → M1: 预处理 → M2: Moiré → M3: 几何分析 → M4: 指标计算 → M5: ML
                                                                       ↓
                                                               严重程度评分
```

每个模块的详细 PRD 见 `docs/modules/`。

### 2.2 架构决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| Pipeline | 混合方案 — 独立模块 + 统一 Pipeline 编排 | 每步可独立调试；通过配置支持批量处理 |
| 配置管理 | 统一 `config.yaml` | 所有参数集中管理，运行可复现 |
| Mesh 交换格式 | `.ply`（Open3D 原生） | 通用格式，无序列化依赖 |
| 特征数据 | pandas DataFrame | 统一接口，ML 模块直接消费 |
| 缓存 | 每步 `.pkl` / `.npz` | 支持断点续跑 |
| 配准 | 简化方案（正方形模板 + 特征点对齐） | QC 配准暂不实现；TPS/ARAP baseline |
| ML 框架 | sklearn Pipeline | 清晰的训练/预测抽象 |

### 2.3 目录结构

```
AIS/
├── ais-cli.py                   # 管线 CLI 入口（--step roi/feature_eng/train）
├── config.yaml                  # 统一 Pipeline 配置
├── commands/                    # CLI 入口脚本（batch/plot/evaluate/export）
│   └── pipeline.py              # 管线编排器（feature_eng → train）
├── features/                    # 特征提取与选择（extractors/ selectors/）
├── landmarks/                   # 解剖标志点检测（6 类子包）
├── mesh/                        # 网格处理（clean/curvature/preprocess + roi/）
├── modeling/                    # ML 训练（training/ models.py metrics.py contracts.py）
│   └── contracts.py             # 共享契约与常量
├── parameterization/            # 调和 UV 参数化
├── reports/                     # Streamlit 报表
├── utils/                       # 通用工具（Mesh I/O, logger）
├── visualization/               # matplotlib 渲染面板
├── experiments/                 # 历史实验代码
├── docs/
│   ├── PRD.md                   # 本文件（历史文档，结构见 README.md）
│   └── modules/                 # 各模块详细 PRD
│       ├── M1_preprocessing.md
│       ├── M2_moire.md
│       ├── M3_geometry.md
│       ├── M4_indices.md
│       └── M5_ml.md             # M5 模块现为 modeling/（文件名史实保留）
├── results/
└── tests/                       # 按领域分组（mesh/features/modeling/…）
```

> 注：`labeling/`（标注平台）已拆为独立仓库；`pipeline/`、`legacy/`、`moire/` 目录已删除（编排并入 `commands/pipeline.py`，契约并入 `modeling/contracts.py`）。

---

## 3. Pipeline Runner 规格

### 3.1 入口（2026-08 重构后）

> 旧 `run_pipeline.py` 已移除，入口拆分为两个 CLI（均从项目根运行）：

```
# 全管线编排（feature_eng → train，可指定 --step 多次）
python ais-cli.py --step feature_eng --step train
python ais-cli.py --step roi --roi-algo bfs --step train --model HistGBRT

# 参数化（单受试者，调和 UV）
uv run python -m commands.plot --domain parameterization S0069
```

### 3.2 `config.yaml`

```yaml
pipeline:
  cache_dir: results/cache

subjects:
  - id: "S0069"
    mesh_path: "data/raw/S0069/STD_fuse_mesh_20250619.ply"
    clinical:
      age: 14
      sex: F
      bmi: 19.5

preprocess: { ... }
moire: { ... }
curvature: { ... }
geometry: { ... }
features: { ... }
ml: { ... }
```

### 3.3 缓存机制

- 每步中间结果写入 `results/cache/{subject}/{step_name}/`
- 支持类型：`.ply`（网格）、`.npz`（数组）、`.pkl`（DataFrame）、`.json`（元数据）
- Pipeline 在重新运行已完成步骤前检查缓存

---

## 4. 数据流

```
受试者 PLY
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  M1: load_mesh → preprocess → extract_roi          │
│       → denoise → smooth → fill_holes → align      │
│  输出: clean_roi.ply                                │
└─────────────────────────────────────────────────────┘
    │
    ├────────────────────────────────┐
    ▼                                ▼
┌──────────────────┐   ┌──────────────────────────────┐
│ M2: get_moire_img │   │ M3: calculate_curvature     │
│     find_spine    │   │     extract_landmarks        │
│     count_fringes │   │     compute_surface_height   │
│ 输出: M.png, M    │   │     segment_template         │
└──────────────────┘   │     register_to_template     │
    │                   │ 输出: curv, lmks, h, seg    │
    │                   └──────────────────────────────┘
    │                              │
    └──────────────┬───────────────┘
                   ▼
┌──────────────────────────────────────┐
│ M4: compute_asymmetric_index         │
│     compute_z_index                  │
│ 输出: AI, AI_seg, Z, Z_seg, M       │
└──────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ M5: build_feature_row                │
│     → predict                        │
│ 输出: 严重程度评分                    │
└──────────────────────────────────────┘
```

---

## 5. 依赖

| 包 | 版本 | 用途 |
|----|------|------|
| open3d | ≥0.19 | Mesh I/O、ICP、可视化 |
| pyvista | ≥0.44 | 曲率计算、孔洞填充 |
| numpy | ≥1.26 | 核心数值计算 |
| scipy | ≥1.12 | KDTree、插值、信号处理 |
| matplotlib | ≥3.10 | 可视化 |
| scikit-image | ≥0.24 | 图像处理（Otsu、形态学） |
| scikit-learn | ≥1.5 | ML pipeline、预处理 |
| pandas | ≥2.2 | 特征 DataFrame |
| lightgbm | ≥4.5 | baseline ML 模型 |
| loguru | ≥0.7 | 日志 |
| pyyaml | ≥6.0 | 配置管理 |
| imageio | ≥2.37 | GIF 生成 |

---

## 6. 测试策略

| 层级 | 范围 | 工具 |
|------|------|------|
| 单元测试 | 各函数（曲率、指标、标志点） | pytest |
| 集成测试 | 在已知受试者上端到端运行 | pytest |
| 回归测试 | 代码变更后的输出一致性 | pytest + 保存的参考输出 |
| 可视化审计 | 含诊断图的 Pipeline PDF | `pipeline_audit.py` |

---

## 7. 后续工作（论文投稿后）

- 完整拟共形映射配准实现（LSCM + LBS）
- 实时推理 API（FastAPI）
- 扩展训练数据与交叉验证
- 移动端 / 临床部署
