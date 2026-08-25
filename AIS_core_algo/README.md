# AIS — 青少年特发性脊柱侧弯无创筛查系统

> **A Non-Invasive Radiation-Free Screening System for Adolescent Idiopathic Scoliosis: Integration of Digital Moiré, Geometric Analysis and Machine Learning**
>
> 陈然 — 香港中文大学
>
> 预印本投稿至 *EClinicalMedicine*，2026 年 3 月

本仓库是上述论文的完整代码实现。系统通过 3D 背部扫描生成数字 Moiré 影像，提取曲率与表面高度等几何特征，利用拟共形映射进行标准化配准，构建不对称指数（Asymmetric Index / Z Index / Moiré Number），并使用梯度残差决策森林（GRDF）模型进行 AIS 严重程度融合预测。

---

## 目录

- [论文 Pipeline 与代码对应关系](#论文-pipeline-与代码对应关系)
- [目录结构](#目录结构)
- [环境构建](#环境构建)
- [快速开始](#快速开始)
- [运行测试](#运行测试)
- [项目配置](#项目配置)
- [图例标注](#图例标注)

---

## 论文 Pipeline 与代码对应关系

论文提出的系统由 **5 个顺序模块** 组成（对应 Section 3）：

### Module 1 — 3D 扫描数据获取与网格重建 (§3.1)

| 步骤 | 说明 | 代码位置 | 状态 |
|------|------|----------|------|
| 数据采集 | 3D 扫描仪获取背部点云 | 外部设备 | N/A |
| 去噪 | 点云去噪 | `mesh/preprocess/clean.py` (denoise_mesh) | ✅ 已实现 |
| 平滑 | 点云平滑 | `mesh/preprocess/clean.py` (smooth_mesh) | ✅ 已实现 |
| 离群点移除 | 移除运动伪影/噪声 | `mesh/roi/` | ✅ 部分实现 |
| 补洞 | 填补网格空洞 | `mesh/preprocess/clean.py` (fill_mesh_holes) | ✅ 已实现 |
| 刚体对齐 | 初始对齐 | `mesh/preprocess/alignment.py` (align_mesh) | ✅ 已实现 |
| Poisson 重建 | 点云→三角网格 | `utils/mesh.py → build_mesh()` | ✅ 已实现 |
| ROI 提取 | 背部区域提取 | `mesh/roi/` | ✅ 已实现 |

### Module 2 — 数字 Moiré 影像生成与 Moiré Number 计算 (§3.2)

| 步骤 | 说明 | 代码位置 | 状态 |
|------|------|----------|------|
| Moiré 影像生成 | 光学模拟生成条纹图 | ~~`moire/moire.py`~~（已移除） | — |
| 旋转 Moiré 动画 | 多角度旋转 GIF | ~~`moire/rotate_moire.py`~~（已移除） | — |
| Moiré Number 自动计算 | 计算不对称条纹数 | ~~`moire/moire_number.py`~~（已移除） | — |

### Module 3 — 几何分析与特征提取 (§3.3)

| 步骤 | 说明 | 代码位置 | 状态 |
|------|------|----------|------|
| 脊柱中线检测 | 背部凹陷中心线 | `landmarks/spine/core.py` (fit_spine_midline) | ✅ 已实现 |
| 解剖特征点提取 | 颈根/肩臂转点/腋窝/腰/脊柱 | `landmarks/` | ✅ 已实现 |
| 曲率计算 (κ_M, κ_G) | Mean / Gaussian 曲率 | `mesh/curvature.py` | ✅ 已实现 |
| 表面高度场 z(v) | 相对最佳拟合平面的高度 | `features/extractors/asymmetry/surface_height.py` | ✅ 已实现 |
| 调和映射参数化 | 标准化跨主体坐标系 | `parameterization/` | ✅ 已实现 |

### Module 4 — 不对称量化指标构建 (§3.4)

| 步骤 | 说明 | 代码位置 | 状态 |
|------|------|----------|------|
| 解剖分割（4 区域） | 胸椎/腰椎/肩/骨盆 | `mesh/roi/segmentation.py` | ✅ 已实现 |
| Asymmetric Index (AI) | 左右曲率不对称指数 | `features/extractors/asymmetry/asymmetric_index.py` | ✅ 已实现 |
| Z Index | 左右表面高度不对称指数 | `features/extractors/asymmetry/z_index.py` | ✅ 已实现 |

### Module 5 — 机器学习严重程度融合预测 (§3.5)

| 步骤 | 说明 | 代码位置 | 状态 |
|------|------|----------|------|
| GRDF 模型训练 | 梯度残差决策森林 | `modeling/models.py` | ✅ 实验中 |
| 不对称分析 notebook | 特征探索 | `modeling/models.py` | ✅ 实验中 |
| 特征工程 pipeline | 输入特征组装 | `modeling/training/feature_selector.py` | ✅ 已实现 |

## 生产路径

⚠️ 自动 ROI（去衣/去裤）与 landmark 目前仅作预标，**不是最终 GT**。

```
mesh/ROI/landmarks 预标 → annotation-platform 人工修正 → data/ground_truth/ 导出
  → 参数化 → 特征提取 → results/extraction/features_extraction/v0.1.0/ CSV
  → scheme 训练 → results/modeling/prediction/
  → export
```

---

## 预测（Cobb 角预测）

单 subject 推理（PLY → landmarks / cobb 预测 + 可视化报告）有 **脚本** 与 **HTTP API** 两种方式，均封装同一核心（`prediction/predict.py`），输入输出契约一致，只是渠道不同。

### 脚本方式（三模式，`--model` 缺省 v1.0.0）

```bash
# ① landmarks 预处理：PLY → ROI → landmark 检测 → landmarks.json（两段式第一段）
python -m prediction.cli landmarks --ply data/x.ply --subject S0001

# ② 预测：PLY + clinical + landmarks → cobb + 报告图（--landmarks 必填）
python -m prediction.cli predict --ply data/x.ply --subject S0001 \
    --clinical data/form/clinical_data.json \
    --landmarks prediction/outputs/S0001/landmarks.json

# ③ auto：PLY + clinical → 自动 landmarks → 预测（单步便捷入口）
python -m prediction.cli auto --ply data/x.ply --subject S0001 \
    --clinical data/form/clinical_data.json
```

### HTTP API 方式（FastAPI）

```bash
uvicorn prediction.api:app --host 0.0.0.0 --port 8000   # 交互文档 http://localhost:8000/docs
```

两个接口：`POST /api/landmarks`（预处理）+ `POST /api/predict`（预测，双状态：auto 自动检测 / 精确完整 landmarks；`model` 字段缺省 v1.0.0，历史 v0.1.0 可选）。

> **详细文档**：`prediction/README.md`（脚本三模式 + HTTP 双接口 + 完整调用/返回示例 + 报告图 + 特征链路 + 已知限制）

---

## 目录结构

```
prediction/               预测服务（prediction/predict.py 核心 + cli.py 脚本三模式 + api.py FastAPI）
commands/                  CLI 入口（batch/plot/evaluate/export）
features/                 特征提取与选择
  extractors/               basic / morphology / asymmetry
  selectors/                特征方案筛选
landmarks/                解剖特征点检测（6 类）
mesh/                     网格处理（ROI、曲率、清理）
modeling/                  建模域（模型+训练+评估）
  models.py                 模型
  training/                 训练策略、加权、HP 搜索
parameterization/         调和 UV 参数化
visualization/            matplotlib 渲染
utils/                    工具函数
tests/                    测试（含 numerics 数值黄金测试）
docs/                     项目文档
config.yaml               运行时配置（受检者列表 / 算法参数）
ais-cli.py                Pipeline CLI 入口（ROI → 特征工程 → 训练）
```

### 产物结构

离线产物统一落在 `results/` 下，按 `阶段/子阶段/subject|方案/文件` 分层；预测服务运行期产物在 `prediction/outputs/<subject_id>/`；日志在 `logs/`。路径常量单一来源 `utils/paths.py`，禁止硬编码路径字符串。

```
results/
├── roi/                    # ROI 提取（逐 subject）
├── landmarks/              # 地标可视化
├── ground-truth/           # 算法生成 GT（original.ply + ground_truth.json）
├── parameterization/       # 参数化
├── extraction/             # 特征工程（curvature/features/features_extraction/features_selection）
├── modeling/               # 建模（models/ + prediction 训练结果 + feature_importance + composite）
├── eval/                   # 评估（tables/cut_eval/evaluation）
├── export/                 # 导出打包（论文表格/图片/ZIP）
├── formulas/  figures/  cache/  labeling/  validation/
└── archive/                # 归档：一次性/历史产物
```

- 预测推理产物 → `prediction/outputs/<subject_id>/`（roi.ply / landmarks.json / prediction.json / report/，API/CLI 每次预测落盘）
- 运行日志 → `logs/`（`utils/logger.py` 管理）

`data/` 是只读输入（mesh/ground_truth/form/features），代码禁止写入。

---

## 环境构建

**前置要求**：Python 3.11+（推荐 3.12，`.python-version` 已锁定）、[uv](https://docs.astral.sh/uv/) 包管理器。

```bash
# 安装 uv（macOS/Linux）
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或用 pip
pip install uv

# 安装全部依赖（创建 .venv，同步 pyproject.toml + uv.lock）
uv sync
```

常用命令（在**项目根目录**运行——包间绝对导入依赖当前工作目录，跨目录运行需 `PYTHONPATH=.`）：

```bash
uv run python ais-cli.py --list-steps   # 管线 CLI
uv run pytest -q                        # 测试（tests/numerics 黄金测试依赖敏感 mesh 数据，缺失自动 skip）
uv run ruff check .                     # lint
uv run ruff format .                    # format
```

---

## 快速开始

```bash
# 1. 查看可用步骤与训练方案（依赖已由 uv sync 装好）
python ais-cli.py --list-steps
python ais-cli.py --list-schemes

# 2. 运行全管线（ROI → 特征工程 → 训练）
python ais-cli.py --step roi --step feature_eng --step train

# 3. 指定模型训练
python ais-cli.py --step train --model HistGBRT --train-scheme v0.1.0

# 4. 可视化调试（受检者 ID 为本地数据目录名，见 config.yaml）
python -m commands.plot --domain landmarks S0069
python -m commands.plot --domain parameterization S0069
python -m commands.plot --domain roughness S0069
```

> **受检者 ID**：`S####` 为匿名占位；本地运行前请将 `config.yaml` 中的示例 ID 替换为真实受检者编号。

> **导入方式**：本项目是 9 个顶层包组成的仓库（`commands/`、`features/`、`landmarks/` 等），包间用绝对导入互相引用，Python 从**当前工作目录**解析顶层包。因此请**在项目根目录运行**命令（`python -m commands.xxx` 或 `python ais-cli.py`）。若需从其他目录运行，设置 `PYTHONPATH` 指向项目根：
> ```bash
> PYTHONPATH=/path/to/project python commands/plot.py --domain landmarks S0069
> ```

---

## 运行测试

```bash
# 运行全部测试
uv run pytest -q

# 数值黄金测试（依赖真实扫描 mesh，见下方说明）
uv run pytest tests/numerics -q -m slow
```

⚠️ **mesh 数据说明**：`tests/data/numerics/mesh/` 下的真实扫描 mesh 属于敏感数据，**不随仓库分发**（已被 .gitignore 排除）。本地测试需手动放置 `STD_fuse_mesh_20250619.ply` 与 `roi_S0006.ply` 至该目录；缺失时相关 golden 测试自动 `skip`。

---

## 项目配置

运行时参数集中在根目录 `config.yaml`：

| 配置 | 说明 |
|------|------|
| `subjects` | 受检者列表（ID 匿名化示例 + mesh 路径） |
| `preprocess.bfs` | ROI 提取：角度/曲率/粗糙度阈值 |
| 其余算法参数 | 见 `config.yaml` 内注释 |

---

## 图例标注

| 标记 | 含义 |
|------|------|
| ✅ 已实现 | 核心逻辑已完成，可直接使用 |
| ✅ 实验中 | Notebook 形式的实验/原型代码 |
| ✅ 部分实现 | 主要功能已有，但缺少完整覆盖 |
| 🔲 占位 | 文件已建立但 `raise NotImplementedError` |

---

## 引用

```bibtex
@article{lin2026ais,
  title={A Non-Invasive Radiation-Free Screening System for Adolescent Idiopathic Scoliosis:
         Integration of Digital Moir\'{e}, Geometric Analysis and Machine Learning},
  author={Lin, Chenran},
  journal={EClinicalMedicine},
  year={2026}
}
```
