> ⚠️ **历史设计文档**。命名已版本化演进（ml/→modeling/、back_v1→v0.1.0、manual_roi→v1.0.0、presets 已并入 TRAINING_SCHEMES、scheme key 统一版本号），仅作历史设计参考，不再反映当前代码。

# AIS 外科清理重构设计（方案 A）

Date: 2026-07-28  
Status: **✅ 已完成**（2026-08-03 P0-P4 全部落地，31 commit；后续 08-04/08-06 持续演进，见 `.claude/docs/2026-08-02-architecture-redesign.md` §实施差异）  
Prerequisite: 用户选定「外科清理 A」——功能零回归，**本阶段不改** landmark/ROI 数值算法  
Supersedes (partial): `2026-07-05-engineering-refactoring-design.md` 中未完成的收口项  
Related status: `.claude/PROJECT.md`「生产现状」

---

## 1. 动机

2026-07-05 工程化重构已完成「功能模块 = 目录」的主迁移（`features/`、`modeling/`、`reports/`、`commands/`），但代码库仍处**半重构态**：

1. **三套管线并存**：旧 per-subject `pipeline/{core,steps,cache}`（DEPRECATED 但仍被测试锁死）、新 `ais-cli → pipeline/run`（roi/feature_eng 为空壳）、领域子管线 `mesh.roi / landmarks / parameterization`（预标用）。
2. **双 registry / 双 scheme**：`roi/registry` ≡ `mesh/roi/registry`；`ml/schemes.py`（特征方案注册表，现并入 `features.selectors.schemes`）与 `features.selectors.schemes` 环依赖；presets 双入口。
3. **死代码与迁移未收口**：`mesh/roi_old/`、`tools/→commands/` 未提交、ML 实验残骸、commands 内 sweep/tune 与生产混放。
4. **文档大面积漂移**：README/PRD 仍指向不存在的 `analysis/`、`curvature/`、`ml_models/`；且把自动 landmark 标成 stable，与生产不符。

目标：**在保持现有功能与数值结果不变的前提下**，删除认知税、合并双轨、让名字与行为一致。不做大爆炸重写，不追求「真·端到端单入口」（那是方案 B，另案）。

### 1.1 生产现实（约束本 spec）

自动 **ROI/decloth** 与 **landmark** 结果目前**均不令人满意**。可复现预测路径是：

```
算法预标 → labeling 人工改去衣物 ROI + 调 landmarks → 导出 GT
  → 参数化 / 特征 / 训练 / 报告
```

- GT 权威：`labeling/` → `data/ground_truth/{sid}/`
- 自动几何 = 预标草稿，不是生产输入
- **下一阶段**（本 spec 之外）：优化 ROI/decloth + landmark，降人工
- **本 spec**：只做工程清理，为下一阶段算法迭代清场；验收锚在**现有人工 GT 下游**（特征表/训练 metrics），不要求自动几何变好

---

## 2. 系统功能地图（真源）

### 2.1 一句话

Cobb 主业务 = **人工 GT（ROI+landmarks）→ 参数化 → 离线特征 CSV → scheme 加载 → 加权 CV 训练 → reports/export**。  
自动几何仅作 labeling 预标。

### 2.2 三段式数据流（现实）

```
[Geometry draft — 预标，质量未过关]
  data/mesh/{sid}/*.ply
    → mesh.roi.pipeline / extract_back_roi   # 含 decloth/pants，自动结果不满意
    → landmarks.extract                      # 自动结果不满意
    → labeling 平台载入预标

[GT stage — 生产权威，人工]
  labeling: 手改去衣物 ROI + 调整 landmarks
    → export → data/ground_truth/{sid}/      # roi.ply, landmarks.csv, …

[Feature stage — 吃 GT，离线]
  parameterization.pipeline (UV，默认读 ground_truth)
    → features.extractors.assemble.extract_all
    → results/features_extraction/back_v1/{basic,morphology,region_asymmetry}.csv

[ML stage — N subjects 表]
  features.selectors.schemes.SELECTION_REGISTRY[scheme].load()  → FeatureSet
    → modeling.training.get_scheme + Trainer/MarginTrainer
    → results/prediction/{feat}/{train}-*/{model}/

[Delivery stage — 只读]
  reports/ (Streamlit) · commands/export/ (论文图表/ZIP)
```

### 2.3 入口职责（重构后目标）

| 入口 | 职责 | 非职责 |
|------|------|--------|
| `ais-cli.py` | ML stage 编排（train；可选 feature_eng 真执行） | 不假装跑 ROI 几何；不宣称全自动筛查 |
| `commands/batch_*` | 几何**预标**批处理 | 不训练；输出不是最终 GT |
| `commands/run_pipeline.py` | 参数化 CLI（吃 GT） | 不与 ais-cli 抢名 |
| `commands/plot_*` / `evaluate_*` | 调试与评估 | — |
| `commands/export/` | 交付物 | 不重跑几何 |
| `labeling/` | **GT 权威**（改 ROI+landmarks 并导出） | 不训练 |
| `experiments/`（新建） | sweep/tune/analyze 一次性 | 不进主 import 图 |

### 2.4 旁路（明确不进主路径）

- `moire/`：论文 Module 2 / 可视化；当前 scheme 无 moire 列
- `file_manager/`：数据整理脚本，core 零 import → 迁 `scripts/data_ops/` 或标实验
- `pipeline/predict/`：空壳，本次不实现 serving

---

## 3. 债务与处置

### P0 正确性 / 认知陷阱

| ID | 问题 | 处置 |
|----|------|------|
| D1 | `pipeline/run._run_roi` / `_run_feature_eng` 只返回描述字符串 | **诚实化**：从默认 steps 移除 roi；feature_eng 真调 selector 或同样降级为显式 no-op 文档化。推荐：ais-cli 默认只暴露真实可执行 step（train + 真 feature_eng） |
| D2 | DEPRECATED `pipeline/{core,steps,cache,feature_pipeline}` 被 tests 锁死 | 移入 `legacy/pipeline_v1/`；对应测试标 `legacy` 或改写为测 contracts/run；主 CI 不强制旧 11-step 注册表 |
| D3 | 文档路径虚构 | README + PROJECT.md 按真码重写；PRD/modules 标 Historical |
| D4 | `FEATURE_SCHEME = "morph_region_ci_37d"` 硬编码串台保存路径 | 全程使用 `params.feature_scheme` / scheme 名；禁止模块级常量覆写输出路径 |

### P1 死代码 / 重复

| ID | 问题 | 处置 |
|----|------|------|
| D5 | `mesh/roi/registry.py` ≡ `roi/registry.py`，前者零外部引用 | **删除** `mesh/roi/registry.py` |
| D6 | `mesh/roi_old/` 无引用 | **删除** |
| D7 | `legacy/` 空清单 | 用于接收 pipeline_v1；或保持约定 |
| D8 | `tools/ → commands/` 工作区未提交 | **Phase 0 先提交收口** |
| D9 | `ml/round2,round3,parallel_train,run_ensemble,test_augment` 等实验（均已删除） | 迁 `experiments/ml/` 或删 |
| D10 | commands 内 ~13 个 sweep/tune/analyze/compare/diagnose | 迁 `experiments/commands/` |

### P2 边界

| ID | 问题 | 处置 |
|----|------|------|
| D11 | `pipeline/run.py` 内嵌 `_load_dual_ci*` 巨量筛选 | 下沉 `features/selectors/`，run 只调 load |
| D12 | `ml/schemes.py`（现拆为 features/selectors/schemes.py + modeling/training/schemes.py）554 行巨石 + 与 selectors 环依赖 | 特征方案定义与 load 归 `features/selectors`；`modeling` 只消费 `FeatureSet`；过渡期 thin re-export 后删除 |
| D13 | presets：`modeling.train` 仍 `from modeling.training.schemes import get_training_preset` | 统一 `modeling.training.presets` |
| D14 | contracts 大量未接通类型 | 保留并文档化**在用**类型：`FeatureSet` / `TrainingConfig` / `TrainingResult` / splitter·searcher Protocol；未用 `RawMesh` 等移 `contracts/future.py` 或标注 `# planned` |
| D15 | 超标文件 `landmark_regions.py` 等 | 按现有包规范拆分（独立 PR，可并行） |

---

## 4. 目标包边界

```
ais-cli.py                 # ML 编排唯一入口
config.yaml                # 旧几何 Pipeline 配置（若保留 legacy）；新默认用 pipeline/config.DEFAULT
pipeline/
  run.py                   # 只编排 ML stage
  config.py
  contracts.py             # 精简到真实使用
  # 删除或 legacy: core/steps/cache/feature_pipeline
commands/                  # 生产 CLI only
experiments/               # 扫参与一次性分析
mesh/                      # 几何原子（无第二份 registry）
landmarks/
parameterization/
features/
  extractors/
  selectors/               # 全部 FeatureScheme 定义 + load → FeatureSet
modeling/
  models/
  training/                # schemes + presets + trainer + weights + splitters + searchers
  metrics.py
  train.py                 # 薄封装
labeling/ · reports/ · visualization/ · utils/
legacy/pipeline_v1/        # 旧 per-subject 管线（只读）
docs/                      # 与代码同步
```

**硬规则**

1. 算法不进 `pipeline/` / `commands/`
2. 一个概念一个 registry
3. 公开 step 必须有可观察副作用（文件或 metrics），禁止「打印即成功」
4. 实验代码不进主 import 图
5. 文档路径 = 代码路径
6. 重构必删旧代码，不留 deprecation 转发 stub（项目红线）

---

## 5. 分阶段计划

每期独立 PR、可回滚；锚点：`pytest -q` + 验证集 5 subject + 固定 seed 训练 Macro-F1 波动 ≤ 0.01。

### Phase 0 — 收口 tools→commands（阻塞）

- 提交 `commands/` 全量与 `tools/` 删除
- 更新 pyproject scripts、CLAUDE.md、PROJECT.md、memory 中 `tools.` 引用 → `commands.` / `python -m commands.export`
- **验收**：无 `tools.` 生产引用；export/plot 可启动

### Phase 1 — 纯删除

- 删 `mesh/roi_old/`
- 删 `mesh/roi/registry.py`
- 确认后删或迁 `ml/legacy/`（已删除）、实验 ml 文件（均已删除）→ `experiments/ml/`
- **验收**：`rg 'roi_old|mesh\.roi\.registry' --type py` 为空；pytest 绿

### Phase 2 — 管线诚实化

- 修复 D4 FEATURE_SCHEME 串台
- ais-cli / `pipeline/run`：
  - `train`：保持并净化
  - `feature_eng`：真执行 selector 落盘或从公开 API 移除
  - `roi`：从默认配置移除；文档指向 `commands/batch_process_all.py` / `mesh.roi`
- 旧 `pipeline/{core,steps,cache,feature_pipeline}` → `legacy/pipeline_v1/`
- 测试：主套件不再依赖 ALL_STEPS 11 项；legacy 测试 optional
- **验收**：`ais-cli --list-steps` 只列真实 step；`--step train` 结果路径含实际 feature_scheme 名

### Phase 3 — Scheme 单轨

- 将 `ml/schemes.py`（特征方案部分）中特征 load 逻辑迁入 `features/selectors/`
- 剪断 `features.selectors.scheme_* → ml.schemes._make_data_dict`：`_make_data_dict` 迁 features 或 utils（已内联删除）
- `modeling.train` / `pipeline.run` 只：
  ```python
  from features.selectors import load_feature_set  # → FeatureSet
  from modeling.training.schemes import get_scheme
  from modeling.training.presets import get_training_preset
  ```
- 删除 `pipeline/run` 内 `_load_dual_ci*`
- 过渡：若需兼容旧 `from features.selectors.schemes import SELECTION_REGISTRY`，**最多一个 minor 版本** 后删除（不留永久 stub）
- **验收**：同 scheme+model+seed metrics 对齐 baseline；无 features→modeling 环依赖

### Phase 4 — 大文件拆分（可并行）

优先序：
1. `features/extractors/asymmetry/landmark_regions.py`（965）
2. `mesh/roi/_cut_analysis.py`（641）
3. `ml/cv.py`（已随 cross_verification 删除）

公开 API 经 `__init__.py` 保持稳定。

### Phase 5 — commands 分层 + 文档真源

- `experiments/commands/`：sweep_*, tune_*, analyze_*, compare_*, diagnose_*
- `commands/` 仅保留：batch_*, plot_*, evaluate_*, export/, run_pipeline, streamlit_server, cli_common
- `file_manager/` → `scripts/data_ops/`（或 experiments）
- 重写 README 目录与模块表；PRD 顶部加 Historical banner
- PROJECT / gt-annotation / README：**写明预标→人工 GT→预测**；禁止把自动 landmark 标成生产 stable
- batch_* 文档字符串标明「预标，非 GT」

### Phase 6 — 测试对齐

- 增补（最低）：`pipeline/run` train 路径冒烟；`features.selectors` 一主 scheme load 形状；`ais-cli --list-*`
- 主覆盖锚点：features.selectors / modeling.training / parameterization(GT 输入契约) / utils
- mesh.roi / landmarks：保持现有测；**不**在本阶段加「自动≈GT」通过线（那是算法优化阶段）
- 不强制 labeling/reports/export 全覆盖（可后续）

### 与「下一阶段：ROI/landmark 算法优化」的边界

| 本 spec（外科清理） | 下一阶段（另开 design） |
|--------------------|------------------------|
| 不改 ROI/decloth/landmark 数值与默认阈值 | 改算法、阈值、评分，冲自动质量 |
| 验收 = 测试绿 + 训练 metrics 稳（人工 GT 下游） | 验收 = vs 人工 GT 的误差/通过率/人工分钟 |
| 可动：删死代码、registry、scheme 边界、文档诚实 | 可动：`mesh/roi/*`、`landmarks/*`、预标→平台工作流 |
| labeling 当 GT 权威保留 | 可能减人工步骤，但 GT 格式 ideally 稳定 |

清理时**保护**（方便下阶段）：
- `data/ground_truth/` 布局与 labeling export 契约
- `landmarks.extract` / `mesh.roi` 公开函数名尽量稳定
- 验证集 5 subject 的人工 GT 不被本阶段覆盖写坏

---

## 6. 明确不做

- **不**重写 landmark / ROI / decloth 算法，**不**改默认阈值（→ 下一阶段）
- 不引入新依赖、新 pipeline 框架
- 不把 labeling frontend 并入 Python 包
- 不实现 Moiré Number / alignment / predict serving
- 不做微内核插件系统
- 不把 CSV 特征表强行改为全内存 object graph
- 不做方案 B（contracts 驱动几何接入 ais-cli）——另案评估
- 不把「去掉 labeling 人工步」当作本阶段成功标准

---

## 7. 验收总门禁

| 门禁 | 标准 |
|------|------|
| 测试 | `pytest -q` 主套件全绿 |
| Lint | 变更文件 `ruff check` 零 error |
| 训练回归 | 固定 morph_region_ci_40d 或当前主 scheme + 一模型 + seed，Macro-F1 Δ≤0.01（**人工 GT 特征表**） |
| 文档诚实 | PROJECT/README 写明预标→人工 GT；无「landmark stable=免人工」误导 |
| GT 契约 | `data/ground_truth/` 与 labeling export 路径不被破坏 |
| 引用 | 无死模块 import；文档无死路径 |
| 红线 | 无新依赖；无永久 deprecation stub；单 PR 聚焦 |

说明：原「验证集自动几何 = baseline」**降级**——本阶段不把自动 ROI/landmark 数值当门禁；若跑 `ais-run-validation`，仅作信息对比，失败不阻断清理 PR（除非误改了算法代码）。

---

## 8. 风险

| 风险 | 缓解 |
|------|------|
| 迁 schemes 时 load 路径/列名漂移 | 先固化 baseline metrics.json；逐 scheme 对比 X shape 与 checksum |
| 旧 pipeline 测试大面积红 | 整建制迁 legacy，主 CI 标记 exclude |
| 未提交 commands 与本地结果路径假设 | Phase 0 单独合并，不夹杂逻辑改动 |
| 训练路径隐式依赖 `morph_region_ci_37d` 目录 | D4 修复时做一次全 `results/prediction` 兼容说明（不强制迁移旧结果） |

---

## 9. 建议执行顺序（落地时）

```
Phase 0 (commit hygiene)
  → Phase 1 (delete dead)
  → Phase 2 (honest pipeline)
  → Phase 3 (scheme single track)
  → Phase 4 || Phase 5 (split || docs+experiments)
  → Phase 6 (tests)
```

第一刀（用户确认 spec 后）：Phase 0 → Phase 1 → Phase 2 的 D4 修复。

---

## 10. 参考证据摘要

- 规模：~310 py / ~45k LOC
- `pipeline/run._run_roi`：只 `get(algo)` 返回字符串
- `mesh/roi/registry.py` 与 `roi/registry.py` 逐字相同；仅 `roi.registry` 被 run 引用
- `mesh/roi_old/`：零 import
- 特征生产表：`results/features_extraction/back_v1/`
- 现行训练默认特征名：`morph_region_ci_37d`（常量）与 PROJECT 记载最佳 `morph_region_ci_40d` 不一致——D4 相关
- 旧设计：`docs/superpowers/specs/2026-07-05-engineering-refactoring-design.md`
