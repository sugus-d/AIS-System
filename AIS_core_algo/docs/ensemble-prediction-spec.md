# AIS 预测脚本规格 — Ensemble 组合模型训练与预测（给预测脚本 agent）

> 日期：2026-08-14
> 用途：**供准备预测脚本的 agent 阅读**。项目预测链路有多个历史口径，本文件给出
> 当前正确的方法与 Ensemble 组合模型的完整训练/预测规格，**务必先读再写代码**。
> 状态：`status: implemented`（v0.1.0 = 0.6×CompositeV7 + 0.4×AI-LR 已落地，见第 4/5 节）。
> **2026-08-15 更新**：API 生产预测已升级到 **v1.0.0**（人工 ROI per-class α + Ridge-AI
> 边界 Ensemble，OOF MF1=0.7364 / MAE=4.38°，见 `results/archive/manual_roi_search/breakthrough_report_round2.md`）；
> 本文件的 v0.1.0 ensemble 作为历史 manuscript 复现规格保留。

---

## 1. 项目预测链路总览（当前正确方法）

```
PLY mesh → ROI → 参数化(run_pipeline → mesh_cut + uv) → extract_all(2736 特征)
        → CI 合成(模型包参数) → 模型 → cobb 角 + 分级
```

**单 subject 预测**（`prediction/cli.py` 已实现且正确，可直接用）：
```bash
.venv/bin/python -m prediction.cli predict \
    --ply data/ground_truth/{sid}/original.ply --subject {sid} \
    --clinical data/form/clinical_data.json \
    --landmarks results/ground-truth/{sid}/ground_truth.json
```

关键事实（predict.py 当前行为）：
- **ROI 选择**：人工 ROI 优先（`data/ground_truth/{sid}/roi.ply`），不存在才用算法 `run_roi_pipeline` fallback。
- **参数化**：`run_pipeline`（mesh_path 默认为人工 ROI）→ `mesh_cut.ply + uv_coords.npy`。
- **特征**：`extract_all` = basic(5) + morph(31) + region(2700) = 2736 列，region 列名与训练 `v0.1.0/region_asymmetry.csv` 完全一致（已逐列验证）。
- **CI 合成**：`_add_ci_features` 用模型包保存的 `ci_formula_params`（全 subject 标准化 + Lasso）合成 4 个 CI，**禁止**用 `compute_ci` 单行标准化（会系统性偏差 ~3°）。
- **模型**：`results/modeling/models/v0.1.0/HistGBRT.joblib`（单模型，无加权）。

---

## 2. Ensemble 组合模型（项目最佳结果，manuscript 采用）

**构成**（已从历史数据反推验证，重构误差 = 0.0）：
```
Ensemble_pred = 0.6 × CompositeV7_pred + 0.4 × AI-LR_pred
```

**OOF（折外 CV）泛化指标**（122 subjects，非 in-sample）：
| 指标 | 值 |
|------|-----|
| Macro-F1 | **0.7242** |
| MAE | **4.53°** |
| RMSE | 5.86° |
| r / R² | 0.849 / 0.72 |

**两个分量**：

| 分量 | 构成 | 说明 |
|------|------|------|
| CompositeV7 | HistGBRT + composite 加权 + MarginTrainer + calibrate | 权重: InvFreqWeight(max_ratio=3.0) × MarginBoostWeight × DecayWeight(clinical=10, class_weight=2.0, dist_k=0.1)，均 normalize=False；transform_target=false；hp_n_iter=100。OOF MF1=0.745 |
| AI-LR | AI 特征 → LinearRegression | AI 特征 = `ai_formula.json`（9 个 region 特征线性组合 + intercept=25.43，r=0.856）；LR 在训练集拟合 y~AI |

---

## 3. Ensemble 的训练/重建方法（当前已实现）

新增模块 **`modeling/ensemble.py`**（modeling 库层，落盘走 `result_paths.save_results` 格式）：

| 函数 | 用途 |
|------|------|
| `build_ai_feature(feature_df, formula)` | AI 特征 = Σ coef×region_feature + intercept |
| `fit_ai_linear_oof(ai, y, n_splits=5)` | AI-LR 折外预测（KFold，防泄漏） |
| `build_ensemble_preds(primary, ai, alpha=0.6)` | `α·primary + (1-α)·ai` 加权 |
| `reproduce_manuscript_ensemble()` | **轻量重建**：复用现有 CompositeV7 预测 + ai_formula，秒级验证 0.724/4.53 |
| `train_ensemble(scheme, model, alpha, hp_n_iter, force_retrain)` | **完整闭环**：训练/复用 CompositeV7 → AI-LR OOF → 集成 → 落盘 |

**入口**：
```bash
# 轻量重建（复用现有 CompositeV7 预测，秒级）
.venv/bin/python -m modeling.ensemble

# 完整闭环（--train 训练或复用）
.venv/bin/python -m modeling.ensemble --train --hp-n-iter 5

# 经 modeling.train（默认复用现有结果，不训练）
.venv/bin/python -m modeling.train --scheme v0.1.0 --ensemble

# 经 pipeline（YAML train 步骤加 "ensemble": true）
python ais-cli.py --step train --model HistGBRT --para train:ensemble=true
```

**产物落盘**：`results/modeling/prediction/v0.1.0/ensemble-ai60-lroof/Ensemble/`
（predictions.csv + metrics.json + config.json，与训练结果同格式）

---

## 4. Ensemble 的预测方法

### 4.1 批量预测（已可用）
直接用已生成的 `ensemble-ai60-lroof/Ensemble/predictions.csv`（122 subjects，逐位复现 manuscript），或重跑 `reproduce_manuscript_ensemble()`。

### 4.2 单 subject 预测（已实现）

**正确逻辑**：
```
1. 特征：extract_all(mesh_cut, sid, clinical, landmarks, uv) → 2736 列
2. C7_pred：CompositeV7 模型对单 subject 预测
   └─ 前置缺口：CompositeV7 需先保存为 joblib（当前 save_model 只存无加权最终模型）
3. AI 特征：ai_formula 对单行 region 特征求值 → AI
4. AI_pred：LR(AI)，LR 系数来自训练集全量拟合（intercept + coef×AI）
5. ens = 0.6×C7_pred + 0.4×AI_pred
```

**~~前置缺口~~（已解决）**：`CompositeV7` joblib 包已由 `modeling.ensemble.save_composite_model` 生成
（`results/modeling/models/v1.0.0/CompositeV7.joblib`，含 estimator + scaler + feature_names +
calibration_bias + CI 参数）。2026-08-15 起生产预测用自包含 `prediction/models/v1.0.0.joblib`；2026-08-16 起结构统一为**展平顶层**（composite 分量并入顶层，与 v0.1.0 对齐，见 `commands/export/migrate_model_package_flat.py`）。
注意：`save_composite_model` 必须应用 best_params（按 `get_param_space()` 过滤），否则部署模型与 OOF 差 ~5.9°。

### 4.3 v1.0.0 模型包训练产物清单（2026-08-15）

`prediction/models/v1.0.0.joblib`（原 `boundary_ensemble_ridge.joblib`）是自包含包；训练阶段（`modeling.ensemble_boundary.save_boundary_model`）
在**同一模型目录**落盘以下可读产物（与 joblib 同一次拟合，不重复计算），供人工核查、论文制表与
`commands/export/v1_0_0_export.py` 批量复用：

| 产物（`results/modeling/models/v1.0.0/`） | 内容 | 消费方 |
|------|------|--------|
| `prediction/models/v1.0.0.joblib` | 展平顶层：模型/scaler/CI 参数 + ai8/ridge 公式 + 边界分类器 + per-class α/β | `prediction/predict.py` 单 subject 预测 |
| `asymmetry_formulas.json` | 5 不对称指数公式 + scaler(mean/scale) + AI OLS 权重 | 人工核查、export indices 表 |
| `ci_formula_params.json` | 4 CI 公式 + ci10/ci20 目标参数 | 人工核查、export CI 反解 |
| `ai_formulas.json` | Lasso-8（8 特征）/ Ridge-267（267 特征）cols+coefs + LR | 人工核查 |
| `ensemble_config.json` | α/β/钳制阈值/边界分类器系数 | 人工核查、复现 |
| `feature_importance.csv/png` | composite 全局特征重要性（permutation, Top15） | 论文图、人工核查 |
| `example_waterfall.png` | 训练集代表 subject 单 case SHAP 瀑布图 | 复现示例 |

> ⚠️ 训练落盘约定：**新模型包必须内嵌全部生成逻辑**（CI 合成 + 不对称指数 + 特征重要性 +
> 瀑布图所需模型），并同步落盘可读 JSON/图表——禁止在 API/export 运行时临时重拟合公式。

### 4.4 导出切换（v0.1.0 ↔ v1.0.0，非替代）

`commands/export` **按 `--scheme` 切换整套输出**（特征重要性 CI 反解 + 论文表 + 论文图），
两套目录独立、互不覆盖：

```bash
# v0.1.0（默认，行为不变）：算法 ROI → results/modeling/feature_importance + results/eval/tables + results/export/分析图片
uv run python -m commands.export --scheme v0.1.0
# v1.0.0（人工 ROI 方案，别名 manual）→ results/export/v1.0.0/
uv run python -m commands.export --scheme v1.0.0
```

| 输出 | v0.1.0 | v1.0.0 |
|------|---------|--------|
| 特征重要性（CI 反解 3 CSV + 三图） | `results/modeling/feature_importance/` | `results/export/v1.0.0/feature_importance/` |
| 论文表 1/3/4/5/6（CSV + raw + xlsx） | `results/eval/tables/` | `results/export/v1.0.0/tables/` |
| 论文关键图（散点/Bland-Altman/混淆矩阵） | `results/export/分析图片/` | `results/export/v1.0.0/figures/` |
| indices 表 + 单 case 瀑布图 | —（v0.1.0 走 charts_waterfall） | `results/export/v1.0.0/` |

核心：`analyze.py` 抽 `_run_analysis`（permutation importance → CI 反解 → 聚合 3 CSV），
v0.1.0 与 v1.0.0 **同一函数同一口径**，仅数据源/模型/CI 公式不同：
- v0.1.0：`_load_back()`（现场训练 + `results_compressed.csv` 公式 + 现场拟合 ci10/ci20）
- v1.0.0：`_load_manual()`（`SELECTION_REGISTRY` 30D + 模型包 composite 模型/scaler +
  `_load_manual_ci_formulas()` 从模型包 `ci_formula_params.json` 转换）

`tables.py`/`raw_tables.py`/`excel_tables.py`/`figures.py` 均参数化（pred_csv + region_csv +
out_dir，默认 None → v0.1.0 路径），v1.0.0 分支传入
`MANUAL_PRED_PATH`/`MANUAL_REGION_CSV`/`MANUAL_TABLES_DIR`/`MANUAL_FIGURES_DIR`（utils/paths.py）。

manual 的 CI 特征重要性**完全反解到 region 特征**（decomposed CSV 无 CI 特征名残留），
与 v0.1.0 同构（by_group + by_horizontal_band 聚合）。

---

## 5. ⚠️ 常见错误/过时方法（绝对不要用）

| # | 错误/过时方法 | 后果 | 正确替代 |
|---|--------------|------|---------|
| 1 | 用 `run_roi_pipeline` 实时提取 ROI 作为预测输入 | ROI 算法已退化，cobb 误差 +3.4°（MAE 4.23°） | 用人工 ROI `data/ground_truth/{sid}/roi.ply`；不存在才算法 fallback |
| 2 | 把 in-sample MAE（如 0.98°）当作模型泛化能力 | 严重虚低（全量重训模型对训练数据预测），误导评估 | 泛化指标必须用 **OOF/CV**（ensemble 4.53°、单模型 4.99°） |
| 3 | 用 `compute_ci` 单行标准化合成 CI 特征 | CI 值系统性偏差，cobb 误差 +3° | 用模型包 `ci_formula_params`（`predict._add_ci_features`） |
| 4 | 找旧 `ml/run_ensemble.py` 或旧 `ensemble_composite_v7_ai60` 生成脚本 | 脚本已被重构删除，不存在 | 用 `modeling/ensemble.py` 重建 |
| 5 | 直接用 `results/roi/{sid}/roi.ply`（算法 ROI）喂给"人工 ROI 重训"的模型 | 训练/预测 ROI 口径不一致 | 确认训练特征 CSV 的 ROI 来源：v0.1.0 = 算法 ROI；新人工 ROI 训练 = `data/ground_truth` |
| 6 | 假设 `features_2700d.csv` 与 `v0.1.0/region_asymmetry.csv` 不同 | 两者逐列一致（2702 列全交集），可互换 | 以 `v0.1.0/` 为准 |
| 7 | 用旧 `results/prediction/...` 路径（重构前） | 路径已迁移到 `results/modeling/prediction/` | 统一用新路径 |

---

## 6. 关键文件清单

| 文件 | 作用 |
|------|------|
| `prediction/predict.py` | 单 subject 预测（人工 ROI 优先 + CI 公式 + 5 指数 + 瀑布图，正确） |
| `modeling/ensemble.py` | **新增**：ensemble 构建/重建/闭环模块 |
| `modeling/ensemble_boundary.py` | v1.0.0：per-class α + Ridge-AI 边界 Ensemble 训练 + 产物落盘 |
| `modeling/training/save_model.py` | 模型包保存（含 ci_formula_params / asymmetry 公式） |
| `modeling/training/schemes.py` | `composite_v7` 训练配置（TRAINING_SCHEMES） |
| `modeling/training/result_paths.py` | 结果落盘统一格式（save_results） |
| `features/extractors/assemble.py` | `extract_all`：2736 特征装配 |
| `parameterization/pipeline.py` | 参数化：ROI → mesh_cut + uv |
| `commands/export/v1_0_0_export.py` | v1.0.0 批量导出（indices + 特征重要性 + 瀑布图） |
| `results/modeling/prediction/v0.1.0/ensemble-ai60-lroof/` | v0.1.0 ensemble 产物（OOF 0.724/4.53） |
| `results/modeling/models/v1.0.0/` | 训练阶段 JSON/图表产物（v1.0.0 模型包已移至 `prediction/models/v1.0.0.joblib`） |
| `results/formulas/archive/ai_formula.json` | AI 特征公式（9 特征，intercept 25.43） |

## 7. 关键命令速查

```bash
# 单 subject 预测（现状可用）
.venv/bin/python -m prediction.cli predict --ply data/ground_truth/{sid}/original.ply \
    --subject {sid} --clinical data/form/clinical_data.json \
    --landmarks results/ground-truth/{sid}/ground_truth.json

# ensemble 轻量重建（秒级验证）
.venv/bin/python -m modeling.ensemble

# ensemble 完整闭环
.venv/bin/python -m modeling.train --scheme v0.1.0 --ensemble

# lint
.venv/bin/ruff check modeling/ensemble.py prediction/cli.py
```
