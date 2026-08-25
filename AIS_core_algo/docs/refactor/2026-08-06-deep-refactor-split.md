# 第二轮：长文件拆分 + skipped 激活 + print 统一（2026-08-06）

> Date: 2026-08-06
> Status: ✅ 已完成
> 详细设计/演进记录见本地 `.claude/docs/2026-08-06-deep-refactor-design.md` §9。
> 基线：拆分前 10 个 >400 行文件、15 个 real-mesh 测试 skip、6+ 文件 print 残留。

## 1. 长文件拆分（等价重构，公共 API 不变）

所有 >400 行 .py 拆至 <400 行（`python-standards.md` 硬线）。原则：**原文件名保留**
（`import`/`python -m` 路径不变），函数移到同目录子模块，原文件 re-export 保持 API 完整。

| 原文件 | 原行数 | 拆分后 |
|--------|-------|--------|
| `commands/evaluate.py` | 924 | evaluate 65 + evaluate_cut/back/roi/metrics |
| `commands/plot.py` | 840 | plot 94 + plot_shared/compare/landmarks/parameterization/roughness |
| `features/.../landmark_regions/_features.py` | 505 | _features + _features_candidates |
| `features/.../landmark_regions/_regions.py` | 461 | _regions + _regions_gen |
| `modeling/models.py` | 443 | models + models_base/spec/ensemble |
| `mesh/roi/_bfs_impl.py` | 440 | _bfs_impl + _bfs_seed/holes/roughness |
| `features/selectors/_loaders.py` | 421 | _loaders + _loaders_ci/canonical |
| `commands/export/excel.py` | 416 | excel 39 + excel_tables 394 |
| `modeling/training/feature_selector.py` | 414 | feature_selector + _scoring/_select/_ci |
| `modeling/train.py` | 402 | train + train_helpers |

再拆分产物：`evaluate_roi.py`(407) → + `evaluate_metrics.py`；
`plot_parameterization.py`(419) → + `plot_parameterization_measures.py`。
**终局全库无 >400 行 .py 文件**（tests 例外）。

## 2. skipped 测试激活（15 个 real-mesh 集成测试）

- **根因**：`tests/mesh/test_angle_utils.py` MESH_ID=`"S0069"`、
  `tests/landmarks/test_lateral_profile.py` MESH_ID=`"S0016"` 不在仓库数据目录
  （实际为 `XX-XXXXX` 格式 129 个 subject）→ skipif 永远为真。
- **修复**：MESH_ID → GT 验证集 subject `17-10745`（数据在仓库、断言全过 62 用例）。
- **校准**：`test_real_pipeline_x_range_check` 的 `l_max - r_min < 60` 是 subject 特异
  阈值，选用 GT 验证集 subject 保证断言有效。
- **结果**：15 skip → 0 skip。

## 3. print 全改 logger

- 覆盖：`commands/{batch_prelabel,batch_process_all,streamlit_server,evaluate*,plot*,export/excel*}`
  、`landmarks/gt_validate.py`、`modeling/train*.py`。
- 残留：全库 `print(` 0 处。CLI 表格/进度改 `logger.info/warning/error` 逐行。

## 4. 验收

| 指标 | 前 | 后 |
|------|-----|-----|
| ruff check | clean | ✅ clean |
| pytest | 300 passed / 26 skipped | ✅ 327 passed / 0 skipped |
| golden md5 | 零漂移 | ✅ 等价重构 |
| >400 行文件 | 10 | ✅ 0 |
| print 残留 | 6+ 文件 | ✅ 0 |
