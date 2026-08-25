# M5 — ML 严重程度预测（§3.5）

注意：ML 与特征工程模块中包含部分中文 docstring（特征生成函数应在源码中描述输入列与缺失值处理策略），在修改特征或训练参数前建议阅读源码 docstring 与 notebooks 获取实验细节。


## 目的

将提取的特征融合为单一的脊柱侧弯严重程度预测。

## Pipeline 步骤

### 特征工程

| 步骤 | 组件 | 描述 | 状态 |
|------|------|------|------|
| 5.1 | `build_feature_row()` | 组装 ML 特征向量 → DataFrame | 🔲 |
| 5.1a | AI | 全局 + 4 区域值 | 🔲 |
| 5.1b | Z Index | 全局 + 4 区域值 | 🔲 |
| 5.1c | Moiré Number M | 条纹计数差 | 🔲 |
| 5.1d | 形态学 | 肩高差、躯干长度等 | 🔲 |
| 5.1e | 临床元数据 | 年龄、性别、BMI | 🔲 |
| 5.2 | 特征工程 | 归一化、交叉项、特征选择 | 🔲 |

### 模型训练

| 步骤 | 组件 | 描述 | 状态 |
|------|------|------|------|
| 5.3 | Notebook 实验 | LGBM.ipynb, draft_asym.ipynb | ✅ |
| 5.4 | GRDF / LightGBM 训练 | 正式 sklearn Pipeline | 🔲 |
| 5.4a | 训练测试划分 | 数据集划分 | 🔲 |
| 5.4b | 超参数调优 | 网格/随机搜索 | 🔲 |
| 5.4c | 模型评估 | RMSE, MAE, R² | 🔲 |
| 5.4d | 模型序列化 | 保存 `.pkl` | 🔲 |
| 5.5 | 模型推理 | 从特征向量预测严重程度 | 🔲 |

## 特征向量

```
[AI_global, AI_shoulder, AI_thoracic, AI_lumbar, AI_pelvic,
 Z_global, Z_shoulder, Z_thoracic, Z_lumbar, Z_pelvic,
 MoireNumber,
 shoulder_height_diff, trunk_length, ...,
 age, sex, BMI]
```

## 输入 / 输出

| | |
|---|---|
| **输入** | 特征 DataFrame |
| **输出** | 预测严重程度评分 |

## 配置参数

| 参数 | 说明 |
|------|------|
| `model_type` | lightgbm / grdf |
| `hyperparameters` | 模型超参数 |
| `feature_columns` | 使用的特征列 |
| `test_split` | 测试集比例 |

## 文件

- `modeling/` — 当前训练管线（`train.py` 入口；`models.py` / `metrics.py` / `contracts.py` / `training/` 子包）
- `modeling/training/` — 训练组件（data_splitters / feature_selector / hp_searchers / presets / schemes / trainer / weights）
- `features/selectors/` — 特征方案注册表（`schemes.py` SELECTION_REGISTRY，默认 `v0.1.0`）
