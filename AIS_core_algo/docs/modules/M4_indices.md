# M4 — 不对称指标（§3.4）

注意：M4 相关实现文件中包含中文 docstring，建议在使用/调整权重参数时先检查代码 docstring 以确认单位和期望输入格式。


## 目的

通过三个互补指标量化左右躯干不对称。

## Pipeline 步骤

| 步骤 | 组件 | 描述 | 状态 |
|------|------|------|------|
| 4.1 | `compute_asymmetric_index()` | AI = Σ w_i · (λ_M\|κ̄_M^L-κ̄_M^R\| + λ_G\|κ̄_G^L-κ̄_G^R\|)（Eq. 10） | ✅ |
| 4.1a | 全局 AI | 所有区域加权和 | ✅ |
| 4.1b | 每区域 AI | 4 区域各自 AI 值 | ✅ |
| 4.2 | `compute_z_index()` | Z = Σ w_i · \|z̄^L-z̄^R\|（Eq. 11） | ✅ |
| 4.2a | 全局 Z Index | 所有区域加权和 | ✅ |
| 4.2b | 每区域 Z | 4 区域各自 Z 值 | ✅ |
| 4.3 | Moiré Number M | \|左条纹数 − 右条纹数\| | ✅ |

## AI 计算公式（Eq. 10）

```
AI = Σ_i  w_i · ( λ_M · |κ̄_M,i^L − κ̄_M,i^R| + λ_G · |κ̄_G,i^L − κ̄_G,i^R| )
```

- i: 4 解剖区域（肩、胸椎、腰椎、骨盆）
- κ̄_M: 区域平均曲率
- κ̄_G: 区域高斯曲率
- λ_M, λ_G: 曲率项权重

## Z Index 计算公式（Eq. 11）

```
Z = Σ_i  w_i · |z̄_i^L − z̄_i^R|
```

- z̄: 区域平均表面高度

## 输入 / 输出

| | |
|---|---|
| **输入** | 曲率（κ_M, κ_G）、高度场、分割标签、左右标签 |
| **输出** | AI（全局 + 4 区域）、Z Index（全局 + 4 区域）、Moiré Number M |

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `segment_weights` | [0.25, 0.25, 0.25, 0.25] | 4 区域权重 |
| `lambda_m` | 1.0 | 平均曲率项权重 |
| `lambda_g` | 1.0 | 高斯曲率项权重 |

## 文件

- `features/extractors/asymmetry/asymmetric_index.py` — Asymmetric Index
- `features/extractors/asymmetry/z_index.py` — Z Index
- ~~`experiments/moire/moire_number.py`~~ — Moiré Number（论文 M2 实验代码，**已删除**；Moiré 模块整体归档，见 `docs/modules/M2_moire.md`）
