# 配置参考 — config.yaml

注意：仓库若干核心代码文件已补充中文 docstring 与关键算法注释，建议在阅读本配置参考时优先查看对应 module 的 docstring 获取更详细的参数说明。

`config.yaml` 是 AIS Pipeline 的统一配置文件。所有模块参数集中管理，支持单受试者和批量处理。

---

## 顶层结构

```yaml
pipeline:     # Pipeline 运行配置
subjects:     # 受试者列表
preprocess:   # M1 预处理参数
moire:        # M2 Moiré 参数
curvature:    # M3 曲率参数
geometry:     # M3 几何分析参数
indices:      # M4 指标参数
features:     # M5 特征列
ml:           # M5 ML 参数
```

---

## pipeline

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cache_dir` | string | `results/cache` | 缓存目录，每受试者中间结果存放于此 |
| `steps` | list | — | 按顺序执行的步骤名称列表 |

### 可用步骤

| 步骤名 | 模块 | 说明 |
|--------|------|------|
| `load_mesh` | M1.1 | 加载 PLY 网格 |
| `extract_roi` | M1.3 | ROI 提取（区域生长） |
| `denoise` | M1.4 | 去噪 |
| `smooth` | M1.5 | 平滑 |
| `align` | M1.7 | PCA 对齐 |
| `moire` | M2.1 | 生成 Moiré 图像 |
| `curvature` | M3.1-2 | 曲率计算与可视化 |
| `landmarks` | M3.3 | 解剖标志点提取 |
| `parameterization` | M3.6 | 调和映射参数化 |
| `height` | M3.4 | 表面高度场 |
| `segmentation` | M3.5 | 解剖分割 |
| `indices` | M4 | 不对称指标计算 |
| `features` | M5.1 | 特征向量组装 |

---

## subjects

受试者列表，每个受试者包含：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 受试者唯一标识 |
| `mesh_path` | string | 是 | PLY 网格文件路径 |
| `clinical` | dict | 否 | 临床元数据 |

### clinical

| 字段 | 类型 | 说明 |
|------|------|------|
| `age` | int | 年龄 |
| `sex` | string | 性别（M/F） |
| `bmi` | float | BMI 值 |

---

## preprocess（M1）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `envelope_grid_resolution_mm` | float | 1.5 | 包络网格 XY 分辨率（mm），越小细节越多 |
| `envelope_smooth_sigma_px` | float | 1.25 | 高斯平滑 sigma（像素），越大越平滑 |
| `envelope_valid_weight_threshold` | float | 0.16 | 有效高斯支撑权重阈值 |
| `envelope_max_triangle_z_span_mm` | float | 35.0 | 重建三角面的最大 Z 跨度 |
| `envelope_max_smooth_delta_mm` | float | 6.0 | 平滑允许的最大 Z 偏移 |
| `denoise_method` | string | `statistical` | 去噪方法：`statistical` 或 `radius` |
| `denoise_nb_neighbors` | int | 20 | 去噪邻居点数 |
| `smooth_method` | string | `laplacian` | 平滑方法：`laplacian` / `taubin` / `simple` |
| `smooth_iterations` | int | 10 | 平滑迭代次数 |

---

## moire（M2）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_levels` | int | 100 | Moiré 等高线层数，越大条纹越细 |
| `plane_a` | float | 0 | 参考平面方程系数 a（ax + by + cz + d = 0） |
| `plane_b` | float | 0 | 参考平面方程系数 b |
| `plane_c` | float | 1 | 参考平面方程系数 c |
| `plane_d` | float | 20 | 参考平面方程系数 d（控制条纹间距） |

---

## curvature（M3）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `clip_range_mean` | list[float] | [-0.015, 0.025] | 平均曲率颜色映射裁剪范围 |
| `clip_range_gauss` | list[float] | [-0.00025, 0.00025] | 高斯曲率颜色映射裁剪范围 |

---

## geometry（M3）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seg_shoulder_thresh` | float | 0.78 | 肩部 Y 阈值（标准化高度比例） |
| `seg_lumbar_thresh` | float | 0.42 | 腰椎 Y 阈值 |
| `seg_pelvic_thresh` | float | 0.18 | 骨盆 Y 阈值 |

---

## indices（M4）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `segment_weights` | list[float] | [0.25, 0.25, 0.25, 0.25] | 4 解剖区域权重（肩/胸椎/腰椎/骨盆） |
| `lambda_m` | float | 1.0 | AI 公式中平均曲率项权重 |
| `lambda_g` | float | 1.0 | AI 公式中高斯曲率项权重 |

---

## features（M5）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `feature_columns` | list | — | ML 模型使用的特征列名列表 |

---

## ml（M5）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_type` | string | `lightgbm` | 模型类型（当前仅支持 lightgbm） |
| `test_split` | float | 0.2 | 测试集比例（0-1） |

---

## 示例

> 旧 `run_pipeline.py` 入口已移除（2026-08 重构），当前入口见下。所有命令**从项目根运行**（`python -m` 将 cwd 加入 sys.path），或设置 `PYTHONPATH=<项目根>`。

### 单受试者参数化

```bash
uv run python -m commands.plot --domain parameterization S0069
```

### 指定步骤运行（全管线 CLI）

```bash
python ais-cli.py --step feature_eng --step train
python ais-cli.py --step roi --roi-algo bfs --step train --model HistGBRT
```

### 从配置批量运行

```bash
python ais-cli.py --config config.yaml
```
