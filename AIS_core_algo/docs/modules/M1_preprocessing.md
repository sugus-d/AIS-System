# M1 — 网格预处理（§3.1）

注意：相关实现文件已补充中文 docstring 与关键算法注释；阅读配置/参数或边界条件时，建议同时查看对应源码中的 docstring 以获取更准确信息。


## 目的

将原始 3D 扫描转换为干净、对齐的背部 ROI 网格。

## Pipeline 步骤

| 步骤 | 组件 | 描述 | 状态 |
|------|------|------|------|
| 1.1 | `load_mesh()` | 加载 PLY 文件（点云自动 Poisson 重建） | ✅ |
| 1.2 | `preprocess_back_scan_mesh()` | 基于 XY 网格的单层包络平滑 | ✅ |
| 1.3 | `extract_back_roi()` | 区域生长 → 背部 ROI → 裤裆裁剪 → 孔洞填充 | ✅ |
| 1.4 | `denoise_mesh()` | 统计/半径离群点去除 | ✅ |
| 1.5 | `smooth_mesh()` | Laplacian / Taubin 平滑 | ✅ |
| 1.6 | `fill_mesh_holes()` | PyVista 孔洞填充 + 过大三角面清理 | ✅ |
| 1.7 | `align_mesh()` | PCA 标准方向校正 + ICP 精配准 | ✅ |

## 输入 / 输出

| | |
|---|---|
| **输入** | 原始 PLY 网格（点云或三角网格） |
| **输出** | 干净、对齐的背部 ROI 网格（PLY） |

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `envelope_grid_resolution_mm` | 1.5 | 包络网格分辨率（mm） |
| `envelope_smooth_sigma_px` | 1.25 | 高斯平滑 sigma（像素） |
| `envelope_valid_weight_threshold` | 0.16 | 有效高斯支撑权重阈值 |
| `envelope_max_triangle_z_span_mm` | 35.0 | 重建三角面最大 Z 跨度 |
| `envelope_max_smooth_delta_mm` | 6.0 | 平滑最大 Z 调整量 |
| `denoise_method` | statistical | 去噪方法：statistical / radius |
| `denoise_nb_neighbors` | 20 | 去噪邻居数 |
| `smooth_method` | laplacian | 平滑方法 |
| `smooth_iterations` | 10 | 平滑迭代次数 |

## 文件

- `mesh/preprocess/preprocess.py::preprocess_back_scan_mesh` — 预处理入口
- `mesh/preprocess/clean.py` — 去噪 / 平滑 / 孔洞填充（denoise_mesh / smooth_mesh / fill_mesh_holes）
- `mesh/preprocess/envelope.py` — 包络重建（rebuild_single_layer_envelope）
- `mesh/preprocess/alignment.py` — 对齐（align_mesh / apply_rotation）
- `mesh/roi/` — 区域生长 ROI 提取（BFS；生产入口 `commands/batch_process_all.py::run_roi_pipeline`）
- `utils/mesh.py` — 网格加载工具
