# M2 — 数字 Moiré（§3.2）

> **⚠️ 模块已归档**（2026-08）：Moiré 源码已从仓库移除（原 `moire/` 目录及 `experiments/moire/` 均无源码，仅 `results/archive/moire/` 保留历史输出图）。本页为历史设计记录，不再对应可运行代码。

注意：本模块的实现文件中包含中文 docstring，可在代码中直接查看函数参数与返回值的详细说明，本文档为高层概览。


## 目的

生成数字 Moiré 条纹图案，计算 Moiré Number 以评估躯干不对称。

## Pipeline 步骤

| 步骤 | 组件 | 描述 | 状态 |
|------|------|------|------|
| 2.1 | `get_moire_img()` | 网格投影到参考平面 → 等高线带 → Moiré 图像 | ✅ |
| 2.2 | `create_rotation_animation()` | 多角度旋转 GIF | ✅ |
| 2.3 | `find_spine_from_mesh()` | 两轮曲率驱动的脊柱中线检测 | ✅ |
| 2.4 | `count_moire_fringes()` | 基于图像的条纹计数（Otsu + 列游程） | ✅ |
| 2.5 | `count_moire_fringes_from_mesh()` | 基于网格的条纹计数（3D 脊柱中线） | ✅ |

## 输入 / 输出

| | |
|---|---|
| **输入** | 背部 ROI 网格 |
| **输出** | Moiré 图像（PNG）、旋转动画（GIF）、Moiré Number M |

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_levels` | 100 | Moiré 等高线层数 |
| `plane_a/b/c/d` | 0,0,1,20 | 参考平面方程系数 |
| `rotation_x/y/z_angle` | 0,0,0 | 初始旋转角度 |
| `frame_num` | 36 | 动画帧数 |
| `duration` | 0.1 | 每帧持续秒数 |

## 文件

> 以下均为已删除路径（模块归档，不再存在）：
- ~~`moire/moire.py`~~ — Moiré 图像生成
- ~~`moire/rotate_moire.py`~~ — 旋转动画
- ~~`moire/find_spine.py`~~ — 脊柱中线检测
- ~~`moire/moire_number.py`~~ — 条纹计数
