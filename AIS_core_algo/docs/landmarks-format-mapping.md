# landmarks 格式契约（扁平 18 键）

全链路（预测 API、参数化、特征、标注平台磁盘、训练数据）统一使用**扁平 18 键**
landmarks 格式。旧嵌套格式（`{name: {L,R}}` + `spine_points` 数组）已**彻底移除**，
不再有读取时自动转换。

## 完整 18 键格式

单源定义：`landmarks/constants.py::FLAT_KEYS`（6 对 bilateral `_L/_R` + 6 个语义 spine 键）。

```json
{
  "neck_root_L": [x, y, z],
  "neck_root_R": [x, y, z],
  "shoulder_transition_L": [x, y, z],
  "shoulder_transition_R": [x, y, z],
  "scapular_peaks_L": [x, y, z],
  "scapular_peaks_R": [x, y, z],
  "axilla_L": [x, y, z],
  "axilla_R": [x, y, z],
  "waist_L": [x, y, z],
  "waist_R": [x, y, z],
  "waist_lower_L": [x, y, z],
  "waist_lower_R": [x, y, z],
  "neck_root_spine_point": [x, y, z],
  "scapular_spine_point": [x, y, z],
  "axilla_spine_point": [x, y, z],
  "waist_spine_point": [x, y, z],
  "waist_lower_spine_point": [x, y, z],
  "thoracic_spine_point": [x, y, z]
}
```

## spine 语义键解剖对应

| 扁平键 | 解剖对应 |
|--------|---------|
| `neck_root_spine_point` | 颈根水平（原 P0） |
| `scapular_spine_point` | 肩胛水平（原 P1） |
| `axilla_spine_point` | 腋窝水平（原 P2） |
| `waist_spine_point` | 腰水平（原 P3） |
| `waist_lower_spine_point` | 腰下水平（原 P4） |
| `thoracic_spine_point` | 胸腰段（原 P5，无 bilateral 对应） |

> **命名依据**：spine 点与 bilateral 点的解剖对应由 UV 参数化模板
> `parameterization/template.py` 的 V 坐标定义（neck=+2、scapular=+1、
> axilla=0、waist=-3、waist lower=-4），并经训练集 125 个完整人工标注
> 的 3D 坐标验证（对应点 y 值同水平）。
>
> **历史命名**：spine 点曾用 `spine_P0..P5` 索引命名（参数化/特征历史遗留）。
> 全链路已改语义名，映射表见 `landmarks/constants.py::SPINE_P_SEMANTIC`。

## 缺失补全机制

算法自动检测缺 `waist_lower`、spine 仅 4/6 点。补全策略（`prediction/predict.py::_complete_landmarks_flat`）：

1. 预置**训练集归一化平均 landmarks**（`prediction/_mean_landmarks.py`，由
   `commands/export/compute_mean_landmarks.py` 从 125 个完整人工标注统计）
2. 运行时用**已检测到的点对**（当前 mesh ↔ 平均模板）拟合 Umeyama 相似变换
   （缩放 + 旋转 + 平移，≥3 对非共线）
3. 缺失键取平均坐标 → 相似变换回当前 mesh 物理空间 → 映射到**最近网格顶点**

## 格式边界（全链路一致）

- **磁盘**（`results/ground-truth/{sid}/ground_truth.json`）：扁平 18 键 +
  `_features` 元数据（手臂状态/不对称标志/标注状态）
- **API 输入**（`/api/predict` 状态 2 landmarks 字段）：扁平 18 键，缺失 422 拒绝
- **算法输出**（`extract_landmarks`）：扁平 18 键（bilateral 6 对 + 4 个 spine
  语义键，`waist_lower_spine_point`/`thoracic_spine_point` 由平均模板补全）
- **标注平台**：磁盘读写扁平；API/前端仍为双边 list 契约（`{name: [[L],[R]]}`，
  由 backend `services/lifter/service.py` 转换）
- **参数化**（`parameterization/`）：消费扁平 dict，`TEMPLATE_LANDMARKS` 键为语义名
