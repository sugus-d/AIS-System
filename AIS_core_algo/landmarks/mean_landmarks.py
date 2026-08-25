"""训练集归一化平均 landmarks（静态先验，供 :mod:`landmarks.complete` 缺失补全使用）。

由一次性脚本从训练集人工 ground_truth 统计生成（生成脚本已归档）。
坐标系：原点 = neck_root_spine_point，脊柱轴对齐 +Y，脊柱长度
（neck_root_spine_point → waist_lower_spine_point）归一化为 1。

运行时把 MEAN_LANDMARKS 相似变换回当前 mesh 物理空间，再映射到最近顶点。
"""

from __future__ import annotations

# 18 点归一化平均坐标（单位：脊柱长度比例）
MEAN_LANDMARKS: dict[str, list[float]] = {
    "neck_root_L": [-0.101579, -0.000730, 0.014661],
    "neck_root_R": [0.088889, 0.001461, 0.055351],
    "shoulder_transition_L": [-0.212512, 0.064563, -0.026614],
    "shoulder_transition_R": [0.204729, 0.064170, 0.060183],
    "scapular_peaks_L": [-0.111567, 0.189130, -0.088786],
    "scapular_peaks_R": [0.138851, 0.191704, -0.041996],
    "axilla_L": [-0.291660, 0.330234, -0.028348],
    "axilla_R": [0.288215, 0.325376, 0.082288],
    "waist_L": [-0.220974, 0.778529, 0.056253],
    "waist_R": [0.191658, 0.788812, 0.130232],
    "waist_lower_L": [-0.256945, 0.999496, 0.047204],
    "waist_lower_R": [0.238476, 0.993265, 0.139870],
    "neck_root_spine_point": [0.000000, 0.000000, 0.000000],
    "scapular_spine_point": [0.008166, 0.190867, -0.037117],
    "axilla_spine_point": [0.010550, 0.332584, -0.041075],
    "waist_spine_point": [-0.003661, 0.788708, 0.017330],
    "waist_lower_spine_point": [0.000075, 0.984001, -0.000067],
    "thoracic_spine_point": [0.004608, 0.562043, -0.014039],
}
