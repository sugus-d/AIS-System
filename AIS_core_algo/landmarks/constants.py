"""所有 landmark 检测常量 — 按模块分 class 命名空间，单一数据源。"""

from typing import NamedTuple

import numpy as np


class AngleCandidate(NamedTuple):
    """颈根候选点，含侧向角度与长轴转角计算结果。"""
    point: np.ndarray      # (2,) 候选点坐标
    angle_deg: float        # 顺时针角度 (deg)
    axis_deg: float          # 长轴转角 (deg)，与垂直方向夹角
    left_pt: np.ndarray     # (2,) 左侧采样点
    right_pt: np.ndarray    # (2,) 右侧采样点
    left_dist: float        # 左侧采样点到候选点距离
    right_dist: float       # 右侧采样点到候选点距离


class AxillaCandidate(NamedTuple):
    """腋窝候选点，含角度与导数度量。"""
    point: np.ndarray      # (2,) 候选点坐标
    cos: float              # 侧向角余弦值
    d2ydx2: float           # 二阶导数值
    is_candidate: bool      # 是否通过 d²<0 过滤


class NeckRoot:
    """颈根检测常量。"""

    TOP_RATIO = 0.3                     # 轮廓裁剪：保留顶部 30%
    CONTOUR_SIGMA = 0.5                  # 轮廓高斯平滑 sigma
    ANGLE_SAMPLE_DISTANCE = 15.0         # 侧向角度采样弧距 (mm)
    LEFT_DERIV_THRESHOLD = 0.3           # 左侧一阶导阈值 (>0.3 保留)
    RIGHT_DERIV_THRESHOLD = -0.3         # 右侧一阶导阈值 (<-0.3 保留)
    LONG_AXIS_ANGLE_MIN = 20.0           # 长轴转角下界 (deg)
    LONG_AXIS_ANGLE_MAX = 70.0           # 长轴转角上界 (deg)
    LONG_AXIS_ARC_LEN = 15.0             # 长轴转角采样弧距 (mm)
    LOWER_BOUND_WIDTH_RATIO = 1.5        # 下界宽度 = 模态宽度 × 1.5
    WAIST_W_UPPER_RATIO = 0.70           # 腰宽 × 0.70 = 颈宽上限
    WAIST_W_LOWER_RATIO = 0.20           # 腰宽 × 0.20 = 颈宽下限
    MIN_NECK_MASK_POINTS = 5             # 直方图 bin 内最少点数
    MIN_HIST_BINS = 5                    # 直方图最少 bin 数
    WIDTH_CANDIDATE_RATIO = 1.05         # 候选行宽度 ≥ 模态宽度 × 1.05
    Y_RANGE_TOLERANCE_RATIO = 0.002      # 候选行 Y 对齐容差比例
    NECK_WIDTH_OK_RATIO = 1.4            # 颈宽 / 模态宽度 < 1.4 判定合理


class Spine:
    """脊柱中线检测常量。"""

    # 分箱与拟合
    N_BINS = 60                          # Y 方向分箱数
    POLY_DEG = 3                         # 样条拟合多项式阶数
    MIN_BIN_POINTS = 5                   # 单 bin 最少点数

    # X 范围裁剪 (25%-75%，排除肩部和侧胸壁)
    X_LO_RATIO = 0.25
    X_HI_RATIO = 0.75

    # 滑动平均
    MIN_SMOOTHING_WINDOW = 3             # 最小滑动窗口
    WINDOW_DIVISOR = 20                  # 窗口 = max(3, len//20)

    # 谷底检测
    VALLEY_STEP_MIN = 2                  # 最小谷底搜索步长
    VALLEY_STEP_DIVISOR = 10             # 步长 = max(2, len//10)

    # Stage 1: MAD 离群值过滤 (基于 X 坐标)
    MAD_THRESHOLD = 2.5                  # |x - median_x| ≤ 2.5×MAD
    MAD_FLOOR = 1.0                      # MAD 下限，防除零

    # Stage 2: 第一次样条拟合
    STAGE1_S_FACTOR = 1.0                # 平滑因子 = len(pts)×1.0

    # Stage 3: 残差过滤
    RESIDUAL_SIGMA_THRESHOLD = 2.0       # 残差 ≤ 2.0×σ
    SIGMA_EPSILON = 1e-6                 # σ 下限，防除零
    CLEAN_POINT_LIMIT = 20               # clean 点低于此数跳过残差过滤

    # Stage 4: 第二次样条拟合 (离群值已剔除)
    STAGE2_S_FACTOR = 0.5                # 平滑因子 = len(pts_clean)×0.5
    Y_RANGE_MIN = 10.0                   # 候选点 Y 范围下限 (mm)

    # 输出
    MIDLINE_N_POINTS = 200               # 平滑中线采样点数

    # derive_spine_points 参数
    LOWER_HALF_PROPORTION = 0.50         # 下半身 Y 比例，估算 mid_x
    LOWER_HALF_MIN_POINTS = 10           # 下半身最少点数阈值
    SEARCH_RADIUS_FRAC = 0.08            # 已弃用，保留签名兼容


class ShoulderTransition:
    """肩臂转点检测常量。"""

    GAUSSIAN_SIGMA = 0.3                 # 轮廓高斯平滑 sigma
    DISTANCE = 10.0                      # 侧向角度采样弧距 (mm)
    ANGLE_ARM_MAX = 175.0                # 有手臂时顺时针角上界 (deg)
    ANGLE_NOARM_MIN = 185.0              # 无手臂时顺时针角下界 (deg)
    MIN_SEGMENT_LEN = 3                  # 候选段最小点数
    GAP_TOLERANCE = 3                    # 段内间隙容忍点数
    OUTER_MARGIN = 50.0                  # arm_boundary_x 推算时外扩距离 (mm)


class ScapularPeak:
    """肩胛峰检测常量。"""

    # Y 带搜索窗口
    Y_LO_RATIO = 0.50                    # 主搜索区间下界（从0.60放宽，配合组合评分覆盖更多解剖变异）
    Y_HI_RATIO = 0.85                    # 主搜索区间上界（从0.80放宽）
    MIN_BAND_SIZE = 10                   # band 点数阈值 (低于则告警)

    # 中线分界
    MIN_SPINE_BAND_POINTS = 3            # spine_midline 在 band 内最少点数
    MID_X_MARGIN_RATIO = 0.015           # 中线缓冲区比例 (防分界重复)

    # 对称性校验
    SYMMETRY_DY_RATIO = 0.03             # 两侧 Y 差超此比例则修正

    # 单侧搜索与回退
    MIN_SIDE_SIZE = 3                    # 一侧点数阈值 (低于则回退)
    Z_PERCENTILE = 85                    # Z 分位阈值，保留 top 15%
    MIN_HIGH_Z_SIZE = 2                  # high-Z 点数阈值 (低于则回退)

    # 候选点选择
    K_BASE = 5                           # K 基础值
    K_FRACTION = 0.10                    # K = max(K_BASE, int(side_size×K_FRACTION))


class Axilla:
    """腋窝检测常量。"""

    # ROI 搜索带边界
    Y_ROI_HI_RATIO = 0.15                # 搜索带上界: nr_y - y_range×0.15
    Y_ROI_LO_RATIO = 0.40                # 搜索带下界: nr_y - y_range×0.40

    # 搜索框
    OUTER_BOUND_RATIO = 0.4              # 外边界 = 腰部外侧半宽 × 0.4
    CLIP_Y_PAD_RATIO = 0.03              # 裁剪 Y 下界额外扩展
    CLIP_X_PAD_RATIO = 2.0               # 裁剪 X 外边界扩展倍数
    BEST_PT_Y_RATIO = 0.30               # 初始 best_pt Y 偏移比例

    # 手臂检测
    ARM_EXTENT_RATIO = 1.0               # 肩/腰 X 范围比 ≥ 1.0 → 有手臂（1.2 过高，倾斜手臂被漏检）
    WAIST_FRAC_LO = 0.40                 # 腰部 X 范围分位下限
    WAIST_FRAC_HI = 0.60                 # 腰部 X 范围分位上限
    SHOULDER_FRAC_LO = 0.60              # 肩部 X 范围分位下限
    SHOULDER_FRAC_HI = 0.90              # 肩部 X 范围分位上限

    # 角度与导数
    ANGLE_SAMPLE_DISTANCE = 10.0         # 侧向角度采样弧距 (mm)
    D2_NEG_THRESHOLD = -0.025            # d²<0 凹性阈值
    DYDX_ARM_MAX = 0.8                   # |dydx| 上界，排除竖直臂段
    MIN_POINTS_DERIV = 11                # SavGol 窗口最小点数
    MIN_POINTS_FALLBACK = 5              # 下边界最小点数

    # 角度过滤
    CW_ANGLE_MIN = 30.0                  # 顺时针角下限 (deg)
    CW_ANGLE_MAX = 170.0                 # 顺时针角上限 (deg)

    # 无手臂侧向选择
    LATERAL_MARGIN = 2.0                 # X 方向 tiebreaker 容差 (mm)
    NOARM_Y_WEIGHT = 0.5                # Y 锚点权重 (Y 每偏离1mm 相当于 X 少0.5mm)
    # 保护性常量：防止 no-arm y_penalty 把候选点拉到极低位置
    NOARM_MAX_PENALTY = 30.0             # 单点最大 y_penalty (mm equivalent in X space)
    NOARM_MAX_DROP = 80.0                # 候选点相对于 y_ref 最大允许下沉 (mm)


# 模块级常量
SPINE_POINT_COUNT = 6  # 脊柱点固定数量（spine_P0..P5）

# 左右成对 landmark 的解剖键名（算法检测 5 类；人工 ground_truth 6 类含 waist_lower）
BILATERAL_KEYS = ["neck_root", "shoulder_transition", "scapular_peaks", "axilla", "waist", "waist_lower"]

# spine 扁平键名（P0-P5，对应 spine_points 数组索引；thoracic 无 bilateral 对应）
FLAT_SPINE_KEYS = [
    "neck_root_spine_point",  # P0 ↔ neck_root
    "scapular_spine_point",  # P1 ↔ scapular_peaks
    "axilla_spine_point",  # P2 ↔ axilla
    "waist_spine_point",  # P3 ↔ waist
    "waist_lower_spine_point",  # P4 ↔ waist_lower
    "thoracic_spine_point",  # P5 无 bilateral 对应（P2/P3 之间）
]
# 18 点扁平键序（6 对 bilateral _L/_R + 6 个 spine 点）— API/预测/导出的单一数据源
FLAT_KEYS = [f"{name}_{side}" for name in BILATERAL_KEYS for side in ("L", "R")] + FLAT_SPINE_KEYS

# ── 历史 P 索引命名 → 语义名 映射（spine_P0..P5 → FLAT_SPINE_KEYS 语义名）──────────
# 参数化/特征历史命名用 P 索引（template/landmark_io/特征列名/区域定义），
# 全链路改为语义名后此映射仅作迁移/兼容用，新增代码一律用 FLAT_SPINE_KEYS。
SPINE_P_SEMANTIC: dict[str, str] = {
    "spine_P0": "neck_root_spine_point",  # P0 ↔ neck_root
    "spine_P1": "scapular_spine_point",  # P1 ↔ scapular_peaks
    "spine_P2": "axilla_spine_point",  # P2 ↔ axilla
    "spine_P3": "waist_spine_point",  # P3 ↔ waist
    "spine_P4": "waist_lower_spine_point",  # P4 ↔ waist_lower
    "spine_P5": "thoracic_spine_point",  # P5 无 bilateral 对应
}

# ── 特征列名段语义缩写（spine_P0_P1_length → spine_neck_scapular_length）──────────
# 段名取自 ci_decompose 展示名（Neck-Scapula 等），供 morphology 特征列名语义化。
SPINE_SEG_SEMANTIC: dict[str, str] = {
    "P0_P1": "neck_scapular",
    "P1_P2": "scapular_axilla",
    "P2_P5": "axilla_thoracic",
    "P5_P3": "thoracic_waist",
    "P3_P4": "waist_waistlower",
    "P0_P3": "neck_waist",
    "P0_P4": "neck_waistlower",
    "P0P1_vs_P3P4": "neck_scapular_vs_waist_waistlower",
}


def spine_p_to_semantic(name: str) -> str:
    """spine_P 索引名 → 语义名（非 P 名原样返回）。"""
    return SPINE_P_SEMANTIC.get(name, name)
