"""lifter 各子模块共享常量。"""

from pathlib import Path

from ...constants import CACHE_DIR

# curvature 缓存的 mapping 文件路径，与 curvature.py 一致
CURV_IMG_DIR: Path = CACHE_DIR / "curvature_images"

# spine 计算用的 bilateral pair 列表（排除 shoulder_transition、waist_lower）
SPINE_SOURCE_PAIRS: list[str] = ["neck_root", "scapular_peaks", "axilla", "waist"]

MIN_PROJECT_COORDS = 2  # 投影所需最少坐标分量
COORD_DIM_3D = 3  # 三维坐标分量数
PAIR_SIDES = 2  # bilateral 成对 landmark 的 L/R 数量
SNAP_DIST_MM = 100.0  # 吸附到最近顶点的最大距离（mm）
