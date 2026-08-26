"""共享常量 — 所有后端模块从这里引用，避免重复定义和 sys.path 插入。

数据路径通过 AIS_DATA_ROOT / AIS_RESULTS_ROOT 环境变量配置（解耦核心仓库），
默认指向标注平台自带的 data/ 与 results/。运行时可用 env 指向共享数据目录。
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(key: str, default: Path) -> Path:
    """从环境变量读路径，未设置时用默认值。"""
    raw = os.environ.get(key)
    return Path(raw) if raw else default


# 标注平台自身根：独立成仓库后即仓库根；解耦前为核心仓库的子目录
PLATFORM_DIR: Path = Path(__file__).resolve().parent.parent
DATA_ROOT: Path = _env_path("AIS_DATA_ROOT", PLATFORM_DIR / "data")
RESULTS_ROOT: Path = _env_path("AIS_RESULTS_ROOT", PLATFORM_DIR / "results")

BILATERAL_LANDMARKS: list[str] = [
    "neck_root",
    "shoulder_transition",
    "scapular_peaks",
    "axilla",
    "waist",
    "waist_lower",
]

LANDMARK_NAMES_ZH: dict[str, str] = {
    "neck_root": "颈根",
    "shoulder_transition": "肩臂转点",
    "scapular_peaks": "肩胛峰",
    "axilla": "腋窝",
    "waist": "腰部",
    "waist_lower": "腰下缘",
    "spine_points": "脊柱",
}

LANDMARK_COLORS: dict[str, str] = {
    "neck_root": "#00FFFF",
    "shoulder_transition": "#FF4444",
    "scapular_peaks": "#44FF44",
    "axilla": "#FF44FF",
    "waist": "#FFFF44",
    "waist_lower": "#FF8C00",
    "spine_points": "#FFFFFF",
}

SPINE_POINT_COUNT: int = 6

LANDMARK_SIDES: list[str] = ["L", "R"]
LANDMARK_AXES: list[str] = ["x", "y", "z"]

# Y 层级顺序（从高到低：Y 值递减，inferior- 到 superior+）
LANDMARK_Y_ORDER: list[str] = [
    "neck_root",
    "shoulder_transition",
    "scapular_peaks",
    "axilla",
    "waist",
    "waist_lower",
]

PLATFORM_CACHE_DIR: Path = RESULTS_ROOT / "labeling" / "cache"

MESH_DIR: Path = DATA_ROOT / "mesh"
CACHE_DIR: Path = RESULTS_ROOT / "cache"
GT_DIR: Path = RESULTS_ROOT / "ground-truth"
CONFIG_PATH: Path = PLATFORM_DIR / "config.yaml"

ROI_DIR: Path = RESULTS_ROOT / "roi"
MESH_PROCESSED_DIR: Path = RESULTS_ROOT / "meshes_processed"
PREDICTION_OUTPUTS_DIR: Path = RESULTS_ROOT / "prediction-outputs"
