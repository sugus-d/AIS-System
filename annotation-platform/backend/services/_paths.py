"""路径工具函数——`_get_latest_edited` 等跨服务共享函数。"""

from __future__ import annotations

import json
from pathlib import Path

from ..constants import CACHE_DIR, PLATFORM_CACHE_DIR, PREDICTION_OUTPUTS_DIR

_BILATERAL_LANDMARKS = ["neck_root", "shoulder_transition", "scapular_peaks", "axilla", "waist", "waist_lower"]
_SPINE_KEYS = [
    "neck_root_spine_point",
    "scapular_spine_point",
    "axilla_spine_point",
    "waist_spine_point",
    "waist_lower_spine_point",
    "thoracic_spine_point",
]


def _get_latest_edited(subject_id: str) -> str | None:
    """找到最新的 edited PLY（按时间戳后缀排序，优先 platform cache，回退外部缓存）。"""
    for base in (PLATFORM_CACHE_DIR, CACHE_DIR):
        for subdir in ("extract_roi", "align"):
            d = base / subject_id / subdir
            if not d.exists():
                continue
            edited = sorted(d.glob("roi_edited_*.ply"))
            if edited:
                return str(edited[-1])
    return None


def _find_algorithm_dir(subject_id: str) -> Path | None:
    """找到 AIS 算法输出目录（prediction-outputs/<subject_id>-*/，取最新修改）。

    AIS 管线把每次分析的产物写到 <caseId>-<taskId> 目录，标注平台按 caseId 查找时
    取最近一次（mtime 最新）的目录，即当前报告对应的最新分析结果。
    """
    if not PREDICTION_OUTPUTS_DIR.exists():
        return None
    candidates = sorted(
        (d for d in PREDICTION_OUTPUTS_DIR.iterdir() if d.is_dir() and d.name.startswith(f"{subject_id}-")),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_algorithm_landmarks(subject_id: str) -> dict:
    """从 AIS 算法输出 landmarks.json 读取并转成标注平台嵌套格式。

    算法 flat 格式: {name_L: [x,y,z], name_R: [x,y,z], *_spine_point: [x,y,z]}
    标注平台格式: {"name": [[L],[R]], "spine_points": [6 点]}。
    """
    algo_dir = _find_algorithm_dir(subject_id)
    if not algo_dir:
        return {}
    lm_file = algo_dir / "landmarks.json"
    if not lm_file.exists():
        return {}
    try:
        flat = json.loads(lm_file.read_text())
    except (OSError, ValueError):
        return {}
    out: dict = {}
    for name in _BILATERAL_LANDMARKS:
        out[name] = [flat.get(f"{name}_L"), flat.get(f"{name}_R")]
    out["spine_points"] = [flat.get(k) for k in _SPINE_KEYS]
    return out
