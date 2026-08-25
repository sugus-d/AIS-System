"""Landmark 检测算法注册表。

当前只注册了一套默认算法（6 个解剖点全部检测）。
未来可替换为其他算法（如 ML 版本）。

# 与 mesh/roi/registry.py 为平行注册表模式（Protocol + 字典 + register/get/list）；若出现第 3 个同构注册表再抽取共享基类。
"""

from __future__ import annotations

from typing import Protocol

import open3d as o3d


class LandmarkDetector(Protocol):
    """Landmark 检测算法接口。"""
    name: str
    description: str

    def detect(self, mesh: o3d.geometry.TriangleMesh, **params: object) -> dict: ...


_DETECTORS: dict[str, LandmarkDetector] = {}


def register(detector: LandmarkDetector) -> None:
    _DETECTORS[detector.name] = detector


def get(name: str) -> LandmarkDetector:
    if name not in _DETECTORS:
        raise KeyError(f"未知 landmark 检测器: {name}，可选: {list(_DETECTORS.keys())}")
    return _DETECTORS[name]


def list_detectors() -> list[str]:
    return list(_DETECTORS.keys())


# ── 默认检测器 ────────────────────────────────────
from landmarks.extract import extract_landmarks  # noqa: E402 — 延迟导入避免循环依赖


class DefaultLandmarkDetector:
    """默认 6 点 landmark 检测（neck_root, axilla, waist, spine, shoulder_transition, scapular_peak）。"""
    name: str = "default"
    description: str = "双侧解剖学标志点检测（6 点，基于轮廓 + 曲率分析）"

    def detect(self, mesh: o3d.geometry.TriangleMesh, **params: object) -> dict:
        is_debug = params.pop("is_debug", True)
        return extract_landmarks(mesh, is_debug=is_debug)


register(DefaultLandmarkDetector())
