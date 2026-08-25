"""Landmark 检测算法注册表。

当前只注册了一套默认算法（6 个解剖点全部检测）。
未来可替换为其他算法（如 ML 版本）。

注册表三件套由 utils.registry.make_registry 提供（与 mesh/roi/registry.py
同构，共享工厂避免重复）。
"""

from __future__ import annotations

from typing import Protocol

import open3d as o3d

from utils.registry import make_registry


class LandmarkDetector(Protocol):
    """Landmark 检测算法接口。"""
    name: str
    description: str

    def detect(self, mesh: o3d.geometry.TriangleMesh, **params: object) -> dict: ...


register, get, list_detectors = make_registry("landmark 检测器")


# ── 默认检测器 ────────────────────────────────────
from landmarks.extract import extract_landmarks  # noqa: E402 — 延迟导入避免循环依赖


class DefaultLandmarkDetector:
    """默认 6 点 landmark 检测（neck_root, axilla, waist, spine, shoulder_transition, scapular_peak）。"""
    name: str = "default"
    description: str = "双侧解剖学标志点检测（6 点，基于轮廓 + 曲率分析）"

    def detect(self, mesh: o3d.geometry.TriangleMesh, **params: object) -> dict:  # noqa: ARG002 — Protocol 接口契约，保留扩展点
        return extract_landmarks(mesh)


register("default", DefaultLandmarkDetector())
