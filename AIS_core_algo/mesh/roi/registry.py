"""ROI 算法注册表 — 各算法实现统一接口，通过名称选择。

注册表三件套由 utils.registry.make_registry 提供（与 landmarks/registry.py
同构，共享工厂避免重复）。
"""

from __future__ import annotations

from typing import Protocol

import open3d as o3d

from utils.registry import make_registry


class ROIAlgorithm(Protocol):
    """ROI 提取算法接口。"""

    name: str
    description: str

    def run(self, mesh: o3d.geometry.TriangleMesh, **params: object) -> o3d.geometry.TriangleMesh: ...


register, get, list_algorithms = make_registry("ROI 算法")


# ── 内置算法注册 ─────────────────────────────────────
from mesh.roi._pants_cut import remove_pants  # noqa: E402
from mesh.roi.extract import extract_by_xy_hull  # noqa: E402
from mesh.roi_extract import extract_back_roi  # noqa: E402


class BFSExtractor:
    """BFS 背部 ROI 提取（默认）。"""

    name = "bfs"
    description = "BFS 种子生长 + 法线角切除"

    def run(self, mesh: o3d.geometry.TriangleMesh, **params: object) -> o3d.geometry.TriangleMesh:
        return extract_back_roi(mesh, **params)


class PantsCutExtractor:
    """裤子切除（BFS 后处理）。"""

    name = "pants_cut"
    description = "裤子/裙子区域切除"

    def run(self, mesh: o3d.geometry.TriangleMesh, **params: object) -> o3d.geometry.TriangleMesh:
        return remove_pants(mesh, **params)


class XYHullExtractor:
    """XY 凸包裁剪。"""

    name = "xy_hull"
    description = "XY 平面凸包裁剪"

    def run(self, mesh: o3d.geometry.TriangleMesh, **params: object) -> o3d.geometry.TriangleMesh:
        return extract_by_xy_hull(mesh, **params)


# 注册所有算法
for algo in [BFSExtractor(), PantsCutExtractor(), XYHullExtractor()]:
    register(algo.name, algo)
