"""Landmark 检测 — 注册表驱动，支持按需选择检测点。

用法:
    from landmarks import detect_all
    result = detect_all(mesh)
"""

from __future__ import annotations

from landmarks.registry import get, list_detectors, register

_DEFAULT = "default"


def detect_all(
    mesh: object,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    **params: object,
) -> dict:
    """检测全部（或指定）landmark 点。

    Args:
        mesh: open3d TriangleMesh。
        include: 未来支持：只检测这些点。
        exclude: 未来支持：排除这些点。
        **params: 传递给底层检测器的参数。

    Returns:
        landmark dict，格式与 extract_landmarks 一致。
    """
    detector = get(_DEFAULT)
    return detector.detect(mesh, **params)


__all__ = ["detect_all", "get", "list_detectors", "register"]
