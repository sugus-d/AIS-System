"""面板内部数据工具 — 含文件读取（load_cached_numpy），是渲染层唯一 I/O 例外（面板预加载数据用，避免在渲染热路径做磁盘 I/O）。"""

from __future__ import annotations

import os

import numpy as np


def is_missing_value(value: object) -> bool:
    """判断值是否缺失（None、空 dict、空 array）。

    用于校验 pipeline 缓存返回的数据完整性。
    """
    if value is None:
        return True
    if isinstance(value, dict):
        return len(value) == 0
    if isinstance(value, np.ndarray):
        return len(value) == 0
    return False


def unique_missing(items: list[str]) -> list[str]:
    """去重缺失项列表，保持顺序。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def load_cached_numpy(cache_dir: str, subject: str, step_name: str, file_name: str) -> np.ndarray | None:
    """从 pipeline 缓存目录加载 numpy 数组。"""
    path = os.path.join(cache_dir, subject, step_name, file_name)
    if not os.path.exists(path):
        return None
    return np.load(path)
