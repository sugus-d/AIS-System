"""通用注册表工厂 — 「名称 → 实现」字典注册模式。

mesh/roi 与 landmarks 两个注册表原本各复制一份 register/get/list 三件套，
出现第 3 个同构注册表（features/extractors）后核查其为死代码已删除；活跃
注册表统一到本工厂。消费方经 get()/list_xxx() 取用，接口不变。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def make_registry(
    label: str,
) -> tuple[Callable[[str, T], None], Callable[[str], T], Callable[[], list[str]]]:
    """创建「名称 → 实现」注册表。

    Args:
        label: 注册表语义名（KeyError 消息用，如 "ROI 算法"、"landmark 检测器"）。

    Returns:
        (register, get, list_names) 三件套：register 接受显式 name（不依赖对象
        属性）；get 未知名抛 KeyError；list_names 返回已注册名称。消费方可把
        list_names 别名包装为 list_xxx 保持语义名。
    """
    _store: dict[str, T] = {}

    def register(name: str, impl: T) -> None:
        _store[name] = impl

    def get(name: str) -> T:
        if name not in _store:
            raise KeyError(f"未知 {label}: {name}，可选: {list(_store.keys())}")
        return _store[name]

    def list_names() -> list[str]:
        return list(_store.keys())

    return register, get, list_names
