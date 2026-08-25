"""CLI 通用框架——为所有 plot_*.py 提供标准化的 typer CLI 入口。

用法
----
    @plot_cli(rebuild_steps=("load_mesh", "extract_roi", "curvature", "landmarks"))
    def render(subject: str, cache_dir: str = "results/cache", ...):
        ...
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer


def _auto_start(wrapper: Callable, caller_name: str | None = None) -> None:
    """自动启动 typer CLI（仅当在 __main__ 模块中定义时触发）。"""
    if caller_name == "__main__":
        typer.run(wrapper)


def _get_caller_name() -> str | None:
    """获取装饰器调用处模块的 __name__。"""
    _frame = inspect.currentframe()
    _caller_frame = _frame.f_back if _frame else None
    _caller_module = _caller_frame.f_back if _caller_frame else None
    name = _caller_module.f_globals.get("__name__") if _caller_module else None
    del _frame, _caller_frame, _caller_module
    return name


def find_mesh_path(subject_id: str, mesh_dir: Path) -> Path | None:
    """根据 subject ID 在 mesh_dir 下查找 STD_fuse_mesh PLY 文件路径。

    模板文件可能不存在或尚未生成，需要先检查目录是否存在。
    取排序后的第一个 PLY 文件（通常只有一个）。

    Args:
        subject_id: Subject ID（作为子目录名）。
        mesh_dir: 原始 mesh 存放目录（调用方按自身运行 cwd 提供）。

    Returns:
        匹配的 PLY 文件路径；目录或文件不存在时返回 None。
    """
    subject_dir = mesh_dir / subject_id
    if not subject_dir.is_dir():
        return None
    ply_files = sorted(subject_dir.glob("STD_fuse_mesh_*.ply"))
    return ply_files[0] if ply_files else None


def plot_cli(
    rebuild_steps: tuple[str, ...] | None = None,
    default_output: str = "results/landmarks",
) -> Callable:
    """装饰器：将渲染函数包装为标准化 typer CLI 应用。"""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 — 泛型装饰器，参数类型由被包装函数决定
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return fn(*bound.args, **bound.kwargs)
        _auto_start(wrapper, _get_caller_name())
        return wrapper
    return decorator


def app_cli() -> Callable:
    """装饰器：通用 CLI 入口，无预设参数。"""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 — 泛型装饰器，参数类型由被包装函数决定
            return fn(*args, **kwargs)
        _auto_start(wrapper, _get_caller_name())
        return wrapper
    return decorator


def batch_cli(default_subjects: str = "") -> Callable:
    """装饰器：批处理 CLI，自动注入 subjects 位置参数。"""
    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        if "subjects" not in sig.parameters:
            original = fn
            @functools.wraps(fn)
            def patched(subjects: str = typer.Argument(default_subjects, help="逗号分隔的 subject ID 列表"),
                        **kwargs: Any) -> Any:  # noqa: ANN401 — 泛型装饰器，参数类型由被包装函数决定
                return original(subjects=subjects, **kwargs)
            fn = patched
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 — 泛型装饰器，参数类型由被包装函数决定
            return fn(*args, **kwargs)
        _auto_start(wrapper, _get_caller_name())
        return wrapper
    return decorator
