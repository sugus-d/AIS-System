"""export 包冒烟测试 — 验证 import 链不崩。"""

from __future__ import annotations

from pathlib import Path


def test_import_paths() -> None:
    from utils.paths import EXPORT_DIR, RESULTS_DIR

    assert isinstance(RESULTS_DIR, Path)
    assert isinstance(EXPORT_DIR, Path)


def test_import_main() -> None:
    import commands.export  # noqa: F401
