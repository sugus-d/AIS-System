"""核心算法数值黄金测试的共享工具。

黄金值 = 已验证正确的 HEAD 输出（shape + 总和 + md5 三要素，md5 逐字节）。
任何代码改动导致数值偏差 → 断言失败；有意的算法变更必须显式更新黄金值
（重跑 tests/numerics/_generate_golden.py 并人工确认新值正确后替换断言）。

生成环境锁定：numpy 版本变化可能触发末位浮点差异导致误报。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "numerics"
RNG_SEED = 2026


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: 端到端数值黄金测试（ROI/landmark/训练，耗时较长）")


def fingerprint(val: object) -> tuple[str, str, str]:
    """计算 (shape 字符串, sum 字符串, md5)。dict 用 repr 序列化。"""
    if isinstance(val, dict):
        payload = repr(sorted(val.items())).encode()
        return "dict", "-", hashlib.md5(payload).hexdigest()
    arr = np.ascontiguousarray(np.asarray(val))
    return str(arr.shape), f"{float(arr.sum()):.10f}", hashlib.md5(arr.tobytes()).hexdigest()


def assert_golden(name: str, val: object, shape: str, total: str, md5: str) -> None:
    """断言数值与黄金值逐位一致。"""
    got_shape, got_sum, got_md5 = fingerprint(val)
    assert got_shape == shape, f"{name}: shape {got_shape} != 黄金值 {shape}"
    assert got_sum == total, f"{name}: 总和 {got_sum} != 黄金值 {total}"
    assert got_md5 == md5, f"{name}: md5 不一致 —— 数值已改变！"


def walk_golden(name: str, val: object, golden: dict[str, tuple[str, str, str]]) -> None:
    """递归遍历 dict/数组，对叶子断言黄金值（命名与 _generate_golden.py 的 walk_dict 一致）。"""
    if isinstance(val, dict):
        for key, sub in val.items():
            walk_golden(f"{name}_{key}", sub, golden)
    elif isinstance(val, (list, tuple, np.ndarray)):
        assert_golden(name, np.asarray(val, dtype=float), *golden[name])
