# ruff: noqa: T201, N802
"""MODULE_NAME — ONE_LINE_DESCRIPTION

DETAILED_DESCRIPTION

Usage:
    python -m commands.FILENAME <subject> [--threshold N]
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# ── 类型别名 ──────────────────────────────────────────────────────────
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

# 输入数组维度约定：(N, 3) 顶点坐标
EXPECTED_NDIM = 2
POINT_DIM = 3


# ── 公开函数（在前） ──────────────────────────────────────────────────

def compute_FUNC_NAME(
    input_array: FloatArray,
    threshold: float = 1.0,
    *,
    debug: bool = False,
) -> dict[str, Any]:
    """计算 FUNC_DESCRIPTION。

    Args:
        input_array: (N, 3) 顶点坐标数组。
        threshold: 角度阈值（度），默认 1.0。
        debug: 是否返回 debug 信息。

    Returns:
        dict: {
            "result": FloatArray — 计算结果 (M, 3)，
            "indices": IntArray — 顶点索引 (M,)，
            "debug": dict — 中间数据（仅 debug=True）
        }

    Raises:
        ValueError: 输入为空或维度不匹配时。
    """
    if input_array.size == 0:
        return {"result": np.empty((0, 3)), "indices": np.empty(0, dtype=int)}

    if input_array.ndim != EXPECTED_NDIM or input_array.shape[1] != POINT_DIM:
        raise ValueError(
            f"input_array 应为 (N, 3)，实际 {input_array.shape}"
        )

    # ── 核心逻辑 ──────────────────────────────────────────────────
    # TODO: 实现主算法
    result = input_array.copy()
    indices = np.arange(len(input_array))

    out: dict[str, Any] = {
        "result": result,
        "indices": indices,
    }
    if debug:
        out["debug"] = {
            "intermediate": None,
        }
    return out


def validate_result(
    result: dict[str, Any],
    expected_shape: tuple[int, int] | None = None,
) -> bool:
    """验证计算结果完整性。

    Args:
        result: compute_* 返回的字典。
        expected_shape: 期望的 shape，如 (N, 3)。

    Returns:
        True 验证通过，否则 False。
    """
    if not isinstance(result, dict):
        return False
    if "result" not in result or "indices" not in result:
        return False
    r = result["result"]
    if not isinstance(r, np.ndarray) or r.ndim != EXPECTED_NDIM:
        return False
    return not (expected_shape and r.shape != expected_shape)


# ── 私有函数（在后） ─────────────────────────────────────────────────

def _validate_input(array: FloatArray) -> None:
    """验证输入数组合法性。"""
    if not np.isfinite(array).all():
        raise ValueError("输入数组包含 NaN 或 Inf")


# ── Demo / 自检 ─────────────────────────────────────────────────────

def demo() -> None:
    """冒烟测试 —— 不依赖 pytest，可独立运行验证。"""
    data = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=float)
    out = compute_FUNC_NAME(data)
    assert "result" in out, "compute_* 应返回 result"
    assert out["result"].shape == data.shape, "shape 应一致"
    assert validate_result(out), "validate_result 应通过"
    print(f"✅ demo 通过: shape={out['result'].shape}")


if __name__ == "__main__":
    demo()
