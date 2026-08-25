#!/usr/bin/env python3
"""CLI entry point for the parameterisation pipeline.

用法:
    uv run python -m commands.run_parameterization S0004
    uv run python -m commands.run_parameterization S0004 --smoothing 3.0
    uv run python -m commands.run_parameterization S0004 --output results/parameterization
"""

from __future__ import annotations

import os

from commands.cli_common import app_cli
from utils.logger import logger


@app_cli()
def main(
    subject: str,
    output: str = "results/parameterization",
    smoothing: float = 5.0,
) -> None:
    """运行完整的参数化 pipeline。

    包含：测地线边界提取、调和 UV 映射、高度图生成等步骤。
    参数化结果用于后续的对称性分析、截面提取、展平可视化等。

    Args:
        subject: Subject ID.
        output: 输出目录，存放 UV / 高度图等中间结果。
        smoothing: 平滑强度（mm），越大轮廓越平滑。
    """
    from parameterization.pipeline import run_pipeline

    # 参数化结果写入独立目录，避免与 pipeline 缓存混在一起
    os.makedirs(output, exist_ok=True)
    logger.info(f"Running parameterisation pipeline for {subject}")
    run_pipeline(subject, output, smoothing)


if __name__ == "__main__":
    main()
