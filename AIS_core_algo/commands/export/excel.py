#!/usr/bin/env python3
"""生成 Table 1-6 Excel 文件，对齐 HTML 版本布局和数据。

用法:
    uv run python -m commands.export.excel

拆分说明: 表格生成函数移至 commands.export.excel_tables，本文件仅保留 CLI 入口。
"""

from __future__ import annotations

from pathlib import Path

from commands.export.excel_tables import (
    make_table1,
    make_table2,
    make_table3,
    make_table4,
    make_table5,
    make_table6,
)
from utils.logger import logger


def main(out_dir: Path | None = None):
    """生成 Table 1-6 Excel（读指定目录 raw CSV，输出同目录 xlsx）。"""
    make_table1(out_dir)
    make_table2(out_dir)
    make_table3(out_dir)
    make_table4(out_dir)
    make_table5(out_dir)
    make_table6(out_dir)
    logger.info("\n全部 Excel 表格已生成。")


if __name__ == "__main__":
    main()
