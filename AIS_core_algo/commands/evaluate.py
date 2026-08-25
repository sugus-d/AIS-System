#!/usr/bin/env python3
"""统一评估入口 — 按 --domain 选择评估类型。

拆分自原 commands/evaluate.py（924 行），现按 domain 划分:
  - evaluate_cut.py:  曲线切割评估 + 批量参数组合评估
  - evaluate_roi.py:  ROI 全链路验收 + 指标计算
  - evaluate.py:      CLI 入口（main + argparse）

用法:
  uv run python -m commands.evaluate --domain batch                  # 批量参数组合评估
  uv run python -m commands.evaluate --domain cut [--subjects ...]   # 曲线切割评估
  uv run python -m commands.evaluate --domain roi [--subjects | --old | --skip-run | --regions | --regions-only | --stats | --output ...]  # ROI 全链路验收
"""

from __future__ import annotations

import argparse

from commands.evaluate_cut import evaluate_batch, evaluate_cut
from commands.evaluate_roi import evaluate_roi, ROI_OUTPUT_DIR


def main() -> None:
    """统一评估入口 — --domain 必选，选择评估类型。"""
    parser = argparse.ArgumentParser(description="统一评估入口 — 按 --domain 选择评估类型")
    parser.add_argument(
        "--domain",
        required=True,
        choices=["batch", "cut", "roi"],
        help="评估类型: batch=批量参数组合 / cut=曲线切割 / roi=ROI 验收",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--subjects", type=str, help="[cut/roi] 逗号分隔的 subject ID")
    group.add_argument("--old", action="store_true", help="[roi] 评估旧算法 baseline")
    parser.add_argument("--skip-run", action="store_true", help="[roi] 复用已有 mesh，不重新跑 pipeline")
    parser.add_argument("--regions", action="store_true", help="[roi] 执行区域化三角面评估（与现有指标并行）")
    parser.add_argument("--regions-only", action="store_true", help="[roi] 只跑区域评估，跳过现有指标")
    parser.add_argument("--stats", action="store_true", help="[roi] 对全部 GT subject 跑统计，生成阈值文件")
    parser.add_argument("--output", type=str, default=ROI_OUTPUT_DIR, help="[roi] 输出目录")
    args = parser.parse_args()

    if args.domain == "batch":
        evaluate_batch()
    elif args.domain == "cut":
        evaluate_cut(args.subjects)
    else:
        evaluate_roi(
            subjects=[s.strip() for s in args.subjects.split(",")] if args.subjects else None,
            use_old=args.old,
            skip_run=args.skip_run,
            output_dir=args.output,
            regions=args.regions,
            regions_only=args.regions_only,
            stats=args.stats,
        )


if __name__ == "__main__":
    main()
