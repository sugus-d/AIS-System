#!/usr/bin/env python3
"""统一绘图入口——按 --domain 选择绘图类型。

合并自 plot_compare_subjects / plot_landmarks / plot_parameterization / plot_roughness 四个脚本。
拆分为 6 个模块：
  - plot_shared.py: 共享辅助函数
  - plot_compare.py: render_compare
  - plot_landmarks.py: render_landmarks
  - plot_parameterization.py: render_parameterization
  - plot_roughness.py: render_roughness
  - plot.py: CLI 入口（_build_parser + main）

用法:
    uv run python -m commands.plot --domain compare S0119,S0113
    uv run python -m commands.plot --domain landmarks S0004
    uv run python -m commands.plot --domain parameterization S0004 --show-heightmap
    uv run python -m commands.plot --domain roughness S0119 --threshold 0.20
"""

from __future__ import annotations

import argparse

import matplotlib

from commands.plot_compare import render_compare
from commands.plot_landmarks import render_landmarks
from commands.plot_parameterization import render_parameterization
from commands.plot_roughness import render_roughness
from utils.paths import ARCHIVE_DIR, CACHE_DIR, LANDMARKS_DIR, PARAM_DIR

# 各 domain 的默认输出目录（与原独立脚本一致）
_DEFAULT_OUTPUT_DIRS = {
    "compare": str(ARCHIVE_DIR / "debug_roi"),
    "landmarks": str(LANDMARKS_DIR),
    "parameterization": str(PARAM_DIR),
    "roughness": str(ARCHIVE_DIR / "debug_roi"),
}


def _build_parser() -> argparse.ArgumentParser:
    """构建 argparse 主入口，注册全部 domain 的公共与独有参数。"""
    parser = argparse.ArgumentParser(
        prog="plot.py",
        description="统一绘图入口：--domain 选择绘图类型（compare/landmarks/parameterization/roughness）。",
    )
    parser.add_argument("--domain", choices=sorted(_DEFAULT_OUTPUT_DIRS), required=True, help="绘图类型")
    parser.add_argument("subject", help="subject ID（compare 域可传逗号分隔的多个 ID）")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认按 domain 取各自默认值）")
    parser.add_argument("--skip-run", action="store_true", help="跳过 pipeline 重建缓存")
    # compare 独有
    parser.add_argument("--angle", type=int, default=15, help="切除角度")
    parser.add_argument("--dilate", type=int, default=0, help="桥面扩张层数")
    parser.add_argument("--min-area", type=float, default=150.0, help="最小面积 mm²")
    parser.add_argument("--min-al-ratio", type=float, default=5.0, help="面积/边长比")
    # landmarks 独有
    parser.add_argument("--cache-dir", default=str(CACHE_DIR), help="缓存目录")
    # parameterization 独有
    parser.add_argument("--cut-only", action="store_true", help="只画 cut 图")
    parser.add_argument("--show-heightmap", action="store_true", help="显示高度图")
    parser.add_argument("--show-surfaces", action="store_true", help="显示粗糙程度和法向量")
    parser.add_argument("--smoothing", type=float, default=5.0, help="测地线平滑 sigma")
    # roughness 独有
    parser.add_argument("--threshold", "-t", type=float, default=None, help="粗糙度阈值，不指定则自适应")
    return parser


def main() -> None:
    """CLI 入口：解析参数后按 --domain 分发到对应渲染函数。"""
    matplotlib.use("Agg")
    args = _build_parser().parse_args()
    output_dir = args.output_dir or _DEFAULT_OUTPUT_DIRS[args.domain]

    if args.domain == "compare":
        render_compare(
            args.subject, output_dir, args.skip_run, args.angle, args.dilate, args.min_area, args.min_al_ratio
        )
    elif args.domain == "landmarks":
        render_landmarks(args.subject, args.cache_dir, output_dir, args.skip_run)
    elif args.domain == "parameterization":
        render_parameterization(
            args.subject,
            output_dir,
            args.skip_run,
            args.cut_only,
            args.show_heightmap,
            args.show_surfaces,
            args.smoothing,
        )
    else:
        render_roughness(args.subject, output_dir, args.skip_run, args.threshold)


if __name__ == "__main__":
    main()
