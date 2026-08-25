#!/usr/bin/env python3
"""AIS 预测 CLI 入口 — 三模式（landmarks / predict / auto）。

与 HTTP API（`prediction/api.py`）共用同一核心 `prediction.predict`，
输入输出契约完全一致，只是渠道不同（命令行 vs HTTP）。

用法:
  python -m prediction.cli landmarks --ply data/x.ply --subject S0001
  python -m prediction.cli predict --ply prediction/outputs/S0001/roi.ply --subject S0001 \\
      --clinical data/form/clinical_data.json --landmarks prediction/outputs/S0001/landmarks.json \\
      [--model v1.0.0|v0.1.0|<joblib 路径>]
  python -m prediction.cli auto --ply data/x.ply --subject S0001 \\
      --clinical data/form/clinical_data.json [--model v1.0.0|v0.1.0|<joblib 路径>]

模型选择 `--model`：缺省 `v1.0.0`（生产模型）；`v0.1.0` 为历史 manuscript 复现口径
（算法 ROI + 0.6×CompositeV7 + 0.4×AI-LR Ensemble）；也接受直接 joblib 路径。
别名 production/beta 亦可用。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prediction.model_registry import resolve_model_path
from prediction.predict import (
    _predict_flow,
    PREDICT_ROOT,
    run_landmarks,
)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--ply",
        required=True,
        help="输入 PLY mesh：landmarks/auto 为原始扫描，predict 为 ROI 网格（roi.ply）",
    )
    p.add_argument("--subject", required=True, help="subject ID（输出目录名）")
    p.add_argument("--output", default=None, help="输出根目录（默认 prediction/outputs）")


def main() -> None:
    """CLI 入口：三模式（landmarks / predict / auto），共用 prediction.predict 核心。"""
    parser = argparse.ArgumentParser(description="AIS 预测入口 — PLY → landmarks / cobb 预测 + 报告")
    sub = parser.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("landmarks", help="PLY → landmarks.json")
    _add_common(p1)

    p2 = sub.add_parser("predict", help="ROI + clinical + landmarks → cobb 预测")
    _add_common(p2)
    p2.add_argument("--clinical", required=True, help="临床数据 JSON（身高/体重等）")
    p2.add_argument(
        "--landmarks",
        required=True,
        help="landmarks.json（人工 ground_truth 或 landmarks 命令产出的自动结果）",
    )
    p2.add_argument(
        "--model",
        default="v1.0.0",
        help="模型选择：v1.0.0（缺省，生产）/ v0.1.0（历史）/ 直接 joblib 路径；别名 production/beta",
    )

    p3 = sub.add_parser("auto", help="PLY + clinical → 自动 landmarks → cobb 预测")
    _add_common(p3)
    p3.add_argument("--clinical", required=True, help="临床数据 JSON（身高/体重等）")
    p3.add_argument(
        "--model",
        default="v1.0.0",
        help="模型选择：v1.0.0（缺省，生产）/ v0.1.0（历史）/ 直接 joblib 路径；别名 production/beta",
    )

    args = parser.parse_args()
    out_root = Path(args.output) if args.output else PREDICT_ROOT
    out_dir = out_root / args.subject

    if args.mode == "landmarks":
        run_landmarks(args.ply, args.subject, out_dir)
    elif args.mode == "predict":
        # predict 模式输入已是 ROI + landmarks，不重复产出 roi.ply/landmarks.json
        model_path = resolve_model_path(args.model)
        _predict_flow(args.ply, args.subject, args.clinical, args.landmarks, model_path, out_dir, persist_roi_lm=False)
    elif args.mode == "auto":
        model_path = resolve_model_path(args.model)
        _predict_flow(args.ply, args.subject, args.clinical, None, model_path, out_dir)


if __name__ == "__main__":
    main()
