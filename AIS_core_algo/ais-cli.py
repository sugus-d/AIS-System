#!/usr/bin/env python3
"""AIS 管线 CLI — 可拼接的多步骤执行入口。

用法
══════════════════════════════════════════════════════════

  # 训练
  python ais-cli.py --step train --model Ridge --train-scheme margin_inv

  # 全管线
  python ais-cli.py --step roi --step feature_eng --step train

  # 列出可用步骤
  python ais-cli.py --list-steps

训练入口（等效）:
  python -m modeling.train --scheme v0.1.0 --algo HistGBRT
"""

# ruff: noqa: T201 — CLI 入口脚本，print 作为 CLI 输出

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AIS 管线 — 可拼接的多步骤执行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 配置文件路径")
    parser.add_argument("--step", type=str, action="append", default=None,
                        help="执行指定步骤，可多次使用")
    parser.add_argument("--roi-algo", type=str, default=None,
                        help="ROI 提取算法")
    parser.add_argument("--feature-eng", type=str, default=None,
                        help="特征工程方案")
    parser.add_argument("--model", type=str, nargs="+", default=None,
                        help="模型名列表（如 --model Ridge SVR XGBoost）")
    parser.add_argument("--train-scheme", type=str, default=None,
                        help="训练预设名（如 margin_inv, weighted_inv, baseline）")
    parser.add_argument("--label", type=str, default=None,
                        help="结果目录标签（追加到目录名，如 --label final_v1）")
    parser.add_argument("--para", type=str, action="append", default=None,
                        help="步骤参数 step:key=value（如 --para train:thorough=1 --para train:cv=3）")
    parser.add_argument("--list-steps", action="store_true",
                        help="列出可用步骤")
    parser.add_argument("--list-schemes", action="store_true",
                        help="列出所有训练方案")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_steps:
        _list_steps()
        return
    if args.list_schemes:
        _list_schemes()
        return

    from commands.pipeline import run

    # 构造覆盖参数
    overrides: dict[str, dict] = {}
    if args.step:
        for s in args.step:
            if s == "train":
                override: dict = {}
                if args.model:
                    override["model"] = args.model
                if args.train_scheme:
                    override["scheme"] = args.train_scheme
                if args.label:
                    override.setdefault("params", {})["_label"] = args.label
                overrides["train"] = override
            elif s == "feature_eng":
                if args.feature_eng:
                    overrides["feature_eng"] = {"scheme": args.feature_eng}
            elif s == "roi":
                if args.roi_algo:
                    overrides["roi"] = {"algo": args.roi_algo}

    # 解析 --para step:key=value 参数
    if args.para:
        for kv in args.para:
            if ":" not in kv or "=" not in kv:
                print(f"  ⚠ --para 格式错误: {kv}（应为 step:key=value）")
                continue
            step_part, rest = kv.split(":", 1)
            k, v_raw = rest.split("=", 1)
            step_name = step_part.strip()
            # 类型推断：true/false → bool, 数字 → int/float, 其他 → str
            v_lower = v_raw.lower()
            if v_lower in ("true", "false"):
                v = v_lower == "true"
            elif v_lower == "none":
                v = None
            else:
                try:
                    v = int(v_raw) if "." not in v_raw else float(v_raw)
                except ValueError:
                    import ast
                    try:
                        v = ast.literal_eval(v_raw)
                    except (ValueError, SyntaxError):
                        v = v_raw
            if step_name not in overrides:
                overrides[step_name] = {}
            overrides[step_name].setdefault("params", {})[k] = v

    if args.config:
        result = run(config_path=args.config, steps=args.step, overrides=overrides or None)
    else:
        result = run(steps=args.step, overrides=overrides or None)

    for step_name, output in result.items():
        if isinstance(output, dict):
            print(f"\n  [{step_name}] 完成")
            for model_name, metrics in output.items():
                print(f"    {model_name}: RMSE={metrics.get('rmse', '?'):>6.2f}  F1={metrics.get('f1', '?'):>6.3f}  r={metrics.get('r', '?'):>6.3f}")
        else:
            print(f"  [{step_name}] {output}")


def _list_steps() -> None:
    print("可用步骤:\n")
    print("  roi  — ROI/decloth prelabeling (preview; 不适用于生产)")
    print("         生产路径：commands/batch_process_all.py → annotation-platform 人工修正")
    print("         算法: bfs (BFS 种子生长), pants_cut, xy_hull\n")
    print("  feature_eng  — 特征工程")
    print("         方案: 全部来自 features/selectors（默认 v0.1.0，archived/* 为历史保留）\n")
    print("  train  — 模型训练")
    print("         模型: Ridge, SVR, ElasticNet, DecisionTree, BaggingEN, ...")
    print("         方案: baseline(base) / composite_v7(c7) / weighted_inv(winv) / severe_boost(sboost) 等 17 个（--list-schemes 查看全部）")
    print("  python ais-cli.py --list-schemes  # 查看全部训练方案\n")


def _list_schemes() -> None:
    from modeling.training.schemes import list_schemes
    print("可用训练方案:\n")
    for name in list_schemes():
        print(f"  {name}")
    print()


if __name__ == "__main__":
    main()
