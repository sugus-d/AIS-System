"""训练入口：加载特征方案 → 5×5 CV → 每折独立筛选（按方案配置）。

用法:
  python -m modeling.train --scheme v0.1.0        # 按方案名跑全部模型
  python -m modeling.train --scheme v0.1.0 --algo Ridge # 单模型
  python -m modeling.train --list-schemes                    # 查看可用方案

加权模式:
  python -m modeling.train --scheme v0.1.0 --algo HistGBRT --weighting inv_freq

入口关系（壳层归一）:
  - --scheme          → features.selectors.schemes.SELECTION_REGISTRY（特征方案）
  - 统一走 Trainer（modeling.training.trainer）；常规模式启用 per-fold 嵌入式筛选
    （modeling.training.feature_selector，原 cross_validate 行为），加权模式用方案特征
  - 结果均写 results/modeling/prediction/（modeling.training.result_paths）

拆分说明: 内部实现（_NumpyEncoder/_train_one/_ensemble_preds/_run_weighted）移至
modeling.train_helpers，本文件保留 run/main CLI 入口。
"""

from __future__ import annotations

import json

import numpy as np

from features.selectors.schemes import SELECTION_REGISTRY as SCHEME_REGISTRY
from modeling._shared import CLINICAL
from modeling.metrics import compute_metrics
from modeling.models import REGISTRY as MODEL_REGISTRY
from modeling.train_helpers import (
    _ensemble_preds,
    _NumpyEncoder,
    _run_weighted,
    _train_one,
)
from modeling.training.result_paths import RESULTS_DIR
from utils.logger import logger

OUT_DIR = RESULTS_DIR
_DEFAULT_HP_N_ITER = 20  # run() 的 hp_n_iter 默认值
_DEFAULT_SCHEME = "v0.1.0"  # --scheme 缺省时的默认特征方案（文档最优 🏆）


def run(
    algo_filter: str | None = None,
    ensemble_only: bool = False,
    scheme_name: str | None = None,        # 方案名（必填）
    use_stratified: bool = True,
    hp_n_iter: int | None = None,          # HP 搜索次数；None=默认值兜底
    weighting: str | None = None,          # 加权策略: "inv_freq" 等
    calibrate: bool = False,               # 启用 per-class 后处理校准
    ensemble: bool = False,                # 生成 ensemble 预测（0.6×CompositeV7 + 0.4×AI-LR）
) -> list[dict]:
    """全模型训练入口。

    Args:
        scheme_name: 方案名（features.selectors.schemes.SELECTION_REGISTRY 的 key）。
        weighting:   加权策略名。设置后路由到 pipeline Trainer 而非 cross_validate。
        calibrate:   是否启用 per-class 后处理偏差校正。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 方案选择（旧 --scheme-b* 别名已删除，注册表见 features/selectors/schemes.py）
    if scheme_name is None:
        scheme_name = _DEFAULT_SCHEME
        logger.info(f"未指定 --scheme，使用默认方案: {scheme_name}")
    scheme_label = scheme_name

    # ensemble 模式：生成 manuscript 集成预测（默认复用现有 CompositeV7，不训练）
    if ensemble:
        from modeling.ensemble_train import train_ensemble

        path = train_ensemble(
            scheme_name=scheme_name,
            model_name=algo_filter or "HistGBRT",
            hp_n_iter=hp_n_iter or 5,
        )
        logger.info(f"ensemble 已生成: {path}")
        return []

    data = SCHEME_REGISTRY[scheme_name].load()
    y = data["y"]

    # 训练参数（无 preset：weighting/calibrate 由 CLI 直接传）
    trainer_cls = None
    weight_components: list | None = None
    if hp_n_iter is None:
        hp_n_iter = _DEFAULT_HP_N_ITER

    # 加权模式 → 路由到 pipeline Trainer（weight_components 提供时同路由）
    if weighting or weight_components:
        return _run_weighted(
            y=y, data=data, scheme_label=scheme_label,
            algo_filter=algo_filter,
            weighting=weighting,
            weight_components=weight_components,
            hp_n_iter=hp_n_iter,
            calibrate=calibrate,
            trainer=trainer_cls,
        )

    # ── 常规模式（旧 cross_validate 路径） ──
    all_names = list(MODEL_REGISTRY.keys())
    single_names = [m for m in all_names if not m.startswith("Ensemble")]
    ensemble_names = [m for m in all_names if m.startswith("Ensemble")]

    all_results: list[dict] = []
    single_preds: dict[str, np.ndarray] = {}

    if not ensemble_only:
        models_to_run = [algo_filter] if algo_filter else single_names
        for algo in models_to_run:
            if algo not in MODEL_REGISTRY:
                logger.warning(f"未知模型: {algo}，跳过")
                continue
            result = _train_one(algo, y, data, scheme_name=scheme_name,
                                use_stratified=use_stratified, hp_n_iter=hp_n_iter)
            all_results.append(result)
            single_preds[algo] = np.array(result["preds"])

    # Ensemble
    if single_preds and (ensemble_only or not algo_filter):
        for ens_name in ensemble_names:
            ens = MODEL_REGISTRY[ens_name]()
            preds = _ensemble_preds(ens, single_preds, y)
            m = compute_metrics(y, preds, threshold=CLINICAL)
            logger.info(f"Ensemble {ens_name}: F1={m['f1']:.3f} Sens={m['sens']:.3f} Spec={m['spec']:.3f} RMSE={m['rmse']:.2f}")
            all_results.append({
                "algo": ens_name, "r": m["r"], "rmse": m["rmse"],
                "f1": m["f1"], "sens": m["sens"], "spec": m["spec"],
                "cm": [int(m["tn"]), int(m["fp"]), int(m["fn"]), int(m["tp"])],
                "preds": [float(x) for x in preds],
                "best_params": {},
            })

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AIS 训练（统一方案入口）")

    # ---- 新接口 ----
    parser.add_argument("--scheme", type=str, default=None,
                        help="方案名（见 features.selectors.schemes.SELECTION_REGISTRY），缺省 v0.1.0")
    parser.add_argument("--list-schemes", action="store_true",
                        help="列出所有注册方案")

    # ---- 模型 ----
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--ensemble-only", action="store_true")

    # ---- CV / HP ----
    parser.add_argument("--kfold", action="store_true",
                        help="使用 KFold（默认 StratifiedKFold）")
    parser.add_argument("--thorough", type=int, nargs="?", const=40, default=None,
                        help="HP 搜索次数（默认 20，--thorough=40，--thorough=60 等）")

    # ---- 加权模式 ----
    parser.add_argument("--weighting", type=str, default=None,
                        choices=["inv_freq", "uniform", "per_class", "severe_boost"],
                        help="样本加权策略: inv_freq=逆频率加权, severe_boost=Severe强化, per_class=自定义")
    parser.add_argument("--calibrate", action="store_true",
                        help="启用 per-class CV 后偏差校正")

    # ---- ensemble ----
    parser.add_argument("--ensemble", action="store_true",
                        help="生成 manuscript ensemble 预测（0.6×CompositeV7 + 0.4×AI-LR），复用现有 CompositeV7 结果")
    args = parser.parse_args()

    # --list-schemes 快速查看
    if args.list_schemes:
        logger.info(f"{'名称':<24} {'标签':<16} {'维度':<5} {'选择方式':<18} {'组件'}")
        logger.info("-" * 100)
        for scheme in SCHEME_REGISTRY.values():
            logger.info(f"{scheme.name:<24} {scheme.label:<16} {scheme.n_features:<5} {scheme.selection_method:<18} {scheme.components}")
        return

    results = run(
        algo_filter=args.algo, ensemble_only=args.ensemble_only,
        scheme_name=args.scheme,
        use_stratified=not args.kfold,
        hp_n_iter=args.thorough,
        weighting=args.weighting,
        calibrate=args.calibrate,
        ensemble=args.ensemble,
    )

    if not args.ensemble_only and not args.ensemble:
        p = OUT_DIR / "all_results.json"
        with open(p, "w") as f:
            json.dump(results, f, indent=2, cls=_NumpyEncoder)
        logger.info(f"保存: {p}")

    if results:
        logger.info("=" * 80)
        logger.info("汇总")
        logger.info("=" * 80)
        logger.info(f"{'算法':<14} {'F1':<7} {'Sens':<7} {'Spec':<7} {'RMSE':<7} {'r':<7}")
        logger.info("-" * 60)
        for r in results:
            logger.info(f"{r['algo']:<14} {r['f1']:.3f}  {r['sens']:.3f}  {r['spec']:.3f}  {r['rmse']:<6.2f} {r['r']:.3f}")


if __name__ == "__main__":
    main()
