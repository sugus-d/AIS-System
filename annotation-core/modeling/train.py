"""训练入口：加载特征方案 → 5×5 CV → 每折独立筛选（按方案配置）。

用法:
  python -m modeling.train --scheme morph_region_ci_40d        # 按方案名跑全部模型
  python -m modeling.train --scheme morph_region_ci_40d --algo Ridge # 单模型
  python -m modeling.train --list-schemes                    # 查看可用方案

加权模式:
  python -m modeling.train --scheme morph_region_ci_40d --algo HistGBRT --weighting inv_freq

入口关系（壳层归一）:
  - --scheme          → features.selectors.schemes.SELECTION_REGISTRY（特征方案）
  - --training-preset → modeling.training.presets.TRAINING_PRESETS（快捷预设，仅覆盖
                         weighting/calibrate/hp，与 --scheme 的完整方案不同层）
  - 统一走 Trainer（modeling.training.trainer）；常规模式启用 per-fold 嵌入式筛选
    （modeling.training.feature_selector，原 cross_validate 行为），加权模式用方案特征
  - 结果均写 results/modeling/prediction/（modeling.training.result_paths）
"""

# ruff: noqa: T201

from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import KFold

from features.selectors.schemes import SELECTION_REGISTRY as SCHEME_REGISTRY
from modeling._shared import CLINICAL
from modeling.contracts import FeatureSet, TrainingConfig
from modeling.metrics import compute_metrics
from modeling.models import REGISTRY as MODEL_REGISTRY
from modeling.training.result_paths import RESULTS_DIR
from modeling.training.trainer import Trainer
from utils.logger import logger

OUT_DIR = RESULTS_DIR
_DEFAULT_HP_N_ITER = 20  # run() 的 hp_n_iter 默认值（preset 解析后兜底）
_DEFAULT_SCHEME = "morph_region_ci_40d"  # --scheme 缺省时的默认特征方案（文档最优 🏆）

N_SPLITS = 5  # 交叉验证折数（与原 cross_validate 路径一致）
N_REPEATS = 5  # 交叉验证重复次数


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: object) -> object:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run(
    algo_filter: str | None = None,
    ensemble_only: bool = False,
    scheme_name: str | None = None,        # 方案名（必填）
    use_stratified: bool = True,
    hp_n_iter: int | None = None,          # HP 搜索次数；None=preset 或默认值兜底
    weighting: str | None = None,          # 加权策略: "inv_freq" 等
    calibrate: bool = False,               # 启用 per-class 后处理校准
    training_preset: str | None = None,    # 训练预设（覆盖上面的 weighting/calibrate/hp_n_iter）
) -> list[dict]:
    """全模型训练入口。

    Args:
        scheme_name: 方案名（features.selectors.schemes.SELECTION_REGISTRY 的 key）。
        weighting:   加权策略名。设置后路由到 pipeline Trainer 而非 cross_validate。
        calibrate:   是否启用 per-class 后处理偏差校正。
        training_preset: 训练预设名（modeling.training.presets.TRAINING_PRESETS）。如果指定，覆盖
                        weighting/calibrate/hp_n_iter。但 --algo 的优先级更高。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 方案选择（旧 --scheme-b* 别名已删除，注册表见 features/selectors/schemes.py）
    if scheme_name is None:
        scheme_name = _DEFAULT_SCHEME
        logger.info(f"未指定 --scheme，使用默认方案: {scheme_name}")
    scheme_label = scheme_name
    data = SCHEME_REGISTRY[scheme_name].load()
    y = data["y"]

    # 训练预设解析
    trainer_cls = None
    weight_components: list | None = None
    if training_preset:
        from modeling.training.presets import get_training_preset
        preset = get_training_preset(training_preset)
        pc = preset.config
        # 预设提供默认值，但 CLI 显式参数优先
        if weighting is None:
            weighting = pc.weighting
        if not calibrate:
            calibrate = pc.calibrate
        if hp_n_iter is None:
            hp_n_iter = pc.hp_searcher_params.get("n_iter", _DEFAULT_HP_N_ITER)
        if pc.weight_components:
            weight_components = pc.weight_components
        if algo_filter is None and pc.models:
            algo_filter = pc.models[0]
        trainer_cls = pc.trainer
        logger.info("训练预设: %s (weighting=%s, calibrate=%s, hp=%d, algo=%s, trainer=%s)",
                     training_preset, weighting, calibrate, hp_n_iter, algo_filter, trainer_cls)
    if hp_n_iter is None:
        hp_n_iter = _DEFAULT_HP_N_ITER

    # 加权模式 → 路由到 pipeline Trainer（preset 提供 weight_components 时同路由）
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
            print(f"\nEnsemble: {ens_name}  >>> [20°] F1={m['f1']:.3f} Sens={m['sens']:.3f} Spec={m['spec']:.3f} RMSE={m['rmse']:.2f}")
            all_results.append({
                "algo": ens_name, "r": m["r"], "rmse": m["rmse"],
                "f1": m["f1"], "sens": m["sens"], "spec": m["spec"],
                "cm": [int(m["tn"]), int(m["fp"]), int(m["fn"]), int(m["tp"])],
                "preds": [float(x) for x in preds],
                "best_params": {},
            })

    return all_results


def _train_one(algo: str, y: np.ndarray, data: dict, scheme_name: str,
               use_stratified: bool = True, hp_n_iter: int = 20) -> dict:
    """单模型训练：Trainer + per-fold 嵌入式筛选（原 cross_validate 行为）。

    Args:
        scheme_name: 特征方案名（结果目录/session 日志/TrainingResult.scheme 均用此名）。
    """
    logger.info(f"训练: {algo}, y∈[{y.min():.0f}°, {y.max():.0f}°]")
    print(f"\n训练: {algo}  y∈[{y.min():.0f}°, {y.max():.0f}°]")

    config = TrainingConfig(
        models=[algo],
        data_splitter="stratified_kfold" if use_stratified else "kfold",
        data_splitter_params={"n_splits": N_SPLITS, "n_repeats": N_REPEATS},
        hp_searcher_params={"n_iter": hp_n_iter, "score_metric": "r2"},
        transform_target=True,
        feature_selector="per_fold",
    )
    feature_set = FeatureSet(
        name=scheme_name,
        y=y,
        X=np.asarray(data["X_basic"]),
        feature_names=[],
        X_raw_blocks={
            k: np.asarray(v) for k, v in {
                "basic": data.get("X_basic"),
                "morph": data.get("X_morph"),
                "region": data.get("X_region_full"),
            }.items() if v is not None
        },
        region_column_names=data.get("region_col_names"),
    )
    result = Trainer(config).train(feature_set)[0]
    preds = np.array([float(x) for x in result.predictions])
    m = result.metrics
    logger.info(f"{algo} [20°] F1={m['f1']:.3f} Sens={m['sens']:.3f} Spec={m['spec']:.3f} RMSE={m['rmse']:.2f}")
    print(f"  >>> [20°] F1={m['f1']:.3f} Sens={m['sens']:.3f} Spec={m['spec']:.3f} RMSE={m['rmse']:.2f}")

    bp = result.best_params or {}
    if bp:
        logger.info(f"{algo} 最佳参数: {bp}")
    return {
        "algo": algo, "r": m["r"], "rmse": m["rmse"],
        "f1": m["f1"], "sens": m["sens"], "spec": m["spec"],
        "cm": [int(m["tn"]), int(m["fp"]), int(m["fn"]), int(m["tp"])],
        "preds": [float(x) for x in preds],
        "best_params": bp,
    }


def _ensemble_preds(ensemble_model: object, single_preds: dict[str, np.ndarray],
                    y: np.ndarray,
                    n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS) -> np.ndarray:
    """对多个单模型的折预测做集成加权（原 cross_validate_ensemble 逻辑）。"""
    model_names = list(single_preds.keys())
    P = np.column_stack([single_preds[name] for name in model_names])
    n = len(y)
    ap = np.zeros(n)
    ac = np.zeros(n)
    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
        for tr_idx, te_idx in kf.split(P):
            m = ensemble_model.__class__()
            m.fit(P[tr_idx], y[tr_idx])
            ap[te_idx] += m.predict(P[te_idx])
            ac[te_idx] += 1
    return ap / np.maximum(ac, 1)


def _run_weighted(
    y: np.ndarray,
    data: dict,
    scheme_label: str,
    algo_filter: str | None,
    weighting: str,
    hp_n_iter: int,
    calibrate: bool = False,
    trainer: str | None = None,
    weight_components: list | None = None,
) -> list[dict]:
    """加权模式 — 通过 pipeline Trainer 训练。

    与旧 cross_validate 路径不同的地方：
    - 使用 Trainer（新 pipeline 架构）
    - 支持样本加权和 per-class 后处理校准
    - 适用于预选固定特征方案（morph_region_ci_40d 等）

    Args:
        trainer: Trainer 实现选择。None=原版 Trainer, "margin"=MarginTrainer。
        weight_components: 预设提供的权重组件（优先）；None 时按 weighting 策略名构建。
    """
    # 延迟导入避免 circular import
    from modeling.contracts import FeatureSet, TrainingConfig  # noqa: PLC0415
    from modeling.training.weights import build_weight_components  # noqa: PLC0415
    if trainer == "margin":
        from modeling.training.trainer_margin import MarginTrainer as TrainerCls  # noqa: PLC0415
    else:
        from modeling.training.trainer import Trainer as TrainerCls  # noqa: PLC0415
    # 共享壳层：与 pipeline/run.py::_run_train 一致的路径签名 / 跳过 / 落盘
    from modeling.training.result_paths import (  # noqa: PLC0415
        extra_para_signature,
        find_existing_metrics,
        save_results,
    )

    X = data.get("X_basic")
    if X is None:
        logger.error("加权模式需要预选固定特征（X_basic），当前方案不支持")
        return []

    models_to_run = [algo_filter] if algo_filter else ["HistGBRT"]
    config = TrainingConfig(
        models=models_to_run,
        weighting=weighting,
        weight_components=weight_components if weight_components is not None
        else build_weight_components(weighting),
        hp_searcher_params={"n_iter": hp_n_iter, "score_metric": "r2"},
        calibrate=calibrate,
        trainer=trainer,
    )

    scheme = SCHEME_REGISTRY.get(scheme_label)
    fnames = getattr(scheme, "feature_names", None) or []
    feature_set = FeatureSet(name=scheme_label, y=y, X=X, feature_names=fnames)

    # 路径签名 + 跳过已有结果（与 pipeline/run.py::_run_train 一致）
    extra_para = extra_para_signature(config)
    out = []
    for model_name in models_to_run:
        existing = find_existing_metrics(scheme_label, "weighted", extra_para, model_name)
        if existing is not None:
            logger.info(f"  {model_name}: 结果已存在，跳过")
            continue

        config.models = [model_name]
        tr = TrainerCls(config)
        results = tr.train(feature_set)

        # 转为旧 dict 格式（向前兼容）+ 共享落盘
        for r in results:
            m = r.metrics
            cm = [int(m.get(k, 0)) for k in ("tn", "fp", "fn", "tp")]
            save_results(
                r.model_name,
                np.array([float(x) for x in r.predictions]),
                config,
                "weighted",
                extra_para,
                y,
                feat_scheme_name=scheme_label,
            )
            logger.info(f"{r.model_name} [加权={weighting}] "
                        f"F1={m['f1']:.3f} Sens={m['sens']:.3f} Spec={m['spec']:.3f} RMSE={m['rmse']:.2f}")
            print(f"\n{r.model_name} [加权={weighting}]  >>> "
                  f"F1={m['f1']:.3f} Sens={m['sens']:.3f} Spec={m['spec']:.3f} RMSE={m['rmse']:.2f}")
            out.append({
                "algo": r.model_name, "r": m["r"], "rmse": m["rmse"],
                "f1": m["f1"], "sens": m["sens"], "spec": m["spec"],
                "cm": cm,
                "preds": [float(x) for x in r.predictions],
                "best_params": r.best_params,
            })
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AIS 训练（统一方案入口）")

    # ---- 新接口 ----
    parser.add_argument("--scheme", type=str, default=None,
                        help="方案名（见 features.selectors.schemes.SELECTION_REGISTRY），缺省 morph_region_ci_40d")
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

    # ---- 训练预设 ----
    parser.add_argument("--training-preset", type=str, default=None,
                        help="训练预设（见 modeling.training.presets.TRAINING_PRESETS），如 baseline/weighted_inv/weighted_9x")
    parser.add_argument("--list-presets", action="store_true",
                        help="列出所有训练预设")

    # ---- 加权模式 ----
    parser.add_argument("--weighting", type=str, default=None,
                        choices=["inv_freq", "uniform", "per_class", "severe_boost"],
                        help="样本加权策略: inv_freq=逆频率加权, severe_boost=Severe强化, per_class=自定义")
    parser.add_argument("--calibrate", action="store_true",
                        help="启用 per-class CV 后偏差校正")
    args = parser.parse_args()

    # --list-presets 快速查看
    if args.list_presets:
        from modeling.training.presets import TRAINING_PRESETS
        print(f"{'预设名':<22} {'标签':<12} {'模型':<12} {'加权':<14} {'校准':<5} {'HP':<5}")
        print("-" * 80)
        for name, p in sorted(TRAINING_PRESETS.items()):
            c = p.config
            print(f"{name:<22} {p.label:<12} {c.models[0]:<12} {c.weighting or 'none':<14} {'✓' if c.calibrate else '✗':<5} {c.hp_searcher_params.get('n_iter', 20):<5}")
        return

    # --list-schemes 快速查看
    if args.list_schemes:
        print(f"{'名称':<24} {'标签':<16} {'维度':<5} {'选择方式':<18} {'组件'}")
        print("-" * 100)
        for scheme in SCHEME_REGISTRY.values():
            print(f"{scheme.name:<24} {scheme.label:<16} {scheme.n_features:<5} {scheme.selection_method:<18} {scheme.components}")
        return

    results = run(
        algo_filter=args.algo, ensemble_only=args.ensemble_only,
        scheme_name=args.scheme,
        use_stratified=not args.kfold,
        hp_n_iter=args.thorough,
        weighting=args.weighting,
        calibrate=args.calibrate,
        training_preset=args.training_preset,
    )

    if not args.ensemble_only:
        p = OUT_DIR / "all_results.json"
        with open(p, "w") as f:
            json.dump(results, f, indent=2, cls=_NumpyEncoder)
        logger.info(f"保存: {p}")

    if results:
        print(f"\n{'='*80}\n汇总\n{'='*80}")
        print(f"{'算法':<14} {'F1':<7} {'Sens':<7} {'Spec':<7} {'RMSE':<7} {'r':<7}")
        print("-" * 60)
        for r in results:
            print(f"{r['algo']:<14} {r['f1']:.3f}  {r['sens']:.3f}  {r['spec']:.3f}  {r['rmse']:<6.2f} {r['r']:.3f}")


if __name__ == "__main__":
    main()
