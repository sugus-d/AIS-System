"""训练辅助实现 — modeling.train 拆出的内部函数。

拆分说明: 原 modeling/train.py 的 _NumpyEncoder / _train_one / _ensemble_preds /
_run_weighted 移至此文件，train.py 保留 run/main CLI 入口（公共 API 不变）。
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import KFold

from features.selectors.schemes import SELECTION_REGISTRY as SCHEME_REGISTRY
from modeling.contracts import FeatureSet, TrainingConfig
from modeling.training.save_model import save_trained_model
from modeling.training.trainer import Trainer
from utils.logger import logger

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


def _train_one(algo: str, y: np.ndarray, data: dict, scheme_name: str,
               use_stratified: bool = True, hp_n_iter: int = 20) -> dict:
    """单模型训练：Trainer + per-fold 嵌入式筛选（原 cross_validate 行为）。

    Args:
        scheme_name: 特征方案名（结果目录/session 日志/TrainingResult.scheme 均用此名）。
    """
    logger.info(f"训练: {algo}, y∈[{y.min():.0f}°, {y.max():.0f}°]")

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

    bp = result.best_params or {}
    if bp:
        logger.info(f"{algo} 最佳参数: {bp}")
    # 全量重训最终模型并保存（供 api.predict 加载）；失败不影响训练结果
    try:
        save_trained_model(
            feature_set, bp, algo, scheme_name, config,
            feature_names=data.get("feature_names"),
        )
    except Exception as exc:  # noqa: BLE001 — 保存失败只告警，不阻断训练
        logger.warning(f"保存最终模型失败（不影响训练结果）: {exc}")
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
    - 适用于预选固定特征方案（v0.1.0 等）

    Args:
        trainer: Trainer 实现选择。None=原版 Trainer, "margin"=MarginTrainer。
        weight_components: 预设提供的权重组件（优先）；None 时按 weighting 策略名构建。
    """
    # 延迟导入避免 circular import
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
            out.append({
                "algo": r.model_name, "r": m["r"], "rmse": m["rmse"],
                "f1": m["f1"], "sens": m["sens"], "spec": m["spec"],
                "cm": cm,
                "preds": [float(x) for x in r.predictions],
                "best_params": r.best_params,
            })
    return out
