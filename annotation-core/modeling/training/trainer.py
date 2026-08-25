"""训练编排器 — 组合 DataSplitter + HPSearcher + 模型训练。

支持加权训练和 per-class 后处理校准。
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from modeling._shared import CLINICAL, inv_transform, transform_target
from modeling.contracts import DataSplitter, FeatureSet, HPSearcher, TrainingConfig, TrainingResult
from modeling.metrics import CLASS_RANGES, compute_4class_metrics, compute_metrics, SEVERITY_BINS
from modeling.models import REGISTRY as MODEL_REGISTRY
from modeling.training.data_splitters import KFoldSplitter, SPLITTERS
from modeling.training.feature_selector import PerFoldFeatureSelector
from modeling.training.hp_searchers import SEARCHERS
from modeling.training.result_paths import RESULTS_DIR
from utils.logger import logger

MAX_COBB = 90  # 预测值钳制上限（临床合理范围）
_MIN_CORR_SAMPLES = 2
"""计算相关系数所需的最少样本数。"""


class Trainer:
    """训练编排器 — 装配 DataSplitter + HPSearcher 完成多模型训练。

    Args:
        config: 训练配置（模型列表、切分策略、搜索策略等）。
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self, feature_set: FeatureSet) -> list[TrainingResult]:
        """训练配置中的所有模型。"""
        results: list[TrainingResult] = []
        for model_name in self.config.models:
            logger.info(f"开始训练模型: {model_name}")
            model = MODEL_REGISTRY[model_name]()
            result = self.train_one(model, feature_set)
            results.append(result)
            r_val = result.metrics.get("r", 0)
            rmse_val = result.metrics.get("rmse", 0)
            logger.info(f"模型 {model_name} 完成: r={r_val:.4f}, rmse={rmse_val:.4f}")
        return results

    def train_one(self, model: object, feature_set: FeatureSet) -> TrainingResult:
        """训练单个模型。"""
        model_name = getattr(model, "name", model.__class__.__name__)
        # archived/xxx 方案名含 "/"——替换为 "_" 避免日志路径断裂
        session_name = f"{feature_set.name.replace('/', '_')}_{model_name}"
        log_path = logger.begin_session(session_name)

        y = feature_set.y
        X = feature_set.X
        n = len(y)
        all_preds = np.zeros(n)
        all_counts = np.zeros(n)

        weight_components: list | None = getattr(self.config, "weight_components", None)
        search_weight_components = getattr(self.config, "search_weight_components", None) or weight_components
        sample_weight: NDArray | None = None
        if weight_components:
            sample_weight = np.ones(len(y))
            for wc in weight_components:
                sample_weight *= wc.compute(y)
            logger.info(f"样本加权: {len(weight_components)} 乘区 "
                        f"mean={sample_weight.mean():.3f} "
                        f"range=[{sample_weight.min():.3f}, {sample_weight.max():.3f}]")

        splitter = self._build_splitter()
        inner_splitter_name = getattr(self.config, "search_data_splitter", None) or "kfold"
        inner_cls = SPLITTERS.get(inner_splitter_name, KFoldSplitter)
        inner_splitter = inner_cls(n_splits=3, n_repeats=1, random_state=42)
        searcher = self._build_searcher()
        best_params: dict | None = None
        best_params_final: dict = {}
        fold_total = self.config.data_splitter_params.get("n_repeats", 5) \
                     * self.config.data_splitter_params.get("n_splits", 5)

        logger.info(f"开始训练: {fold_total}折, "
                    f"n_iter={self.config.hp_searcher_params.get('n_iter', 40)}, "
                    f"transform={self.config.transform_target}")

        fold_preds: list[tuple[NDArray, NDArray]] = []
        fold_times: list[float] = []
        fold_details: list[dict] = []

        selector = None
        if self.config.feature_selector == "per_fold":
            selector = PerFoldFeatureSelector()
        raw_blocks = getattr(feature_set, "X_raw_blocks", None)

        for fold_idx, (tr_idx, te_idx) in enumerate(splitter.split(y)):
            fold_start = time.time()
            if selector is not None and raw_blocks:
                raw_tr = {k: v[tr_idx] for k, v in raw_blocks.items()}
                raw_te = {k: v[te_idx] for k, v in raw_blocks.items()}
                X_tr = selector.fit_transform(raw_tr, y[tr_idx], feature_set.region_column_names)
                X_te = selector.transform(raw_te)
            else:
                X_tr, X_te = X[tr_idx], X[te_idx]
            n_tr, n_te = len(tr_idx), len(te_idx)
            y_tr = y[tr_idx]
            logger.info(f"=== Fold {fold_idx + 1}/{fold_total}: n_train={n_tr} n_test={n_te} ===")

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            yt_tr = transform_target(y_tr) if self.config.transform_target else y_tr.copy()

            if searcher is not None:
                best_model, best_params = searcher.search(
                    model, X_tr_s, yt_tr, inner_splitter,
                    n_iter=self.config.hp_searcher_params.get("n_iter", 40),
                    score_metric=self.config.hp_searcher_params.get("score_metric", "r2"),
                    hp_space_overrides=self.config.hp_space_overrides,
                    weight_components=search_weight_components,
                )
                best_params_final = best_params
                hp_str = ", ".join(f"{k}={v}" for k, v in best_params.items()) if best_params else "默认"
                logger.info(f"  Fold {fold_idx + 1}: 最佳 HP=[{hp_str}]")
            else:
                best_model = type(model)()

            if sample_weight is not None:
                w = sample_weight[tr_idx]
                best_model.external_weight = w
                logger.info(f"  Fold {fold_idx + 1}: 权重 mean={w.mean():.3f}")

            best_model.fit(X_tr_s, yt_tr)

            y_pred_raw = best_model.predict(X_te_s)
            y_pred = np.maximum(inv_transform(y_pred_raw) if self.config.transform_target else y_pred_raw, 0)
            y_true_te = y[te_idx]
            fold_rmse = float(np.sqrt(np.mean((y_pred - y_true_te) ** 2)))
            fold_r = float(np.corrcoef(y_pred, y_true_te)[0, 1]) if len(y_pred) > _MIN_CORR_SAMPLES else 0
            fold_elapsed = time.time() - fold_start
            logger.info(f"  Fold {fold_idx + 1} 结果: RMSE={fold_rmse:.2f} r={fold_r:.4f} ({fold_elapsed:.1f}s)")

            all_preds[te_idx] += y_pred
            all_counts[te_idx] += 1
            fold_preds.append((te_idx.copy(), y_pred.copy()))
            fold_times.append(fold_elapsed)

            fold_details.append({
                "fold": fold_idx + 1,
                "n_train": int(n_tr),
                "n_test": int(n_te),
                "best_params": best_params,
                "fold_rmse": round(fold_rmse, 4),
                "fold_r": round(fold_r, 4),
                "fold_time": round(fold_elapsed, 2),
            })

        # 平均预测
        final_preds = all_preds / np.maximum(all_counts, 1)
        final_preds = np.clip(final_preds, 0, MAX_COBB)

        # Per-class 校准（按配置：仅加权方案等显式启用时执行）
        cal_details = {}
        if fold_preds and self.config.calibrate:
            final_preds, cal_details = self._apply_calibration(final_preds, y, fold_preds)
            if cal_details.get("bias"):
                logger.info(f"per-class 校准: bias={{{', '.join(f'{k}={v:.2f}°' for k, v in cal_details.get('bias', {}).items())}}}")

        # 指标
        metrics = compute_metrics(y, final_preds, clinical=CLINICAL)
        m4 = compute_4class_metrics(y, final_preds)
        total_time = sum(fold_times)

        y_cls = np.digitize(y, SEVERITY_BINS[1:-1])
        p_cls = np.digitize(final_preds, SEVERITY_BINS[1:-1])
        cm = confusion_matrix(y_cls, p_cls).tolist()
        total_acc = int((y_cls == p_cls).sum())

        # 详细日志
        logger.info(f"{model_name} 完成: "
                    f"MF1={m4['macro_f1']:.4f}  RMSE={metrics.get('rmse',0):.2f}  "
                    f"r={metrics.get('r',0):.4f}  Acc={total_acc}/{n}  "
                    f"F1(20°)={metrics.get('f1',0):.3f}  "
                    f"Sens={metrics.get('sens',0):.3f}  Spec={metrics.get('spec',0):.3f}  "
                    f"总耗时={total_time:.1f}s")
        pc = m4.get("per_class", {})
        for c_name in ["Normal", "Mild", "Moderate", "Severe"]:
            p = pc.get(c_name, {})
            logger.info(f"  {c_name:>8s}: F1={p.get('f1',0):.3f} "
                        f"Recall={p.get('recall',0):.3f} Prec={p.get('precision',0):.3f} "
                        f"n={p.get('support',0):2d}")

        # 保存结果
        session_dir = self._save_session(feature_set, model_name, final_preds, y, m4, metrics, cm, fold_details)

        logger.end_session()

        details = {"n_models": len(self.config.models), "fold_details": fold_details}
        if cal_details:
            details["calibration"] = cal_details

        return TrainingResult(
            scheme=feature_set.name,
            model_name=model_name,
            predictions=final_preds,
            metrics=metrics,
            best_params=best_params_final,
            details=details,
            fold_details=fold_details,
            session_dir=session_dir,
            training_log=log_path,
        )

    def _save_session(self, feature_set: FeatureSet, model_name: str,
                      preds: NDArray, y_true: NDArray,
                      m4: dict, m1: dict, cm: list,
                      fold_details: list[dict]) -> str:
        """保存训练结果到磁盘（落盘内容与拆分前逐项一致）。"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        scheme_name = feature_set.name
        out_dir = RESULTS_DIR / scheme_name / f"{ts}_{model_name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        self._write_config_json(out_dir, scheme_name)
        _write_predictions_csv(out_dir, y_true, preds)
        self._write_metrics_json(out_dir, m4, m1, cm, y_true, preds)
        with open(out_dir / "fold_results.json", "w") as f:
            json.dump(fold_details, f, indent=2)

        logger.info(f"结果已保存: {out_dir}")
        return str(out_dir)

    def _write_config_json(self, out_dir: Path, scheme_name: str) -> None:
        """写 config.json（训练配置快照）。"""
        cfg_dict = {
            "models": self.config.models,
            "scheme": scheme_name,
            "hp_searcher": self.config.hp_searcher,
            "hp_n_iter": self.config.hp_searcher_params.get("n_iter", 40),
            "score_metric": self.config.hp_searcher_params.get("score_metric", "r2"),
            "trainer": self.config.trainer,
            "transform_target": self.config.transform_target,
            "data_splitter": self.config.data_splitter,
            "data_splitter_params": self.config.data_splitter_params,
            "search_data_splitter": self.config.search_data_splitter,
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(cfg_dict, f, indent=2)

    def _write_metrics_json(self, out_dir: Path, m4: dict, m1: dict, cm: list,
                            y_true: NDArray, preds: NDArray) -> None:
        """写 metrics_summary.json / per_class_metrics.json / confusion_matrix.json。"""
        y_cls = np.digitize(y_true, SEVERITY_BINS[1:-1])
        p_cls = np.digitize(preds, SEVERITY_BINS[1:-1])
        pc = m4.get("per_class", {})
        summary = {
            "macro_f1": round(m4["macro_f1"], 4),
            "rmse": round(m1.get("rmse", 0), 4),
            "r": round(m1.get("r", 0), 4),
            "total_acc": round((y_cls == p_cls).mean(), 4),
            "f1_20": round(m1.get("f1", 0), 4),
            "sens_20": round(m1.get("sens", 0), 4),
            "spec_20": round(m1.get("spec", 0), 4),
            "n_samples": int(len(y_true)),
        }
        with open(out_dir / "metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        pc_serializable = {}
        for c_name in ["Normal", "Mild", "Moderate", "Severe"]:
            p = pc.get(c_name, {})
            pc_serializable[c_name] = {k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in p.items()}
        with open(out_dir / "per_class_metrics.json", "w") as f:
            json.dump(pc_serializable, f, indent=2)

        with open(out_dir / "confusion_matrix.json", "w") as f:
            json.dump({"matrix": cm, "labels": ["Normal", "Mild", "Moderate", "Severe"]}, f, indent=2)


    def _apply_calibration(
        self,
        preds: NDArray,
        y: NDArray,
        fold_preds: list[tuple[NDArray, NDArray]],
    ) -> tuple[NDArray, dict]:
        """Per-class 偏差校正。

        从每折的验証集上统计每个类的平均偏差（y_pred - y_true），
        跨折平均后，对最终预测做类内校正 + 边界保护。

        Args:
            preds:     当前的平均预测 (N,)。
            y:         真实值 (N,)。
            fold_preds: 每折的 (test_idx, y_pred) 列表。

        Returns:
            (校正后的预测, 校正信息 dict)。
        """
        # 收集每折每类的偏差
        class_biases: dict[int, list[float]] = {c: [] for c in range(4)}
        for te_idx, y_pred_te in fold_preds:
            y_true_te = y[te_idx]
            for c in range(4):
                lo, hi = SEVERITY_BINS[c], SEVERITY_BINS[c + 1]
                m = (y_true_te >= lo) & (y_true_te < hi)
                if m.sum() > 0:
                    class_biases[c].append(float(np.mean(y_pred_te[m] - y_true_te[m])))

        avg_bias = {}
        for c in range(4):
            vals = class_biases[c]
            avg_bias[c] = float(np.mean(vals)) if vals else 0.0

        # 应用校正（向量化：逐类偏差 + 边界钳制，逐位一致）
        pred_class = np.digitize(preds, SEVERITY_BINS[1:-1])
        bias_vec = np.array([avg_bias[pc] for pc in pred_class])
        lo_vec = np.array([CLASS_RANGES[pc][0] for pc in pred_class])
        hi_vec = np.array([CLASS_RANGES[pc][1] for pc in pred_class])
        corrected = np.clip(preds - bias_vec, lo_vec, hi_vec)

        return corrected, {"bias": avg_bias}

    def _build_splitter(self) -> DataSplitter:
        """根据配置构建 DataSplitter 实例。"""
        splitter_cls = SPLITTERS.get(self.config.data_splitter)
        if splitter_cls is None:
            msg = f"未知 data_splitter: {self.config.data_splitter}，可选: {list(SPLITTERS.keys())}"
            raise ValueError(msg)
        return splitter_cls(**self.config.data_splitter_params)

    def _build_searcher(self) -> HPSearcher | None:
        """根据配置构建 HPSearcher 实例。None 表示跳过搜索。"""
        hp_name = self.config.hp_searcher
        if hp_name in SEARCHERS:
            return SEARCHERS[hp_name]()
        if hp_name != "none":
            logger.warning(f"未知 hp_searcher: {hp_name}，跳过超参搜索")
        return None


def _write_predictions_csv(out_dir: Path, y_true: NDArray, preds: NDArray) -> None:
    """写 predictions.csv（逐样本真实/预测值与类别标签）。"""
    y_cls = np.digitize(y_true, SEVERITY_BINS[1:-1])
    p_cls = np.digitize(preds, SEVERITY_BINS[1:-1])
    labels = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
    df = pd.DataFrame({
        "true_cobb": y_true,
        "pred_cobb": np.round(preds, 2),
        "true_class": [labels.get(c, "?") for c in y_cls],
        "pred_class": [labels.get(c, "?") for c in p_cls],
    })
    df.to_csv(out_dir / "predictions.csv", index=False)
