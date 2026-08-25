"""v1.0.0 边界钳制 Ensemble — 突破 0.701 顶的集成训练编排。

第一轮（边界钳制）: blend 系统性低估 Moderate（16/78 被压到 Mild），用 P(y>20)
边界分类器识别低估并上钳，MF1 0.7010→0.7282。但 MAE 4.68 差目标 0.15°。

第二轮（per-class α + Ridge-AI，本轮）: 双目标达标 MF1=0.7364 / MAE=4.38。
两条互补路径：
1. Ridge-AI 分量：Lasso-8 强制稀疏丢弱信号，RidgeCV 全量 267 筛选特征线性组合
   r=0.869 / MAE=4.37（vs Lasso-8 r=0.831 / MAE=5.03），是好回归器。
2. per-class α：两端预测类（Normal/Severe）多信 CompositeV7（其尾部 bias 远小于
   refit-AI：Normal +2.0 vs +5.25，Severe -2.15 vs -6.88），中间 Moderate 保持 α=0.48。

最终 Ensemble 四步：
1. Lasso-8 blend b0 = 0.48×CompositeV7 + 0.52×refit-AI（分类依据）
2. per-class α: 按 b0 预测类查 α（0.8/0.7/0.48/0.5）→ pbase = α×C7 + (1-α)×refit-AI
3. Ridge-AI 加权: final = 0.6×pbase + 0.4×Ridge-AI-LR（OOF r=0.869）
4. 边界钳制（20° 上钳 + 10° 下钳，特征 = 30D + refit-AI + Ridge-AI）

钳制阈值 / per-class α / β 在完整 OOF 上选择（与 v0.1.0 α=0.6 同口径），
30×62 分裂 held-out 验证乐观度极低（held-out MF1≈0.729 / MAE≈4.44），固定后不再搜索。

模块分层（2026-08 拆分，等价重构）：
  - ensemble_boundary_features:     特征构建（refit-AI/Ridge-AI/blend/钳制/分类器）
  - ensemble_boundary_artifacts:    自包含 composite 分量 + 模型包/JSON 落盘
  - ensemble_boundary（本模块）:   训练编排（train_ensemble_manual）+ CLI
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.selectors.schemes import SELECTION_REGISTRY
from modeling.contracts import TrainingConfig
from modeling.ensemble import fit_ai_linear_oof
from modeling.ensemble_boundary_artifacts import save_boundary_model
from modeling.ensemble_boundary_features import (
    _ALPHA,
    _BETA,
    _PERCLASS_ALPHA,
    apply_boundary_clamps,
    apply_perclass_blend,
    boundary_oof_probs,
    build_refit_ai_feature,
    build_ridge_ai_feature,
)
from modeling.metrics import compute_4class_metrics
from modeling.training.result_paths import save_results
from utils.logger import logger

_SCHEME = "v1.0.0"
_COMPOSITE_PRED = (
    "results/modeling/prediction/v1.0.0/composite_v7-wc3-calmargin-hp100/HistGBRT/predictions.csv"
)


def train_ensemble_manual(
    scheme_name: str = _SCHEME,
    alpha: float = _ALPHA,
    beta: float = _BETA,
    label: str = "ai_refit_ridge_boundary",
    composite_pred_csv: str = _COMPOSITE_PRED,
    out_scheme: str = _SCHEME,
) -> Path:
    """端到端：CompositeV7 OOF → refit-AI/Ridge-AI OOF → per-class α+β → 钳制 → 落盘。

    Args:
        scheme_name: 30D 特征方案名（取 X_basic 与 AI 特征拼接做边界分类器特征）。
        alpha: b0 分类用 CompositeV7 权重（Lasso-8 blend）。
        beta: per-class α blend 权重（剩余给 Ridge-AI）。
        label: 落盘签名后缀。
        composite_pred_csv: 复用 CompositeV7 OOF 预测 CSV。
        out_scheme: 落盘特征方案目录。

    Returns:
        落盘的 metrics.json 路径。
    """
    # 1. CompositeV7 OOF
    c7_df = pd.read_csv(composite_pred_csv)
    sids = c7_df["subject_id"].tolist()
    y = c7_df["max_cobb_true"].values.astype(float)
    c7_pred = c7_df["max_cobb_pred"].values.astype(float)

    # 2. Lasso-8 refit-AI 特征 + LR OOF（与 CompositeV7 subject 顺序对齐）
    ai8, ai8_params = build_refit_ai_feature(sids, y)
    ai8_pred = fit_ai_linear_oof(ai8, y)

    # 3. Ridge-AI 特征 + LR OOF
    ai_ridge, ridge_params = build_ridge_ai_feature(sids, y)
    ridge_pred = fit_ai_linear_oof(ai_ridge, y)

    # 4. per-class α blend + Ridge-AI 加权
    blend_preds = apply_perclass_blend(c7_pred, ai8_pred, ridge_pred, beta=beta)

    # 5. 边界分类器 OOF（特征 = 30D + refit-AI + Ridge-AI）
    scheme_data = SELECTION_REGISTRY[scheme_name].load()
    X30 = np.asarray(scheme_data["X_basic"], dtype=np.float64)
    if len(X30) != len(y):
        raise ValueError(f"30D 特征行数 {len(X30)} 与 CompositeV7 预测 {len(y)} 不一致")
    X_boundary = np.column_stack([X30, ai8, ai_ridge])
    probs = boundary_oof_probs(X_boundary, y)

    # 6. 边界钳制
    final_preds = apply_boundary_clamps(blend_preds, probs["p10"], probs["p20"])

    # 7. 指标 + 落盘（save_results 格式，独立目录）
    metrics = compute_4class_metrics(y, final_preds)
    mae = float(np.abs(final_preds - y).mean())
    rmse = float(np.sqrt(np.mean((final_preds - y) ** 2)))
    r_val = float(np.corrcoef(final_preds, y)[0, 1])
    logger.info(
        f"边界 Ensemble(ridge): MF1={metrics['macro_f1']:.4f} MAE={mae:.2f}° "
        f"RMSE={rmse:.2f} r={r_val:.4f} (β={beta}, 目标 MF1≥0.724 MAE≤4.53°)"
    )

    cfg = TrainingConfig(models=["Ensemble"], hp_searcher_params={"n_iter": 5}, trainer=None, transform_target=False)
    save_results("Ensemble", final_preds, cfg, "ensemble", label, y, sids, feat_scheme_name=out_scheme)

    # 8. 保存单 subject 预测模型包（refit-AI + Ridge-AI + 边界分类器 + 配置）
    save_boundary_model(
        sids,
        y,
        ai8,
        ai8_params,
        ai_ridge,
        ridge_params,
        X_boundary,
        alpha,
        beta,
        _PERCLASS_ALPHA,
        Path("results/modeling/models") / out_scheme,
    )
    return Path("results/modeling/prediction") / out_scheme / f"ensemble-{label}" / "Ensemble" / "metrics.json"


def main() -> None:
    """CLI：落盘 v1.0.0 per-class α + Ridge-AI 边界钳制 Ensemble。"""
    import argparse

    parser = argparse.ArgumentParser(description="v1.0.0 per-class α + Ridge-AI 边界 Ensemble")
    parser.add_argument("--alpha", type=float, default=_ALPHA, help="b0 分类用 CompositeV7 权重")
    parser.add_argument("--beta", type=float, default=_BETA, help="per-class α blend 权重")
    parser.add_argument("--label", type=str, default="ai_refit_ridge_boundary", help="落盘签名后缀")
    args = parser.parse_args()
    path = train_ensemble_manual(alpha=args.alpha, beta=args.beta, label=args.label)
    logger.info(f"边界 Ensemble(ridge) 已落盘: {path}")


if __name__ == "__main__":
    main()
