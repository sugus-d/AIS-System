"""Ensemble AI 分量 — AI 特征构建、AI-LR 折外预测、线性拟合。

manuscript（2026-07-14 导出）最佳集成模型 ``Ensemble = α·CompositeV7 + (1-α)·AI-LR``
（α=0.6，OOF MF1=0.724 / MAE=4.53°）。本模块只保留 AI 分量层：
  - build_ai_feature:   AI 特征（region 线性组合 + 截距）
  - fit_ai_linear_oof:  AI-LR 折外预测（评估口径）
  - _fit_ai_linear:     AI-LR 全量拟合（单 subject 预测口径，参数随模型包保存）

训练闭环在 ensemble_train（reproduce_manuscript_ensemble / train_ensemble），
模型保存协议在 ensemble_save（save_composite_model）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.logger import logger
from utils.paths import FEATURE_FILE, FORMULA_DIR, MODELING_PREDICTION_DIR

# 历史 composite_v7 OOF 预测（manuscript ensemble 的主分量）
_COMPOSITE_V7_PRED = str(
    MODELING_PREDICTION_DIR
    / "v0.1.0"
    / "composite_v7-wc3-calmargin-hp100-composite_v7_stability"
    / "HistGBRT"
    / "predictions.csv"
)
# 训练时代 region 特征（含 AI 公式所需全部列）
_FEATURES_2700D = str(FEATURE_FILE)
# AI 特征公式（9 个 region 特征线性组合，r=0.856）
_AI_FORMULA = str(FORMULA_DIR / "archive" / "ai_formula.json")
# ensemble 落盘用特征方案名（与 CompositeV7 同方案）
_SCHEME = "v0.1.0"


def build_ai_feature(feature_df: pd.DataFrame, formula: dict) -> np.ndarray:
    """用公式生成 AI 特征（region 特征线性组合 + 截距）。

    复用 features.synthesis.eval_linear_formula（与 prediction._linear_combo 单点）。

    Args:
        feature_df: 含 region 特征列与 max_cobb 的 DataFrame。
        formula: ai_formula.json 结构（feats/coefs/intercept）。
    """
    from features.synthesis import eval_linear_formula

    return eval_linear_formula(feature_df, formula["feats"], formula["coefs"], float(formula["intercept"]))


def fit_ai_linear_oof(
    ai: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """AI 单特征线性回归的折外（OOF）预测。

    LR 在每折训练集拟合 y~AI，对测试折预测；与 manuscript 的 AI 分量口径一致
    （折外，避免数据泄漏）。

    Args:
        ai: (N,) AI 特征值。
        y: (N,) 真实 Cobb 角。
        n_splits: KFold 折数。
        random_state: 折划分随机种子。

    Returns:
        (N,) AI-LR 折外预测。
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y))
    for tr_idx, te_idx in kf.split(ai):
        model = LinearRegression().fit(ai[tr_idx].reshape(-1, 1), y[tr_idx])
        oof[te_idx] = model.predict(ai[te_idx].reshape(-1, 1))
    return oof


def build_ensemble_preds(
    primary_pred: np.ndarray,
    ai_pred: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """加权集成：``alpha·primary + (1-alpha)·ai``。"""
    return alpha * np.asarray(primary_pred) + (1.0 - alpha) * np.asarray(ai_pred)


def _load_ai_feature(
    sids: list[str],
    region_csv: str = _FEATURES_2700D,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """按 subject 顺序加载 AI 特征与真实 Cobb 角。

    Args:
        sids: 目标 subject 顺序（对齐 CompositeV7 预测行序）。
        region_csv: region 特征 CSV（默认 v0.1.0 的 features_2700d；
                    人工 ROI 重训时传 v1.0.0/region_asymmetry.csv）。

    Returns:
        (ai, y, subjects)：AI 特征、真实 Cobb、对齐后的 subject 列表。
    """
    formula = json.loads(Path(_AI_FORMULA).read_text(encoding="utf-8"))
    feats = formula["feats"]
    # 只读所需列（subject_id/max_cobb + AI 公式特征），避免整表碎片化
    feature_df = pd.read_csv(
        region_csv,
        usecols=lambda col: col in ("subject_id", "max_cobb") or col in feats,
    )
    feature_df = feature_df.dropna(subset=["max_cobb"])
    feature_df = feature_df[feature_df["subject_id"].isin(sids)]
    feature_df = feature_df.set_index("subject_id").loc[sids].reset_index()
    ai = build_ai_feature(feature_df, formula)
    return ai, feature_df["max_cobb"].values.astype(float), feature_df["subject_id"].tolist()


def _fit_ai_linear(sids: list[str], formula: dict, region_csv: str = _FEATURES_2700D) -> dict:
    """AI-LR 全量拟合：按 subject 顺序加载 region 特征 → AI → LinearRegression(y~AI)。

    单 subject 预测对单行 AI 用 ``intercept + coef×AI`` 求 AI_pred（spec 4.2 口径），
    与批量 OOF 的 :func:`fit_ai_linear_oof` 不同（后者每折拟合，仅用于评估）。
    region 特征从训练时代 CSV（默认 v0.1.0 的 ``_FEATURES_2700D``；人工 ROI 重训时
    传 v1.0.0/region_asymmetry.csv），SELECTION_REGISTRY 的 ``X_region_full`` 是
    None 占位（方案加载器不填充 region 块）。
    """
    from sklearn.linear_model import LinearRegression

    feats = formula["feats"]
    df = pd.read_csv(
        region_csv,
        usecols=lambda col: col in ("subject_id", "max_cobb") or col in feats,
    )
    df = df.dropna(subset=["max_cobb"])
    df = df[df["subject_id"].isin(sids)].set_index("subject_id").loc[sids].reset_index()
    ai = build_ai_feature(df, formula)
    y = df["max_cobb"].values.astype(float)
    lr = LinearRegression().fit(ai.reshape(-1, 1), y)
    # 列均值随包保存：瀑布图线性分量基线需 mean(combo)=Σcoef_j×mean(col_j)（SHAP expected 口径）
    formula = dict(formula)
    formula["mean"] = {feat: float(df[feat].mean()) for feat in feats}
    return {
        "ai_formula": formula,
        "ai_lr_intercept": float(lr.intercept_),
        "ai_lr_coef": float(lr.coef_[0]),
    }


def main() -> None:
    """CLI：重建 manuscript ensemble（默认阶段 1 轻量，--train 走闭环）。"""
    import argparse

    from modeling.ensemble_train import reproduce_manuscript_ensemble, train_ensemble

    parser = argparse.ArgumentParser(description="重建 manuscript ensemble（0.6×CompositeV7 + 0.4×AI-LR）")
    parser.add_argument("--alpha", type=float, default=0.6, help="CompositeV7 权重（默认 0.6）")
    parser.add_argument("--train", action="store_true", help="走完整闭环（训练/复用 CompositeV7）")
    parser.add_argument("--hp-n-iter", type=int, default=5, help="composite_v7 HP 搜索次数（--train 时生效）")
    args = parser.parse_args()

    if args.train:
        path = train_ensemble(alpha=args.alpha, hp_n_iter=args.hp_n_iter)
    else:
        path = reproduce_manuscript_ensemble(alpha=args.alpha)
    logger.info(f"ensemble 已落盘: {path}")


if __name__ == "__main__":
    main()
