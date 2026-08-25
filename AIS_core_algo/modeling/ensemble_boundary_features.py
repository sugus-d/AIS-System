"""v1.0.0 边界 Ensemble 特征层 — AI 特征构建、per-class blend、边界钳制、分类器。

从 ensemble_boundary 拆出（等价重构）：纯特征/预测变换逻辑（无文件落盘），
供训练编排（ensemble_boundary.train_ensemble_manual）与模型包落盘
（ensemble_boundary_artifacts.save_boundary_model）复用。

突破配置常量（OOF 调优后固定）也归本模块——它们是特征构建/钳制的参数。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from features.selectors.scheme_morph_region_ci_35d import _dedup_by_r
from utils.constants import SEVERITY_BINS

# ── 突破配置（OOF 调优后固定） ──
_ALPHA = 0.48  # CompositeV7 权重
_THR20 = 0.5  # P(y>20) 上钳阈值
_THR10 = 0.4  # P(y>10) 下钳阈值
_TGT20 = 20.5  # 20° 上钳目标值
_TGT10 = 9.5  # 10° 下钳目标值
_C_CLF = 0.5  # 边界分类器正则
_N_AI_FEATS = 8  # refit-AI Lasso 特征数
_R_SCREEN = 0.1  # refit-AI 粗筛 |r|
_R_DEDUP = 0.85  # refit-AI 去高相关阈值
_COEF_EPS = 1e-6  # Lasso/Logistic 系数非零下限
_N_SPLITS = 5  # OOF 折数
# cobb 分级边界（单源自 utils.constants.SEVERITY_BINS）
_COBB_BOUNDARIES = tuple(SEVERITY_BINS[1:4])  # (10.0, 20.0, 40.0)
_COBB_10, _COBB_20, _COBB_40 = _COBB_BOUNDARIES
_BETA = 0.6  # per-class α blend 权重（剩余 1-β 给 Ridge-AI）
_PERCLASS_ALPHA = (0.8, 0.7, 0.48, 0.5)  # 4 预测类的 CompositeV7 权重（NF/Mild/Mod/Severe）
_RIDGE_ALPHAS = np.logspace(-1, 3, 30)  # RidgeCV 正则网格
_RANDOM_STATE = 42

_REGION_CSV = "results/extraction/features_extraction/v1.0.0/region_asymmetry.csv"


def build_refit_ai_feature(sids: list[str], y: np.ndarray, region_csv: str = _REGION_CSV) -> tuple[np.ndarray, dict]:
    """从 region 特征拟合 refit-AI 线性组合（Lasso 选 8 特征）。

    与探索阶段一致的筛选：|r|>0.1 → dedup 0.85 → LassoCV → 按 |coef|×|r| 取 top-8。

    Args:
        sids: subject 顺序（对齐 CompositeV7 预测行序）。
        y: 真实 Cobb 角（与 sids 对齐）。
        region_csv: v1.0.0 region 特征 CSV。

    Returns:
        (ai_feature, params)：AI 特征向量 + 拟合参数（cols/coefs）。
    """
    dfr = pd.read_csv(region_csv).dropna(subset=["max_cobb"])
    dfr = dfr[dfr["subject_id"].isin(sids)].set_index("subject_id").loc[sids].reset_index()
    rc = [c for c in dfr.columns if c not in ("subject_id", "max_cobb")]
    Xr = dfr[rc].values.astype(float)
    keep = [i for i in range(Xr.shape[1]) if abs(pearsonr(Xr[:, i], y)[0]) > _R_SCREEN]
    kc = _dedup_by_r(Xr[:, keep], y, [rc[i] for i in keep], _R_DEDUP)
    ki = [rc.index(c) for c in kc]
    Xf = Xr[:, ki]
    lcv = LassoCV(cv=5, max_iter=100000, random_state=_RANDOM_STATE, n_jobs=1)
    lcv.fit(StandardScaler().fit_transform(Xf), y)
    nz = np.where(np.abs(lcv.coef_) > _COEF_EPS)[0]
    rv = np.array([abs(pearsonr(Xf[:, i], y)[0]) for i in nz])
    order = nz[np.argsort(-np.abs(lcv.coef_[nz]) * rv)][:_N_AI_FEATS]
    ai = Xf[:, order] @ lcv.coef_[order]
    # 列均值随公式保存：瀑布图线性分量基线需 mean(combo)（见 predict._render_waterfall）
    params = {
        "cols": [kc[i] for i in order],
        "coefs": lcv.coef_[order].tolist(),
        "mean": {kc[i]: float(Xf[:, i].mean()) for i in order},
    }
    return ai, params


def build_ridge_ai_feature(sids: list[str], y: np.ndarray, region_csv: str = _REGION_CSV) -> tuple[np.ndarray, dict]:
    """从 region 特征拟合 Ridge-AI 线性组合（RidgeCV 全量筛选特征）。

    与 :func:`build_refit_ai_feature` 同口径粗筛（|r|>0.1 → dedup 0.85），但不强制稀疏——
    RidgeCV 保留全部筛选特征并收缩系数，捕捉 Lasso-8 稀疏丢弃的弱信号
    （r=0.869 / MAE=4.37 vs Lasso-8 的 0.831 / 5.03），是本轮 MAE 突破的关键分量。

    Args:
        sids: subject 顺序（对齐 CompositeV7 预测行序）。
        y: 真实 Cobb 角（与 sids 对齐）。
        region_csv: v1.0.0 region 特征 CSV。

    Returns:
        (ridge_ai, params)：Ridge-AI 特征向量 + 拟合参数（cols/coefs）。
    """
    dfr = pd.read_csv(region_csv).dropna(subset=["max_cobb"])
    dfr = dfr[dfr["subject_id"].isin(sids)].set_index("subject_id").loc[sids].reset_index()
    rc = [c for c in dfr.columns if c not in ("subject_id", "max_cobb")]
    Xr = dfr[rc].values.astype(float)
    keep = [i for i in range(Xr.shape[1]) if abs(pearsonr(Xr[:, i], y)[0]) > _R_SCREEN]
    kc = _dedup_by_r(Xr[:, keep], y, [rc[i] for i in keep], _R_DEDUP)
    ki = [rc.index(c) for c in kc]
    Xf = Xr[:, ki]
    rcv = RidgeCV(alphas=_RIDGE_ALPHAS).fit(StandardScaler().fit_transform(Xf), y)
    ridge_ai = Xf @ rcv.coef_
    # 列均值随公式保存：瀑布图线性分量基线需 mean(combo)（见 predict._render_waterfall）
    params = {
        "cols": kc,
        "coefs": rcv.coef_.tolist(),
        "mean": {col: float(Xf[:, i].mean()) for i, col in enumerate(kc)},
    }
    return ridge_ai, params


def boundary_oof_probs(
    X: np.ndarray, y: np.ndarray, thresholds: tuple[int, int] = (10, 20), n_splits: int = _N_SPLITS
) -> dict[str, np.ndarray]:
    """边界分类器 KFold OOF 概率。

    Args:
        X: 30D + refit-AI 特征矩阵（与 y 同序）。
        y: 真实 Cobb 角。
        thresholds: 边界阈值（10°=Normal/Mild，20°=Mild/Moderate）。

    Returns:
        {"p10": (N,), "p20": (N,)}：P(y>thr) 折外概率。
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=_RANDOM_STATE)
    probs: dict[str, np.ndarray] = {}
    for thr in thresholds:
        yb = (y > thr).astype(int)
        P = np.zeros(len(y))
        for tr_idx, te_idx in kf.split(X):
            scaler = StandardScaler()
            Xs_tr = scaler.fit_transform(X[tr_idx])
            Xs_te = scaler.transform(X[te_idx])
            clf = LogisticRegression(C=_C_CLF, max_iter=5000).fit(Xs_tr, yb[tr_idx])
            P[te_idx] = clf.predict_proba(Xs_te)[:, 1]
        probs[f"p{thr}"] = P
    return probs


def apply_perclass_blend(
    composite_pred: np.ndarray,
    ai8_pred: np.ndarray,
    ridge_pred: np.ndarray,
    perclass_alpha: tuple[float, float, float, float] = _PERCLASS_ALPHA,
    beta: float = _BETA,
    alpha_base: float = _ALPHA,
) -> np.ndarray:
    """per-class α blend + Ridge-AI 加权（最终 Ensemble 的 blend 步骤）。

    1. b0 = alpha_base×composite + (1-alpha_base)×ai8 —— 分类依据
    2. 按 b0 预测类查 perclass_alpha（两端多信 composite，中间 Moderate 保持 α=0.48）
    3. pbase = α×composite + (1-α)×ai8
    4. final = β×pbase + (1-β)×ridge —— Ridge-AI 负责压 MAE

    Args:
        composite_pred: CompositeV7 OOF 预测。
        ai8_pred: Lasso-8 refit-AI LR OOF 预测。
        ridge_pred: Ridge-AI LR OOF 预测。
        perclass_alpha: 4 预测类的 CompositeV7 权重（Normal/Mild/Moderate/Severe）。
        beta: per-class α blend 的权重（剩余 1-β 给 Ridge-AI）。
        alpha_base: b0 分类用固定 α（Lasso-8 blend）。

    Returns:
        blend 预测（未钳制）。
    """
    b0 = alpha_base * np.asarray(composite_pred) + (1.0 - alpha_base) * np.asarray(ai8_pred)
    pc = np.digitize(b0, _COBB_BOUNDARIES)
    alphas = np.asarray(perclass_alpha)[pc]
    pbase = alphas * np.asarray(composite_pred) + (1.0 - alphas) * np.asarray(ai8_pred)
    return beta * pbase + (1.0 - beta) * np.asarray(ridge_pred)


def apply_boundary_clamps(
    preds: np.ndarray,
    p10: np.ndarray,
    p20: np.ndarray,
    thr20: float = _THR20,
    thr10: float = _THR10,
    tgt20: float = _TGT20,
    tgt10: float = _TGT10,
) -> np.ndarray:
    """边界钳制（20° 上钳 + 10° 下钳）。

    Args:
        preds: 已 blend 的预测（如 :func:`apply_perclass_blend` 输出）。
        p10/p20: P(y>10)/P(y>20) 折外概率。
        thr20/thr10: 上钳/下钳概率阈值。
        tgt20/tgt10: 上钳/下钳目标值。

    Returns:
        钳制后的最终预测（钳到 [0,90]）。
    """
    preds = np.clip(np.asarray(preds), 0, 90).astype(float).copy()
    up = (preds < _COBB_20) & (np.asarray(p20) > thr20)
    preds[up] = np.maximum(preds[up], tgt20)
    down = (preds >= _COBB_10) & (preds < _COBB_20) & (np.asarray(p10) < thr10)
    preds[down] = np.minimum(preds[down], tgt10)
    return preds


def fit_boundary_classifiers(X: np.ndarray, y: np.ndarray) -> dict:
    """在全量数据上训练 P(y>10)/P(y>20) 边界分类器（预测单 subject 用）。

    Returns:
        {"p10": {"scaler": ..., "clf": ...}, "p20": {...}}
    """
    classifiers: dict = {}
    for thr in (10, 20):
        yb = (y > thr).astype(int)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(C=_C_CLF, max_iter=5000).fit(Xs, yb)
        classifiers[f"p{thr}"] = {"scaler": scaler, "clf": clf}
    return classifiers
