"""特征合成器 — CI 公式特征 / ci10-ci20 单目标 / 不对称指数的拟合与复现。

统一「训练时拟合参数」与「预测时应用参数」为成对 fit/transform 的合成器类，
消除 save_model（拟合端）与 prediction（应用端）对同一逻辑的逐字复制：

- `fit`：用训练集 region 特征拟合合成参数（与训练特征方案同口径）
- `transform`：对任意 subject 特征用保存参数合成（预测单行 / 论文批量通用）
- `to_params` / `from_params`：参数序列化（模型包存 dict，预测时构造合成器）

模型包格式不变（dict 参数 + from_params 现构造），不迁移既有 joblib。
alpha 按调用方参数化（scheme 用 CI_ALPHA、预测复现用 0.5），保持各自历史口径。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

from features.selectors._utils import _anova_groups, _dedup_by_r, _pass_r_or_anova
from utils.constants import SEVERITY_BINS

# 特征筛选超参（与训练特征方案同口径）
_COEF_EPS = 1e-6        # Lasso/Logistic 系数视为非零的绝对值下限
_CORR_DEDUP = 0.85      # ci10/ci20 目标筛选去高相关阈值
CI_R = 0.2              # CI 组内 |r| 保留阈值
CI_ANOVA_P = 0.05       # CI ANOVA p 阈值（None 时仅 |r| 单条件）
CI_GROUP_CORR = 0.85    # CI 组内去高相关阈值

# 5 不对称指数中 AI 的 OLS 组合权重（论文表3 口径，训练/预测共用）
_AI_WEIGHTS = {
    "curvature_index": 1.18684,
    "height_index": 0.32729,
    "nai": -0.08249,
    "ri": 0.19200,
}


def _search(
    Xs: np.ndarray,
    y: np.ndarray,
    col_names: list[str],
    n: int = 10,
) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    """Lasso 选特征 + RidgeCV 拟合，返回 (全部列名, 选中索引, 系数, 截距)。

    供 AsymmetrySynthesizer 与 commands.export.tables 复用（5 指数公式拟合原语）。
    """
    lasso = LassoCV(cv=5, max_iter=10000, random_state=42, n_jobs=1).fit(Xs, y)
    sel = np.where(np.abs(lasso.coef_) > _COEF_EPS)[0]
    if len(sel) == 0:
        sel = np.argsort(np.abs([pearsonr(Xs[:, i], y)[0] for i in range(Xs.shape[1])]))[-n:]
    if len(sel) > n:
        sel = sel[np.argsort(np.abs(lasso.coef_[sel]))[::-1][:n]]
    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5).fit(Xs[:, sel], y)
    return col_names, sel, ridge.coef_, ridge.intercept_


def eval_linear_formula(
    feature_df: pd.DataFrame,
    cols: list[str],
    coefs: list[float] | np.ndarray,
    intercept: float = 0.0,
) -> np.ndarray:
    """特征线性组合求值：cols × coefs + intercept（缺列抛错）。

    统一 prediction._linear_combo（AI8/Ridge 公式）与 modeling.ensemble.build_ai_feature
    （v0.1.0 AI 公式）的同一运算，避免两种 schema（cols/coefs vs feats/coefs/intercept）
    各自实现一遍。
    """
    coef_arr = np.asarray(coefs, dtype=np.float64)
    missing = [c for c in cols if c not in feature_df.columns]
    if missing:
        raise ValueError(f"线性公式缺特征 {len(missing)} 个: {missing[:3]}...（模型与特征提取不匹配）")
    return feature_df[cols].values.astype(np.float64) @ coef_arr + intercept


class CiFormulaSynthesizer:
    """CI 公式特征合成器 — 全 subject 标准化 + Lasso 拟合的线性组合。

    训练特征方案（scheme `_build_one_ci`）与预测复现（save_model→prediction）共用。
    fit 对每组 (measure, method) 做 条件筛选 → 去高相关 → Lasso → 保存
    选列/均值/标准差/系数；transform 用保存参数对任意特征行合成 CI 值。

    alpha 按调用方参数化（scheme 传 CI_ALPHA、预测复现用 0.5 缺省），
    保证与各自历史口径一致，不统一。
    """

    def __init__(
        self,
        groups: list[tuple[str, str]],
        alpha_map: dict[str, float] | None = None,
        anova_p: float | None = CI_ANOVA_P,
    ) -> None:
        self._groups = groups
        self._alpha_map = alpha_map or {}
        self._anova_p = anova_p
        self._params: dict[str, dict] = {}

    def fit(self, df_r: pd.DataFrame, y: np.ndarray) -> CiFormulaSynthesizer:
        """用训练集 region 特征拟合每组 CI 的合成参数。"""
        region_cols = [c for c in df_r.columns if c not in ("subject_id", "max_cobb")]
        Xr = df_r[region_cols].values.astype(float)
        y4 = np.digitize(y, SEVERITY_BINS[1:-1]) if self._anova_p is not None else None
        out: dict[str, dict] = {}
        for measure, method in self._groups:
            suffix = f"_{measure}__pw" if method == "pw" else f"_{measure}"
            cols = [c for c in region_cols if c.endswith(suffix)]
            if not cols:
                continue
            X_all = Xr[:, [region_cols.index(c) for c in cols]]
            keep = [
                i for i in range(len(cols))
                if _pass_r_or_anova(
                    abs(pearsonr(X_all[:, i], y)[0]),
                    _anova_groups(X_all[:, i], y4) if y4 is not None else [],
                    CI_R,
                    self._anova_p,
                )
            ]
            if not keep:
                continue
            keep_cols = _dedup_by_r(X_all[:, keep], y, [cols[i] for i in keep], CI_GROUP_CORR)
            if not keep_cols:
                continue
            keep_idx = [cols.index(c) for c in keep_cols]
            Xs = StandardScaler().fit_transform(X_all[:, keep_idx])
            alpha = self._alpha_map.get(f"{measure}_{method}", 0.5)
            lasso = Lasso(alpha=alpha, max_iter=200000, random_state=42).fit(Xs, y)
            nz = np.where(np.abs(lasso.coef_) > _COEF_EPS)[0]
            if not len(nz):
                continue
            nz_cols = [keep_cols[i] for i in nz]
            Xnz = Xr[:, [region_cols.index(c) for c in nz_cols]]
            out[f"{measure}_{method}"] = {
                "columns": nz_cols,
                "mean": np.nanmean(Xnz, axis=0),
                "std": np.maximum(np.nanstd(Xnz, axis=0), 1e-8),
                "coef": lasso.coef_[nz],
            }
        self._params = out
        return self

    def transform(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """用保存参数合成 CI 列（缺列按 full_cols 索引 zeros 回填保维度）。"""
        result: dict[str, np.ndarray] = {}
        for group, p in self._params.items():
            if not p.get("columns"):
                continue  # 空参数组（无合成特征）跳过
            full_cols = list(p["columns"])
            cols = [c for c in full_cols if c in feature_df.columns]
            X = feature_df[cols].values.astype(np.float64) if cols else np.zeros((len(feature_df), 0))
            if len(cols) < len(full_cols):
                full = np.zeros((len(feature_df), len(full_cols)))
                if cols:
                    full[:, [full_cols.index(c) for c in cols]] = X
                X = full
            result[group] = ((X - np.asarray(p["mean"])) / np.asarray(p["std"])) @ np.asarray(p["coef"])
        return pd.DataFrame(result, index=feature_df.index)

    def to_params(self) -> dict:
        """返回可 JSON 落盘的参数（模型包 ci_formula_params 格式）。"""
        return self._params

    @classmethod
    def from_params(cls, params: dict) -> CiFormulaSynthesizer:
        instance = cls([])
        instance._params = params
        return instance


class CiTargetSynthesizer:
    """单目标 CI 特征（ci10_normal/ci20_mild）合成器 — Pearson 筛选 + Logistic 拟合。

    拟合逻辑复现 _loaders_ci::_build_ci_for_target / save_model::_fit_ci_target；
    transform 支持 DataFrame（预测，缺列对齐）与 numpy（训练特征方案，全列）两种输入。
    """

    def __init__(self) -> None:
        self._params: dict = {}

    def fit(
        self,
        region_cols: list[str],
        Xr: np.ndarray,
        target: np.ndarray,
        C: float,
        thr: float,
        corr_threshold: float = _CORR_DEDUP,
    ) -> CiTargetSynthesizer:
        """Pearson 筛选 → 去高相关 → 标准化 → Logistic → 非零系数，保存参数。

        Args:
            corr_threshold: 去高相关阈值。v0.1.0 用 0.85（save_model），
                manual 用 0.95（ensemble_boundary），按训练方案参数化。
        """
        rv = np.array([abs(pearsonr(Xr[:, i], target)[0]) for i in range(len(region_cols))])
        keep = np.where(rv > thr)[0]
        order = np.argsort(-rv[keep])
        corr = np.abs(np.corrcoef(Xr[:, keep].T))
        dd = [order[0]]
        for idx in order[1:]:
            if not any(corr[idx, j] > corr_threshold for j in dd):
                dd.append(idx)
        keep2 = [keep[i] for i in sorted(dd)]
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xr[:, keep2])
        lr = LogisticRegression(
            C=C, l1_ratio=0.95, solver="saga", max_iter=10000,
            class_weight="balanced", random_state=42,
        )
        lr.fit(Xs, target)
        nz = np.where(np.abs(lr.coef_[0]) > _COEF_EPS)[0]
        self._params = {
            "columns": [region_cols[i] for i in keep2],
            "nz": nz.tolist(),
            "coef": lr.coef_[0][nz].tolist(),
            "scaler": scaler,
        }
        return self

    def transform(self, feature_df: pd.DataFrame) -> np.ndarray:
        """用保存参数对 DataFrame 特征合成单列 CI（缺列按 full_cols 对齐）。"""
        full_cols = list(self._params["columns"])
        cols = [c for c in full_cols if c in feature_df.columns]
        X = feature_df[cols].values.astype(np.float64) if cols else np.zeros((len(feature_df), 0))
        if len(cols) < len(full_cols):
            full = np.zeros((len(feature_df), len(full_cols)))
            if cols:
                full[:, [full_cols.index(c) for c in cols]] = X
            X = full
        return self.transform_ndarray(X, full_cols)

    def transform_ndarray(self, X: np.ndarray, columns: list[str]) -> np.ndarray:
        """numpy 输入版（训练特征方案用）：X 为 (N, len(columns)) 全量特征矩阵。"""
        params = self._params
        idx = [columns.index(c) for c in params["columns"]]
        Xs = params["scaler"].transform(X[:, idx])
        nz = np.array(params["nz"], dtype=int)
        coef = np.array(params["coef"], dtype=float)
        return Xs[:, nz] @ coef

    def to_params(self) -> dict:
        """返回可序列化参数（模型包 ci10_params/ci20_params 格式）。"""
        return self._params

    @classmethod
    def from_params(cls, params: dict) -> CiTargetSynthesizer:
        instance = cls()
        instance._params = params
        return instance


class AsymmetrySynthesizer:
    """5 不对称指数合成器 — Lasso+Ridge 拟合公式 + scaler，训练/预测共用。

    fit 用全量 region 特征拟合 4 指数（curvature/height/nai/ri）公式 + AI OLS 权重
    （同 tables._compute_indices 口径）；transform 用保存参数对任意特征行合成指数。
    论文表3 批量与 prediction 单行共用同一实现。
    """

    def __init__(self) -> None:
        self._params: dict = {}

    def fit(self, df_2700: pd.DataFrame) -> AsymmetrySynthesizer:
        cols = [c for c in df_2700.columns if c not in ("subject_id", "max_cobb")]
        X = df_2700[cols].values.astype(float)
        y = df_2700["max_cobb"].values.astype(float)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        # 4 指数各自的候选特征掩码（按列名语义分组，论文表3 口径）
        configs = [
            (
                "curvature_index",
                np.array([
                    "mean_curv" in c or "gauss_curv" in c or "roughness" in c
                    or ("normal_angle" in c and "normal_vector" not in c)
                    for c in cols
                ]),
                9,
            ),
            ("height_index", np.array([c.endswith("_height") for c in cols]), 8),
            ("nai", np.array(["normal_angle" in c and "normal_vector" not in c for c in cols]), 8),
            ("ri", np.array(["roughness" in c for c in cols]), 8),
        ]
        formulas: dict = {}
        for fname, mask, n in configs:
            sub_c = [c for c, m in zip(cols, mask, strict=False) if m]
            sub_Xs = Xs[:, mask]
            if sub_Xs.shape[1] < n:
                sub_Xs, sub_c = Xs, cols
            names, sel, coefs, intercept = _search(sub_Xs, y, sub_c, n)
            formulas[fname] = {
                "feats": [names[i] for i in sel],
                "coefs": coefs.tolist(),
                "intercept": float(intercept),
            }
        self._params = {
            "asymmetry_scaler": scaler,
            "asymmetry_cols": cols,
            "asymmetry_formulas": formulas,
            "asymmetry_ai": dict(_AI_WEIGHTS),
        }
        return self

    def transform(self, feature_df: pd.DataFrame) -> dict[str, np.ndarray]:
        """用保存参数合成 5 指数（键 curvature_index/height_index/nai/ri/ai）。"""
        cols = self._params["asymmetry_cols"]
        missing = [c for c in cols if c not in feature_df.columns]
        if missing:
            raise ValueError(f"指数计算缺少 region 特征 {len(missing)} 个: {missing[:3]}...")
        X = feature_df[cols].values.astype(np.float64)
        Xs = self._params["asymmetry_scaler"].transform(X)
        formulas = self._params["asymmetry_formulas"]
        out: dict[str, np.ndarray] = {}
        for fname, formula in formulas.items():
            idx = [cols.index(feat) for feat in formula["feats"]]
            out[fname] = Xs[:, idx] @ np.array(formula["coefs"])
        ai_coefs = self._params["asymmetry_ai"]
        out["ai"] = sum(ai_coefs[key] * out[key] for key in ai_coefs)
        return out

    def to_params(self) -> dict:
        """返回模型包 asymmetry_* 段参数（scaler/cols/formulas/ai）。"""
        return self._params

    @classmethod
    def from_params(cls, params: dict) -> AsymmetrySynthesizer:
        instance = cls()
        instance._params = params
        return instance
