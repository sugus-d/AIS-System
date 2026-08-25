#!/usr/bin/env python3
"""Compute feature importance (v0.1.0/v1.0.0 scheme): Permutation Importance → CI decomposition → aggregation.

Usage:
    uv run python -m commands.export.analyze
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402  # 需在 warnings 过滤后导入
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from features.utils.ci_decompose import (  # noqa: E402
    _assign_group,
    aggregate_by_measurement,
    aggregate_by_region,
    decompose_ci_importance,
    load_ci_formulas_from_package,
)
from modeling.model_package import load_model_package  # noqa: E402
from modeling.models import _oversample, REGISTRY  # noqa: E402
from utils.logger import logger  # noqa: E402
from utils.paths import EXPORT_DIR, FEATURE_IMPORTANCE_DIR  # noqa: E402

# 特征名按 "_" 切分后的最小/完整段数（如 "nr_normal_angle__pw"）
_NAME_PARTS_MIN = 3
_NAME_PARTS_FULL = 4

# ── 数据加载（scheme 分支：beta / production） ──

_MANUAL_MODEL_PKG = Path("results/modeling/models/v1.0.0/boundary_ensemble_ridge.joblib")
_MANUAL_SCHEME = "v1.0.0"
_BACK_MODEL_PKG = Path("results/modeling/models/v0.1.0/CompositeV7.joblib")


def _load_manual() -> tuple[np.ndarray, np.ndarray, list[str], dict, object, StandardScaler]:
    """production 方案：scheme 30D + 模型包（模型/scaler/CI 公式，展平在顶层）。"""
    from features.selectors.schemes import SELECTION_REGISTRY

    scheme_data = SELECTION_REGISTRY[_MANUAL_SCHEME].load()
    X = np.asarray(scheme_data["X_basic"], dtype=np.float64)
    y = np.asarray(scheme_data["y"], dtype=np.float64)
    feature_names = scheme_data["feature_names"]
    model_pkg = load_model_package(str(_MANUAL_MODEL_PKG))
    ci_formulas = load_ci_formulas_from_package(model_pkg)
    return X, y, feature_names, ci_formulas, model_pkg["model"], model_pkg["scaler"]


def _load_back() -> tuple[np.ndarray, np.ndarray, list[str], dict, object, StandardScaler]:
    """beta 方案（v0.1.0）：数据 + 现场训练 HistGBRT + CI 公式（训练时落盘，运行时读取）。"""
    from features.selectors.schemes import _load_dual_ci

    data = _load_dual_ci()
    y = np.asarray(data["y"], dtype=np.float64)
    X = np.asarray(data["X_basic"], dtype=np.float64)
    feature_names = data.get("feature_names", [f"f{i}" for i in range(X.shape[1])])
    ci_formulas = load_ci_formulas_from_package(load_model_package(str(_BACK_MODEL_PKG)))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = REGISTRY["HistGBRT"]()
    w = model._build_weight(y)  # noqa: SLF001  # modeling.models 训练辅助函数
    X_a, y_a = _oversample(Xs, y, w)
    model._reg.fit(X_a, y_a)  # noqa: SLF001  # HistGBRT 内部 sklearn 回归器
    logger.info(f"  Trained: {model._reg.n_iter_} trees")  # noqa: SLF001
    return X, y, feature_names, ci_formulas, model, scaler


# ── 核心流程（beta/production 共用） ──


def _run_analysis(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    ci_formulas: dict,
    model: object,
    scaler: StandardScaler,
    out_dir: Path,
) -> None:
    """核心流程：Permutation Importance → CI 反解 → 聚合 → 3 CSV。

    beta 与 production 完全同口径（同一函数、同一参数），仅数据源/模型/CI 公式不同。
    CI 公式统一由 features.utils.ci_decompose 从模型包读取（训练时落盘）。

    Args:
        X: 原始（未标准化）特征矩阵。
        y: 真实 Cobb 角。
        feature_names: 特征名（与 X 列序对齐）。
        ci_formulas: 统一 CI 公式 {ci_name: {"columns": [...], "coefs": [...]}}。
        model: HistGBRT 包装（有 _reg）。
        scaler: 已 fit 的 StandardScaler。
        out_dir: 输出目录（3 个 CSV）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    Xs = scaler.transform(X)

    ci_names = [n for n in feature_names if n in ci_formulas]
    ci_set = set(ci_names)
    logger.info(f"  CI features: {ci_names}")

    logger.info("\nComputing Permutation Importance (30 repeats)...")
    result = permutation_importance(
        model._reg, Xs, y, n_repeats=30, random_state=42, n_jobs=1, scoring="neg_mean_absolute_error"  # noqa: SLF001
    )
    importance = result.importances_mean

    # Build importance dict
    feat_imp = dict(zip(feature_names, importance, strict=True))

    # Separate CI vs non-CI features
    direct_imp: dict[str, float] = {}
    ci_imp: dict[str, float] = {}
    for name, imp in feat_imp.items():
        if name in ci_set:
            ci_imp[name] = imp
        else:
            direct_imp[name] = imp
    logger.info(f"  Direct features: {len(direct_imp)}, CI features: {len(ci_imp)}")
    for name, imp in sorted(ci_imp.items(), key=lambda x: -abs(x[1])):
        logger.info(f"    {name}: {imp:.4f}")

    logger.info("\nDecomposing CI features...")
    decomposed = decompose_ci_importance(ci_imp, ci_formulas)
    logger.info(f"  Decomposed to {len(decomposed)} region features")

    # Merge direct + decomposed
    all_imp = dict(direct_imp)
    for name, imp in decomposed.items():
        all_imp[name] = all_imp.get(name, 0.0) + imp

    # Sort by importance
    sorted_imp = sorted(all_imp.items(), key=lambda x: -abs(x[1]))
    total_imp = sum(abs(v) for v in all_imp.values())

    # ── Output 1: per-feature ranking ──
    rows = []
    for rank, (name, imp) in enumerate(sorted_imp, 1):
        share = imp / total_imp * 100 if total_imp > 0 else 0
        g = _assign_group(name)
        # subgroup: for region features, parse region + measure
        subgroup = ""
        if g not in ("Morph", "Clinical", "Normal Angle", "Curvature", "Roughness", "Height"):
            # Try to extract region+measure from the name
            parts = name.split("_")
            if len(parts) >= _NAME_PARTS_MIN and parts[0] in ("nr", "st", "sp", "ax", "wa", "wl"):
                prefix = parts[0]
                if len(parts) >= _NAME_PARTS_FULL and parts[1] in ("nr", "st", "sp", "ax", "wa", "wl"):
                    prefix = f"{parts[0]}_{parts[1]}"
                # Get measure from name
                for m_pat, m_name in [
                    ("normal_angle", "normal_angle"),
                    ("normal_vector_cos", "normal_vector_cos"),
                    ("gauss_curv", "gauss_curv"),
                    ("mean_curv", "mean_curv"),
                    ("roughness", "roughness"),
                    ("_height", "height"),
                ]:
                    if m_pat in name:
                        subgroup = f"{prefix} + {m_name}"
                        break
        rows.append(
            {
                "rank": rank,
                "feature": name,
                "importance": imp,
                "share_pct": round(share, 1),
                "group": g,
                "subgroup": subgroup,
            }
        )
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "feature_importance_decomposed.csv", index=False)
    logger.info(f"\n  Saved: feature_importance_decomposed.csv ({len(df_out)} features)")

    # ── Output 2: by measurement type ──
    by_measure = aggregate_by_measurement(all_imp)
    total_m = sum(by_measure.values())
    mt_rows = []
    for g in ["Normal Angle", "Curvature", "Roughness", "Height", "Morph", "Clinical"]:
        imp = by_measure.get(g, 0.0)
        share = imp / total_m * 100 if total_m > 0 else 0.0
        mt_rows.append({"new_group": g, "importance": round(imp, 4), "share_pct": f"{share:.1f}"})
    pd.DataFrame(mt_rows).to_csv(out_dir / "importance_by_group.csv", index=False)
    logger.info(f"  Saved: importance_by_group.csv ({len(mt_rows)} groups)")

    # ── Output 3: by anatomical region ──
    by_region = aggregate_by_region(all_imp)
    total_r = sum(by_region.values())
    uv_rows = []
    for band in ["Shoulder", "Scapula", "Axilla", "Waist", "Pelvis", "Morph", "Clinical"]:
        imp = by_region.get(band, 0.0)
        share = imp / total_r * 100 if total_r > 0 else 0.0
        uv_rows.append({"horizontal_band": band, "importance": round(imp, 4), "share_pct": f"{share:.1f}"})
    pd.DataFrame(uv_rows).to_csv(out_dir / "importance_by_horizontal_band.csv", index=False)
    logger.info(f"  Saved: importance_by_horizontal_band.csv ({len(uv_rows)} bands)")

    logger.info(f"\nDone!  Total importance: {total_imp:.4f}")


# ── Main ──


def main(scheme: str = "v0.1.0", out_dir: Path | None = None) -> None:
    """特征重要性分析入口（scheme 切换，非替代）。

    Args:
        scheme: 版本号（``v0.1.0``/``v1.0.0``）或精简 alias（``beta``/``production``）；
            v0.1.0=算法 ROI（默认），v1.0.0=人工 ROI（用模型包模型 + 训练落盘 CI 公式）。
        out_dir: 输出目录；None 时 v0.1.0→``FEATURE_IMPORTANCE_DIR``，
            v1.0.0→``EXPORT_DIR/v1.0.0/feature_importance``。
    """
    from features.selectors import get_selector

    version = get_selector(scheme).version
    if version == "v1.0.0":
        X, y, feature_names, ci_formulas, model, scaler = _load_manual()
        target_dir = out_dir or (EXPORT_DIR / "v1.0.0" / "feature_importance")
    elif version == "v0.1.0":
        X, y, feature_names, ci_formulas, model, scaler = _load_back()
        target_dir = out_dir or FEATURE_IMPORTANCE_DIR
    else:
        raise ValueError(f"未知 scheme: {scheme}，可选 v0.1.0/beta / v1.0.0/production")
    logger.info("=" * 60)
    logger.info(f"Loading {scheme} scheme ({len(feature_names)}D)...")
    _run_analysis(X, y, feature_names, ci_formulas, model, scaler, target_dir)


if __name__ == "__main__":
    main()
