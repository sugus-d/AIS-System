#!/usr/bin/env python3
"""生成论文所需的数据表 CSVs（表 1/3/4/5/6）。

用法:
    uv run python -m commands.export.tables
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import f_oneway, pearsonr

from modeling.metrics import CLINICAL, compute_metrics, SEVERITY_LABELS
from utils.constants import classify_cobb, SEVERITY_BINS
from utils.logger import logger
from utils.paths import (
    CLINICAL_DATA,
    ENSEMBLE_PRED_PATH,
    EVAL_TABLES_DIR,
    FEATURES_DIR,
)

# Lasso 系数视为零的阈值
_COEF_EPS = 1e-6
# Pearson 相关最少样本数
_MIN_PEARSON_SAMPLES = 3
# ANOVA 最少分组数
_MIN_ANOVA_GROUPS = 2
# p 值显示阈值：小于此值改用科学计数法
_P_VALUE_TINY = 0.0001


def _fmt_mean_std(arr: np.ndarray) -> str:
    return f"{arr.mean():.1f}±{arr.std():.1f}"


def _fmt_median_iqr(arr: np.ndarray) -> str:
    """中位数与四分位距（IQR=Q3−Q1），用于非正态连续变量描述。"""
    q1, q3 = np.percentile(arr, [25, 75])
    return f"{np.median(arr):.1f}±{q3 - q1:.1f}"


def _fmt_mae_ci(err: np.ndarray, confidence: float = 0.95) -> str:
    mean = err.mean()
    se = err.std(ddof=1) / math.sqrt(len(err))
    ci = se * sp_stats.t.ppf((1 + confidence) / 2, len(err) - 1)
    return f"{mean:.1f} [{mean-ci:.1f},{mean+ci:.1f}]"


def _fmt_rmse_ci(err: np.ndarray, confidence: float = 0.95) -> str:
    sq = err ** 2
    mean = np.sqrt(sq.mean())
    rmses = np.array([np.sqrt(np.mean(np.random.choice(sq, len(sq), replace=True)))
                      for _ in range(2000)])
    lo = np.percentile(rmses, (1 - confidence) / 2 * 100)
    hi = np.percentile(rmses, (1 + confidence) / 2 * 100)
    return f"{mean:.1f} [{lo:.1f},{hi:.1f}]"


def _load_curves() -> dict[str, str]:
    with open(CLINICAL_DATA) as f:
        data = json.load(f)
    result: dict[str, str] = {}
    for sid, cd in data.items():
        curves = cd.get("curves", [])
        if not curves:
            result[sid] = "None"
            continue
        max_cb, primary = -1, None
        for c in curves:
            cb = c.get("cobb") or 0
            if cb > max_cb:
                max_cb, primary = cb, c
        if primary is None or not primary.get("level"):
            result[sid] = "None"
            continue
        level = primary["level"]
        has_t, has_l = "T" in level, "L" in level
        result[sid] = ("Double" if (has_t and has_l)
                       else "Thoracic" if has_t else "Lumbar" if has_l else "Other")
    return result


def _load_clinical() -> pd.DataFrame:
    exported = {d.name for d in Path("data/ground_truth").iterdir() if d.is_dir()}
    with open(CLINICAL_DATA) as f:
        data = json.load(f)
    rows = []
    for sid, cd in data.items():
        if sid not in exported:
            continue
        rows.append({
            "subject_id": sid,
            "gender": cd.get("gender", ""),
            "height_cm": cd.get("height_cm"),
            "weight_kg": cd.get("weight_kg"),
            "max_cobb": cd.get("max_cobb"),
        })
    df = pd.DataFrame(rows)
    df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)
    return df


def _compute_indices(df_2700: pd.DataFrame) -> pd.DataFrame:
    """5 不对称指数表（复用 features.synthesis.AsymmetrySynthesizer 同口径拟合）。

    Curvature/Height/NAI/RI 由 Lasso→Ridge 选特征拟合 + AI OLS 组合（论文表3）；
    与 save_model/ensemble_boundary 拟合端、prediction 应用端共用同一合成器。
    """
    from features.synthesis import AsymmetrySynthesizer

    values = AsymmetrySynthesizer().fit(df_2700).transform(df_2700)
    y = df_2700["max_cobb"].values.astype(float)
    rows = []
    for i, sid in enumerate(df_2700["subject_id"]):
        row = {"subject_id": sid, "y": y[i], "severity": classify_cobb(y[i])}
        for fname in ["curvature_index", "height_index", "nai", "ri"]:
            row[fname] = float(values[fname][i])
        row["ai"] = float(values["ai"][i])
        rows.append(row)
    return pd.DataFrame(rows)


# ── Tables ──


def table1(df_clinical: pd.DataFrame, curve_types: dict, out_dir: Path = EVAL_TABLES_DIR) -> pd.DataFrame:
    df = df_clinical.copy()
    df["ais"] = df["max_cobb"].apply(lambda x: "AIS" if x is not None and x >= SEVERITY_BINS[1] else "Non-AIS")
    df["curve_type"] = df["subject_id"].map(curve_types)
    df["severity"] = df["max_cobb"].apply(lambda x: classify_cobb(x) if x is not None else "None")
    df = df[df["subject_id"].str.startswith(("1", "2"))]

    groups = []
    for gender in ["Male", "Female"]:
        sub = df[df["gender"] == gender]
        for ais in ["Non-AIS", "AIS"]:
            g = sub[sub["ais"] == ais]
            groups.append({
                "gender": gender, "ais": ais, "n": len(g), "age": "-",
                "height": _fmt_median_iqr(g["height_cm"].dropna().values) if len(g) else "-",
                "weight": _fmt_median_iqr(g["weight_kg"].dropna().values) if len(g) else "-",
                "bmi": _fmt_median_iqr(g["bmi"].dropna().values) if len(g) else "-",
                "max_cobb": _fmt_median_iqr(g["max_cobb"].dropna().values) if len(g) else "-",
            })
    for ais in ["Non-AIS", "AIS"]:
        g = df[df["ais"] == ais]
        groups.append({
            "gender": "Total", "ais": ais, "n": len(g), "age": "-",
            "height": _fmt_median_iqr(g["height_cm"].dropna().values) if len(g) else "-",
            "weight": _fmt_median_iqr(g["weight_kg"].dropna().values) if len(g) else "-",
            "bmi": _fmt_median_iqr(g["bmi"].dropna().values) if len(g) else "-",
            "max_cobb": _fmt_median_iqr(g["max_cobb"].dropna().values) if len(g) else "-",
        })
    result = pd.DataFrame(groups)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "table1_demographics.csv", index=False)
    logger.info(f"Table 1: {out_dir / 'table1_demographics.csv'}")
    return result


def table3(df_idx: pd.DataFrame, out_dir: Path = EVAL_TABLES_DIR) -> pd.DataFrame:
    groups = []
    for sev in SEVERITY_LABELS:
        sub = df_idx[df_idx["severity"] == sev]
        for idx_name in ["ai", "curvature_index", "height_index", "nai", "ri"]:
            valid = sub[["y", idx_name]].dropna()
            vals = valid[idx_name].values
            groups.append({
                "severity": sev, "index": idx_name.capitalize().replace("_", " "),
                "n": len(sub), "value": _fmt_median_iqr(vals) if len(vals) else "-",
                "r": round(pearsonr(vals, valid["y"].values)[0], 3) if len(vals) >= _MIN_PEARSON_SAMPLES else "-",
            })
    for idx_name in ["ai", "curvature_index", "height_index", "nai", "ri"]:
        valid = df_idx[["y", idx_name]].dropna()
        vals = valid[idx_name].values
        groups.append({
            "severity": "Whole cohort", "index": idx_name.capitalize().replace("_", " "),
            "n": len(df_idx), "value": _fmt_median_iqr(vals) if len(vals) else "-",
            "r": round(pearsonr(vals, valid["y"].values)[0], 3) if len(vals) >= _MIN_PEARSON_SAMPLES else "-",
        })
    for idx_name in ["ai", "curvature_index", "height_index", "nai", "ri"]:
        gb_sev = []
        for sev in SEVERITY_LABELS:
            sub = df_idx[df_idx["severity"] == sev]
            vals = sub[idx_name].dropna().values
            if len(vals) > 1:
                gb_sev.append(vals)
        p_val = f_oneway(*gb_sev)[1] if len(gb_sev) >= _MIN_ANOVA_GROUPS else 1.0
        groups.append({
            "severity": "P value", "index": idx_name.capitalize().replace("_", " "),
            "n": "", "value": f"{p_val:.4f}" if p_val > _P_VALUE_TINY else f"{p_val:.2e}", "r": "",
        })
    result = pd.DataFrame(groups)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "table3_indices_by_severity.csv", index=False)
    logger.info(f"Table 3: {out_dir / 'table3_indices_by_severity.csv'}")
    return result


def table4(df_idx: pd.DataFrame, out_dir: Path = EVAL_TABLES_DIR) -> pd.DataFrame:
    groups = []
    for sev in ["All"] + SEVERITY_LABELS:
        sub = df_idx if sev == "All" else df_idx[df_idx["severity"] == sev]
        for idx_name in ["ai", "curvature_index", "height_index", "nai", "ri"]:
            valid = sub[["y", idx_name]].dropna()
            vals = valid[idx_name].values
            y_sub = valid["y"].values
            if len(vals) < _MIN_PEARSON_SAMPLES:
                groups.append({"severity": sev, "index": idx_name.capitalize().replace("_", " "), "r": "-", "p": "-"})
                continue
            r, p = pearsonr(vals, y_sub)
            groups.append({"severity": sev, "index": idx_name.capitalize().replace("_", " "),
                           "r": round(r, 4), "p": f"{p:.4f}" if p > _P_VALUE_TINY else f"{p:.2e}"})
    result = pd.DataFrame(groups)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "table4_correlation.csv", index=False)
    logger.info(f"Table 4: {out_dir / 'table4_correlation.csv'}")
    return result


def table5(y_true: np.ndarray, y_pred: np.ndarray, subject_ids: list[str], out_dir: Path = EVAL_TABLES_DIR) -> pd.DataFrame:
    df = pd.DataFrame({"subject_id": subject_ids, "y_true": y_true, "y_pred": y_pred})
    df["severity"] = df["y_true"].apply(classify_cobb)
    curve_types = _load_curves()
    df["curve_type"] = df["subject_id"].map(curve_types)

    def _add_group(name: str, sub_df: pd.DataFrame) -> None:
        if len(sub_df) == 0:
            return
        yt, yp = sub_df["y_true"].values, sub_df["y_pred"].values
        se = yp - yt
        entry = {"subgroup": name, "n": len(sub_df),
                 "gt_mean": f"{yt.mean():.1f}", "gbdt_mean": f"{yp.mean():.1f}",
                 "mae_ci": _fmt_mae_ci(np.abs(se)) if len(se) > 1 else f"{np.abs(se).mean():.1f}",
                 "rmse_ci": _fmt_rmse_ci(se) if len(se) > 1 else f"{np.sqrt((se**2).mean()):.1f}"}
        if len(sub_df) >= _MIN_PEARSON_SAMPLES:
            r_val, _ = pearsonr(yp, yt)
            entry["r2"] = round(r_val**2, 2)
            entry["r"] = round(r_val, 2)
        else:
            entry["r2"] = ""
            entry["r"] = ""
        groups.append(entry)

    groups = []
    _add_group("Overall", df)
    for sev in SEVERITY_LABELS:
        _add_group(sev, df[df["severity"] == sev])
    for ct in ["Thoracic", "Lumbar", "Double"]:
        _add_group(ct, df[df["curve_type"] == ct])
    result = pd.DataFrame(groups)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "table5_prediction.csv", index=False)
    logger.info(f"Table 5: {out_dir / 'table5_prediction.csv'}")
    return result


def table6(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path = EVAL_TABLES_DIR) -> pd.DataFrame:
    true_labels = np.array([SEVERITY_LABELS.index(classify_cobb(v)) for v in y_true])
    pred_labels = np.array([SEVERITY_LABELS.index(classify_cobb(v)) for v in y_pred])

    groups = []
    for i, label in enumerate(SEVERITY_LABELS):
        tp = ((pred_labels == i) & (true_labels == i)).sum()
        fp = ((pred_labels == i) & (true_labels != i)).sum()
        fn = ((pred_labels != i) & (true_labels == i)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        groups.append({"severity": label, "precision": round(precision, 3),
                       "recall": round(recall, 3), "f1": round(f1, 3)})

    accuracy = (pred_labels == true_labels).mean()
    groups.append({"severity": "Overall", "precision": round(accuracy, 3),
                   "recall": round(accuracy, 3), "f1": round(accuracy, 3)})

    total_true = max(len(y_true), 1)
    w_prec = w_rec = w_f1 = 0.0
    for i in range(len(SEVERITY_LABELS)):
        n_true = (true_labels == i).sum()
        tp = ((pred_labels == i) & (true_labels == i)).sum()
        fp = ((pred_labels == i) & (true_labels != i)).sum()
        fn = ((pred_labels != i) & (true_labels == i)).sum()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1_c = 2 * p * r / (p + r) if (p + r) else 0.0
        w_prec += p * n_true / total_true
        w_rec += r * n_true / total_true
        w_f1 += f1_c * n_true / total_true
    groups.append({"severity": "Weighted Average", "precision": round(w_prec, 3),
                   "recall": round(w_rec, 3), "f1": round(w_f1, 3)})

    result = pd.DataFrame(groups)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "table6_classification.csv", index=False)
    logger.info(f"Table 6: {out_dir / 'table6_classification.csv'}")
    return result


def main(
    pred_csv: Path | None = None,
    region_csv: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    """论文数据表（表 1/3/4/5/6），数据源与输出目录可切换（v0.1.0/manual）。

    Args:
        pred_csv: 预测 CSV；None 时用 v0.1.0 ``ENSEMBLE_PRED_PATH``。
        region_csv: 2700 维 region 特征 CSV；None 时用 v0.1.0 ``features_2700d.csv``。
        out_dir: 输出目录；None 时用 ``EVAL_TABLES_DIR``。
    """
    pred_path = Path(pred_csv) if pred_csv else ENSEMBLE_PRED_PATH
    region_path = Path(region_csv) if region_csv else (FEATURES_DIR / "features_2700d.csv")
    target = Path(out_dir) if out_dir else EVAL_TABLES_DIR
    target.mkdir(parents=True, exist_ok=True)

    # Load predictions
    pred_df = pd.read_csv(pred_path)
    y_pred = pred_df["max_cobb_pred"].values.astype(float)
    y = pred_df["max_cobb_true"].values.astype(float)
    sid_list = pred_df["subject_id"].tolist()
    subject_ids = set(sid_list)
    logger.info(f"加载预测: {pred_path} ({len(subject_ids)} subjects)")

    # Load clinical data + curves
    df_clinical = _load_clinical()
    df_clinical = df_clinical[df_clinical["subject_id"].isin(subject_ids)]
    curve_types = _load_curves()
    df_2700 = pd.read_csv(region_path).dropna(subset=["max_cobb"])
    df_2700 = df_2700[df_2700["subject_id"].isin(subject_ids)]

    # Compute indices
    df_idx = _compute_indices(df_2700)

    table1(df_clinical, curve_types, target)
    table3(df_idx, target)
    table4(df_idx, target)

    m = compute_metrics(y, y_pred, threshold=CLINICAL)
    logger.info(f"  Binary: F1={m['f1']:.3f} Sens={m['sens']:.3f} Spec={m['spec']:.3f} RMSE={m['rmse']:.2f}")

    table5(y, y_pred, sid_list, target)
    table6(y, y_pred, target)
    logger.info("\n全部表格已生成。")


if __name__ == "__main__":
    main()
