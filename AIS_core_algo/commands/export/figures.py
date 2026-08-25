#!/usr/bin/env python3
"""编排层：论文关键图 — 散点图、Bland-Altman、混淆矩阵等。

用法:
    uv run python -m commands.export.figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from utils.logger import logger
from utils.paths import ENSEMBLE_PRED_PATH, EXPORT_FIGURES_DIR
from visualization._style import ACADEMIC_STYLE
from visualization.paper_figures_panels import (
    render_bland_altman,
    render_cm_annotation,
    render_confusion_matrix,
    render_scatter,
    render_subject_comparison,
)

# ── Typography (academic-figure-skill baseline) ──
mpl.rcParams.update(ACADEMIC_STYLE)
MM = 1 / 25.4

SEVERITY_BOUNDS = [10, 20, 40]
SEVERITY_LABELS = ["Normal", "Mild", "Moderate", "Severe"]
SEV3_BOUNDS = [20, 40]
SEV3_LABELS = ["0-20°", "20-40°", "40+°"]


def _save_fig(fig, name: str, out_dir: Path | None = None) -> None:
    """Save PNG + PDF for a figure（输出目录可切换）。"""
    target = Path(out_dir) if out_dir else EXPORT_FIGURES_DIR
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / f"{name}.png", dpi=300)
    fig.savefig(target / f"{name}.pdf", dpi=300)
    logger.info(f"  Saved: {name}.png + .pdf")
    plt.close(fig)


def _regression_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    r_val, p_val = stats.pearsonr(y_true, y_pred)
    slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
    return {"r": r_val, "p": p_val, "r2": r_val ** 2, "slope": slope, "intercept": intercept,
            "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2)))}


def _confusion_4class(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, list[str]]:
    tc_true = np.digitize(y_true, SEVERITY_BOUNDS)
    tc_pred = np.digitize(y_pred, SEVERITY_BOUNDS)
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(tc_true, tc_pred, strict=False):
        cm[t, p] += 1
    return cm, SEVERITY_LABELS


def _confusion_3class(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, list[str]]:
    tc_true = np.digitize(y_true, SEV3_BOUNDS)
    tc_pred = np.digitize(y_pred, SEV3_BOUNDS)
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(tc_true, tc_pred, strict=False):
        cm[t, p] += 1
    return cm, SEV3_LABELS


def _cm_stats_lines(cm: np.ndarray, labels: list[str], short_names: dict | None = None) -> str:
    total = cm.sum()
    correct = sum(cm[i, i] for i in range(len(labels)))
    acc = correct / total
    macro_f1 = 0
    lines = f"Accuracy: {acc:.1%}  ({correct}/{total})\n"
    for i, lbl in enumerate(labels):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        p_ = tp / (tp + fp) if (tp + fp) else 0
        r_ = tp / (tp + fn) if (tp + fn) else 0
        f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0
        macro_f1 += f1_
        short = short_names[lbl] if short_names else lbl
        lines += f"{short}: F1={f1_:.3f}  Prec={p_:.3f}  Rec={r_:.3f}\n"
    lines += f"Macro F1: {macro_f1 / len(labels):.3f}"
    return lines


def fig1_scatter(y_true, y_pred, class_true=None, class_pred=None, out_dir: Path | None = None):
    stats_d = _regression_stats(y_true, y_pred)
    correct = None if class_true is None else (class_true == class_pred)
    fig, ax = plt.subplots(figsize=(183 * MM, 165 * MM))
    render_scatter(ax, y_true, y_pred, correct, stats_d,
                   SEVERITY_BOUNDS, SEVERITY_LABELS)
    _save_fig(fig, "散点图_真值vs预测", out_dir)


def fig1_scatter_3class(y_true, y_pred, out_dir: Path | None = None):
    class_true = np.digitize(y_true, SEV3_BOUNDS)
    class_pred = np.digitize(y_pred, SEV3_BOUNDS)
    correct = (class_true == class_pred)
    stats_d = _regression_stats(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(183 * MM, 165 * MM))
    render_scatter(ax, y_true, y_pred, correct, stats_d,
                   SEV3_BOUNDS, SEV3_LABELS,
                   title="True vs Predicted Cobb Angle (3-Class)")
    _save_fig(fig, "散点图_真值vs预测_3分类", out_dir)


def fig2_bland_altman(y_true, y_pred, out_dir: Path | None = None):
    diff = y_pred - y_true
    mean = (y_true + y_pred) / 2
    md = np.mean(diff)
    sd = np.std(diff, ddof=1)
    fig, ax = plt.subplots(figsize=(120 * MM, 105 * MM))
    render_bland_altman(ax, mean, diff, md, sd)
    _save_fig(fig, "Bland_Altman图", out_dir)


def fig3_cm_4class(y_true, y_pred, out_dir: Path | None = None):
    cm, labels = _confusion_4class(y_true, y_pred)
    short = {"Normal": "0-10°", "Mild": "10-20°", "Moderate": "20-40°", "Severe": "40+°"}
    lines = _cm_stats_lines(cm, labels, short)
    fig = plt.figure(figsize=(155 * MM, 95 * MM))
    ax_cm = fig.add_axes([0.08, 0.15, 0.55, 0.75])
    render_confusion_matrix(ax_cm, cm, labels, title="4-Class Confusion Matrix")
    render_cm_annotation(fig, lines)
    _save_fig(fig, "混淆矩阵_4分类", out_dir)


def fig4_cm_3class(y_true, y_pred, out_dir: Path | None = None):
    cm, labels = _confusion_3class(y_true, y_pred)
    short = {"0-20°": "0-20", "20-40°": "20-40", "40+°": "40+"}
    lines = _cm_stats_lines(cm, labels, short)
    fig = plt.figure(figsize=(130 * MM, 90 * MM))
    ax_cm = fig.add_axes([0.08, 0.15, 0.55, 0.75])
    render_confusion_matrix(ax_cm, cm, labels, title="3-Class Confusion Matrix")
    render_cm_annotation(fig, lines)
    _save_fig(fig, "混淆矩阵_3分类", out_dir)


def fig5_subject_comparison(y_true, y_pred, class_true, class_pred, out_dir: Path | None = None):
    order = np.argsort(y_true)
    yt_s, yp_s, ct_s, cp_s = y_true[order], y_pred[order], class_true[order], class_pred[order]
    misclassified = (ct_s != cp_s)
    fig, ax = plt.subplots(figsize=(183 * MM, 85 * MM))
    render_subject_comparison(ax, yt_s, yp_s, misclassified, SEVERITY_BOUNDS)
    _save_fig(fig, "逐受试者预测对比图", out_dir)


def main(pred_csv: Path | None = None, out_dir: Path | None = None) -> None:
    """论文关键图（散点/Bland-Altman/混淆矩阵/逐受试者），数据源/输出目录可切换。

    Args:
        pred_csv: 预测 CSV；None 时用 v0.1.0 ``ENSEMBLE_PRED_PATH``。
        out_dir: 输出目录；None 时用 ``EXPORT_FIGURES_DIR``。
    """
    pred_path = Path(pred_csv) if pred_csv else ENSEMBLE_PRED_PATH
    target = Path(out_dir) if out_dir else EXPORT_FIGURES_DIR
    target.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(pred_path)
    y_true = pred["max_cobb_true"].values.astype(float)
    y_pred = pred["max_cobb_pred"].values.astype(float)
    ct = pred["class_true"].values
    cp = pred["class_pred"].values
    logger.info(f"Loaded {len(y_true)} predictions")

    logger.info("\n[Figure 1] Scatter plot")
    fig1_scatter(y_true, y_pred, ct, cp, target)
    logger.info("\n[Figure 2] Bland-Altman plot")
    fig2_bland_altman(y_true, y_pred, target)
    logger.info("\n[Figure 3] 4-class confusion matrix")
    fig3_cm_4class(y_true, y_pred, target)
    logger.info("\n[Figure 4] 3-class confusion matrix")
    fig4_cm_3class(y_true, y_pred, target)
    logger.info("\n[Figure 5] Per-subject comparison")
    fig5_subject_comparison(y_true, y_pred, ct, cp, target)
    logger.info(f"\nAll figures saved to {target}/")


if __name__ == "__main__":
    main()
