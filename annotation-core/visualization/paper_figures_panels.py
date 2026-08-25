"""Pure rendering: paper figures — scatter, Bland-Altman, confusion matrix, subject comparison.

Rendering layer — no I/O, no calculations, no project imports beyond typing.
Receives np.ndarray + Python natives; returns None.
"""

from __future__ import annotations

import matplotlib.cm as mcm
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ── Academic muted colors ──
C_CORRECT = "#5A8C5A"     # muted green
C_INCORRECT = "#B06060"   # muted red
C_GT = "#4A72A0"          # muted blue
C_PRED = "#B06060"        # muted red
C_DIAG = "#999999"        # academic gray
C_REGR = "#C88A4A"        # muted orange
LABEL_INSIDE_PCT = 5  # 单元格占比 ≥5% 时在格内标注百分比
DARK_CELL_RATIO = 0.55  # 单元格占比超过该值用白色文字


def render_scatter(ax: Axes, y_true: np.ndarray, y_pred: np.ndarray,
                   correct_mask: np.ndarray | None,
                   stats: dict, bounds: list[float], bounds_labels: list[str],
                   title: str = "True vs Predicted Cobb Angle") -> None:
    """Scatter plot: true vs predicted Cobb angle.

    Parameters
    ----------
    ax : Axes
        Target axes.
    y_true : np.ndarray
        Ground truth Cobb angles.
    y_pred : np.ndarray
        Predicted Cobb angles.
    correct_mask : np.ndarray | None
        Boolean mask of correct classifications (None = single-color).
    stats : dict
        Must contain r2, r, p, slope, intercept.
    bounds : list[float]
        Severity threshold boundaries.
    bounds_labels : list[str]
        Labels for severity thresholds.
    title : str
        Plot title.
    """
    if correct_mask is not None:
        ax.scatter(y_true[correct_mask], y_pred[correct_mask],
                   c=C_CORRECT, alpha=0.6, edgecolors="white",
                   linewidth=0.4, s=35, label="Correct")
        ax.scatter(y_true[~correct_mask], y_pred[~correct_mask],
                   c=C_INCORRECT, alpha=0.5, edgecolors="white",
                   linewidth=0.4, s=35, label="Incorrect")
    else:
        ax.scatter(y_true, y_pred, c=C_PRED, alpha=0.5,
                   edgecolors="white", linewidth=0.4, s=35)

    lims = [min(y_true.min(), y_pred.min()) - 2,
            max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "--", color=C_DIAG, alpha=0.4,
            linewidth=1, label="y=x")
    x_line = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_line, stats["slope"] * x_line + stats["intercept"],
            "-", color=C_REGR, linewidth=1.5, alpha=0.8, label="Regression")

    for b in bounds:
        ax.axhline(b, color=C_DIAG, linestyle=":", alpha=0.25, linewidth=0.6)
        ax.axvline(b, color=C_DIAG, linestyle=":", alpha=0.25, linewidth=0.6)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True Cobb Angle (°)")
    ax.set_ylabel("Predicted Cobb Angle (°)")
    ax.set_title(title, fontweight="bold")

    n = len(y_true)
    text = (f"n = {n}\n"
            f"R² = {stats['r2']:.3f}\n"
            f"r = {stats['r']:.3f}\n"
            f"p = {stats['p']:.2e}\n"
            f"RMSE = {stats['rmse']:.2f}°")
    ax.text(0.04, 0.96, text, transform=ax.transAxes, fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
    ax.legend(fontsize=7, loc="lower right")


def render_bland_altman(ax: Axes, mean: np.ndarray, diff: np.ndarray,
                        md: float, sd: float) -> None:
    """Bland-Altman plot.

    Parameters
    ----------
    ax : Axes
        Target axes.
    mean : np.ndarray
        Mean of true and predicted.
    diff : np.ndarray
        Difference (predicted - true).
    md : float
        Mean difference.
    sd : float
        Standard deviation of differences.
    """
    upper = md + 1.96 * sd
    lower = md - 1.96 * sd

    ax.scatter(mean, diff, c=C_GT, alpha=0.5, edgecolors="white",
               linewidth=0.4, s=30)
    ax.axhline(md, color="#222222", linewidth=1, linestyle="-",
               label=f"Mean diff: {md:.2f}°")
    ax.axhline(upper, color=C_INCORRECT, linewidth=0.8, linestyle="--",
               label=f"+1.96 SD: {upper:.2f}°")
    ax.axhline(lower, color=C_INCORRECT, linewidth=0.8, linestyle="--",
               label=f"-1.96 SD: {lower:.2f}°")
    ax.axhline(0, color=C_DIAG, linestyle=":", alpha=0.4, linewidth=0.6)
    ax.set_xlabel("Mean of True and Predicted (°)")
    ax.set_ylabel("Difference (Predicted − True) (°)")
    ax.set_title("Bland-Altman Plot", fontweight="bold")
    ax.legend(fontsize=7)


def render_confusion_matrix(ax: Axes, cm: np.ndarray, labels: list[str],
                            title: str = "") -> None:
    """Confusion matrix heatmap with cell annotations.

    Parameters
    ----------
    ax : Axes
        Target axes.
    cm : np.ndarray
        Square confusion matrix (int).
    labels : list[str]
        Class labels for ticks.
    title : str
        Optional title.
    """
    cm_arr = np.array(cm, dtype=float)
    n = len(labels)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr),
                       where=row_sums > 0)

    cmap = mcm.Blues
    ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=1, alpha=0.85)

    for i in range(n):
        for j in range(n):
            val = int(cm_arr[i, j])
            pct = cm_pct[i, j] * 100
            cell_text = f"{val}" if pct < LABEL_INSIDE_PCT else f"{val}\n({pct:.0f}%)"
            is_dark = cm_pct[i, j] > DARK_CELL_RATIO
            is_diag = i == j
            fs = 10 if is_diag else 9
            fw = "bold" if is_diag else "normal"
            ax.text(j, i, cell_text, ha="center", va="center", fontsize=fs,
                    fontweight=fw, color="white" if is_dark else "#222222")

    for i in range(n + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
        ax.axvline(i - 0.5, color="white", linewidth=1.5)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")


def render_cm_annotation(fig: Figure, stats_lines: str) -> None:
    """Add a monospace stats annotation box to a confusion matrix figure.

    Parameters
    ----------
    fig : Figure
        Target figure.
    stats_lines : str
        Multi-line string with accuracy, per-class F1/prec/rec.
    """
    fig.text(0.7, 0.5, stats_lines, fontsize=6.5,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="#cccccc", linewidth=0.5))


def render_subject_comparison(ax: Axes, y_true: np.ndarray,
                              y_pred: np.ndarray,
                              misclassified_mask: np.ndarray,
                              bounds: list[float]) -> None:
    """Per-subject line plot comparing true vs predicted Cobb angles.

    Parameters
    ----------
    ax : Axes
        Target axes.
    y_true : np.ndarray
        Ground truth (sorted).
    y_pred : np.ndarray
        Predicted (same order as y_true).
    misclassified_mask : np.ndarray
        Boolean mask of misclassified subjects.
    bounds : list[float]
        Severity threshold boundaries.
    """
    x = np.arange(len(y_true))
    ax.plot(x, y_true, "-", color=C_GT, linewidth=1.2, alpha=0.8,
            label="Ground Truth")
    ax.plot(x, y_pred, "-", color=C_PRED, linewidth=1.2, alpha=0.8,
            label="Predicted")
    ax.scatter(x[misclassified_mask], y_pred[misclassified_mask],
               color=C_INCORRECT, s=12, zorder=5, label="Misclassified")

    for b in bounds:
        ax.axhline(b, color=C_DIAG, linestyle=":", alpha=0.3, linewidth=0.6)

    ax.set_xlabel("Subject (sorted by true Cobb)")
    ax.set_ylabel("Cobb Angle (°)")
    ax.set_title("Per-Subject Prediction Comparison", fontweight="bold")
    ax.legend(fontsize=7, ncol=3)
