"""Pure rendering: feature importance charts — Top15 barh, pie charts.

Rendering layer — no I/O, no calculations, no project imports beyond typing.
Receives np.ndarray + Python natives; returns None.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

MIN_INSIDE_PCT = 5  # 扇形占比 ≥5% 时标签绘制在内部


def render_top15_barh(ax: Axes, vals: np.ndarray, lbls: list[str],
                      colors: list[str], measure_types: list[str]) -> None:
    """Horizontal bar chart for Top-15 features.

    Parameters
    ----------
    ax : Axes
        Target axes.
    vals : np.ndarray
        Importance values (sorted ascending for display).
    lbls : list[str]
        Feature display labels (same order as vals).
    colors : list[str]
        Per-bar colors (same order as vals).
    measure_types : list[str]
        Measurement type per bar, used for legend.
    """
    y = np.arange(len(lbls))
    ax.barh(y, vals, 0.6, color=colors, edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=6.5)
    ax.set_xlabel("Permutation Importance", fontsize=7)
    ax.set_title("Top 15 Features by Importance", fontweight="bold", fontsize=8.5)

    # Value labels
    for i, (v, _) in enumerate(zip(vals, measure_types, strict=False)):
        ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=6, color="#444")

    # Legend for unique measurement types (preserve order)
    uniq: dict[str, str] = {}
    for m, c in zip(measure_types, colors, strict=False):
        uniq.setdefault(m, c)
    legend_elements = [Patch(facecolor=c, label=m) for m, c in uniq.items()]
    ax.legend(handles=legend_elements, fontsize=6.5, loc="lower right")
    ax.set_xlim(0, vals.max() * 1.25)


def _smart_pie_labels(ax: Axes, wedges, vals: np.ndarray, total: float) -> None:
    """Add percentage labels to pie wedges — inside for >=5%, connector for <5%."""
    for w, v in zip(wedges, vals, strict=False):
        pct = v / total * 100
        theta = np.deg2rad((w.theta1 + w.theta2) / 2)
        if pct >= MIN_INSIDE_PCT:
            ax.text(0.55 * np.cos(theta), 0.55 * np.sin(theta),
                    f"{pct:.1f}%", ha="center", va="center", fontsize=7)
        else:
            xe, ye = np.cos(theta), np.sin(theta)
            xl, yl = 1.08 * np.cos(theta), 1.08 * np.sin(theta)
            ax.annotate(f"{pct:.1f}%", xy=(xe, ye), xytext=(xl, yl),
                        ha="left" if xl > 0 else "right", va="center",
                        fontsize=6.5,
                        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5))


def render_importance_pie(fig: Figure, vals: np.ndarray, lbls: list[str],
                          colors: list[str], title: str) -> None:
    """Pie chart with colored legend, rendered on a pre-created figure.

    Parameters
    ----------
    fig : Figure
        Pre-created figure (size set by caller).
    vals : np.ndarray
        Slice values.
    lbls : list[str]
        Slice labels.
    colors : list[str]
        Per-slice colors (same order as vals).
    title : str
        Chart title.
    """
    total = vals.sum()
    ax = fig.add_axes([0.05, 0.05, 0.55, 0.9])
    wedges, _ = ax.pie(vals, labels=None, autopct=None, startangle=90,
                        colors=colors,
                        wedgeprops=dict(linewidth=0.5, edgecolor="white"))
    _smart_pie_labels(ax, wedges, vals, total)
    ax.set_title(title, fontweight="bold", fontsize=8.5, pad=8)

    handles = [
        Patch(facecolor=c, label=f"{lb}  {v / total * 100:.1f}%")
        for lb, v, c in zip(lbls, vals, colors, strict=False)
    ]
    ax.legend(handles=handles, loc="center left", fontsize=7,
              bbox_to_anchor=(1, 0.5), frameon=False)
