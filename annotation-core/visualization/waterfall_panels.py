"""Rendering layer: waterfall + convergence plots.

Pure matplotlib rendering, no computation, no I/O.
Receives pre-computed data and draws on passed Axes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

import numpy as np

# ── Default color scheme ──
POS_COLOR     = "#5A8C5A"   # muted green: positive bars + labels
NEG_COLOR     = "#B06060"   # muted red:   negative bars + labels
PRED_COLOR    = "#4A72A0"   # muted blue:  prediction bar + label
GT_COLOR      = "#C88A4A"   # muted orange: ground truth line + label
BASELINE_COLOR = "#888888"

# ── Default hatch config (ss lw3) ──
HATCH_STR = "//"
HATCH_LW_SKINNY = 3.0


# ── Helpers ──

def _fb(ax: Axes, x: float, bottom: float, height: float, color: str,
        hatch: str, hatch_lw: float, bar_w: float) -> None:
    """Draw a single bar + hatch overlay on *ax*."""
    h = max(height, 0.2)                     # ponytail: min visual thickness
    import matplotlib as mpl
    _saved = mpl.rcParams["hatch.linewidth"]
    mpl.rcParams["hatch.linewidth"] = hatch_lw
    ax.bar(x, h, bar_w, bottom=bottom, color=color,
           alpha=0.85, edgecolor="white", linewidth=0.4, zorder=2)
    ax.bar(x, h, bar_w, bottom=bottom, color="none",
           edgecolor="white", alpha=0.3, hatch=hatch, linewidth=0, zorder=2)
    mpl.rcParams["hatch.linewidth"] = _saved


def _conn(ax: Axes, x0: float, x1: float, y: float) -> None:
    """Horizontal dashed connector line at cumulative level *y*."""
    ax.plot([x0, x1], [y, y], color="#999999",
            linestyle="--", linewidth=0.5, alpha=0.6, zorder=1)


# ── Waterfall ──

def render_waterfall(
    ax: Axes,
    group_contrib: dict[str, float],
    expected_val: float,
    true_cobb: float,
    pred_val: float,
    subject_id: str,
    severity: str,
    label: str,
    *,
    hatch: str = HATCH_STR,
    hatch_lw: float = HATCH_LW_SKINNY,
    fixed_order: tuple[str, ...] | None = None,
) -> None:
    """Render a grouped waterfall chart on *ax*.

    Parameters are plain Python types — no project model imports.
    """
    # ── Sort / order categories ──
    if fixed_order:
        cats = [c for c in fixed_order if c in group_contrib]
    else:
        cats = sorted(group_contrib, key=lambda c: -abs(group_contrib[c]))
    vals = [group_contrib[c] for c in cats]

    # ── Cumulative positions ──
    cum = expected_val
    bottoms = [cum]
    for v in vals:
        cum += v
        bottoms.append(cum)

    n = len(cats)
    n_bars = n + 2
    bar_w = 0.55
    xs = np.arange(n_bars)

    # 1) Baseline bar
    _fb(ax, xs[0], 0, expected_val, BASELINE_COLOR, hatch, hatch_lw, bar_w)

    # 2) Floating feature bars
    for i in range(n):
        idx = i + 1
        bottom = min(bottoms[i], bottoms[i + 1])
        height = abs(vals[i])
        color = POS_COLOR if vals[i] >= 0 else NEG_COLOR
        _fb(ax, xs[idx], bottom, height, color, hatch, hatch_lw, bar_w)

        # Value label
        label_y = (max(bottoms[i], bottoms[i + 1]) + 0.35
                   if vals[i] >= 0 else
                   min(bottoms[i], bottoms[i + 1]) - 0.35)
        va = "bottom" if vals[i] >= 0 else "top"
        ax.text(xs[idx], label_y, f"{vals[i]:+.2f}",
                ha="center", va=va, fontsize=6,
                color=color, zorder=4)

        # Connector
        x_r = xs[i] + bar_w / 2
        x_l = xs[idx] - bar_w / 2
        _conn(ax, x_r, x_l, bottoms[i])

    # 3) Total bar
    tot = n_bars - 1
    _fb(ax, xs[tot], 0, pred_val, PRED_COLOR, hatch, hatch_lw, bar_w)
    _conn(ax, xs[n] + bar_w / 2, xs[tot] - bar_w / 2, bottoms[-1])

    # 4) GT line
    ax.axhline(true_cobb, color=GT_COLOR, linestyle="--",
               alpha=0.5, linewidth=0.6, zorder=3)

    # 5) X-axis labels
    display_cats = [c.replace("Normal Angle", "Normal\nAngle") for c in cats]
    x_labels = ["Dataset\nmean"] + display_cats + ["Prediction"]
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=6, rotation=0, ha="center")
    for lb in ax.get_xticklabels():
        lb.set_fontweight("normal")

    # 6) Value labels (baseline, prediction, GT only)
    ax.text(xs[0], expected_val + 0.35, f"{expected_val:.1f}",
            ha="center", va="bottom", fontsize=6, color="#222222")
    ax.text(xs[tot], pred_val + 0.35, f"{pred_val:.1f}",
            ha="center", va="bottom", fontsize=6, color=PRED_COLOR)
    # Ground truth: centered on the GT line, text below, number above
    ax.text(xs[-1] + 0.6, true_cobb - 0.35, "Ground truth",
            ha="left", va="top", fontsize=5.5, color=GT_COLOR)
    ax.text(xs[-1] + 1.0, true_cobb + 0.35, f"{true_cobb:.1f}",
            ha="left", va="bottom", fontsize=5.5, color=GT_COLOR)

    # 7) Title
    ax.set_title(
        f"SHAP waterfall: {subject_id}",
        fontsize=7.5, fontweight="bold", pad=6)

    # 8) Limits
    y_max = max(true_cobb, expected_val, pred_val) + 3
    ax.set_xlim(-1.2, n_bars - 1 + 1.2)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("Cobb angle (degrees)", fontsize=6.5, labelpad=2)


# ── Residual convergence (single subject, 100 trees) ──

def render_residual_convergence(
    ax: Axes, ax2: Axes,
    staged_preds: np.ndarray, per_tree_contrib: np.ndarray,
    residuals: np.ndarray, true_cobb: float, subject_id: str,
    *,
    n_trees: int = 100,
    pos_color: str = POS_COLOR, neg_color: str = NEG_COLOR,
    pred_color: str = PRED_COLOR, res_color: str = "#A0785A",
    gt_color: str = GT_COLOR,
) -> None:
    """Render single-subject residual convergence with dual Y-axes."""
    sv, tc, res = staged_preds[:n_trees], per_tree_contrib[:n_trees], residuals[:n_trees]
    xs = np.arange(n_trees)
    max_tc = max(abs(tc)) * 1.3
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.5)
    ax2.tick_params(axis="y", colors="#666", labelsize=6.5, direction="out")
    for i in range(n_trees):
        b = 0 if tc[i] >= 0 else tc[i]
        h = tc[i] if tc[i] >= 0 else -tc[i]
        c = pos_color if tc[i] >= 0 else neg_color
        ax2.bar(i, h, 0.7, bottom=b, color=c, alpha=0.25, edgecolor="white", linewidth=0.1, zorder=1)
    ax2.set_ylim(-max_tc, max_tc)
    ax2.set_ylabel("Tree contribution (°)", fontsize=7, color="#666")
    ax.plot(xs, sv, color=pred_color, linewidth=0.8, zorder=3, label="Prediction")
    ax.plot(xs, res, color=res_color, linewidth=0.6, zorder=2, label="Residual")
    ax.axhline(true_cobb, color=gt_color, linestyle="--", alpha=0.5, linewidth=0.6, zorder=0)
    ax.text(n_trees - 1, true_cobb + 0.8, f"GT={true_cobb:.0f}", ha="right", va="bottom", fontsize=5.5, color=gt_color)
    ax.set_xlabel("Tree iteration", fontsize=7)
    ax.set_ylabel("Cobb angle (°)", fontsize=7, color="#222")
    ax.tick_params(axis="y", colors="#222", labelsize=6.5)
    ax.set_xlim(0, n_trees - 1)
    y_top = max(true_cobb + 5, np.max(sv) * 1.2 + 3)
    ax.set_ylim(-0.5, y_top)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    ax.legend(handles=[Line2D([],[],color=pred_color,linewidth=1.0,label="Prediction"),Line2D([],[],color=res_color,linewidth=0.8,label="Residual"),Rectangle((0,0),1,1,facecolor=pos_color,alpha=0.35,label="Positive contribution"),Rectangle((0,0),1,1,facecolor=neg_color,alpha=0.35,label="Negative contribution")],fontsize=6,loc="lower center",ncol=4,bbox_to_anchor=(0.5,-0.55))
    ax.set_title(f"Gradient boosting convergence (first 100 trees): {subject_id}", fontsize=7.5, fontweight="bold", pad=6)


# ── Tree structure (first N trees) ──

def _collect_nodes(nodes, ni: int, depth: int = 0) -> list:
    rows = [(ni, depth, bool(nodes[ni]["is_leaf"]), int(nodes[ni]["feature_idx"]), float(nodes[ni]["num_threshold"]), float(nodes[ni]["value"]), int(nodes[ni]["left"]), int(nodes[ni]["right"]))]
    if not nodes[ni]["is_leaf"]:
        rows += _collect_nodes(nodes, int(nodes[ni]["left"]), depth + 1)
        rows += _collect_nodes(nodes, int(nodes[ni]["right"]), depth + 1)
    return rows

def _subject_path_nodes(nodes, ni: int, x0: np.ndarray) -> set:
    path = set()
    while not nodes[ni]["is_leaf"]:
        path.add(ni)
        f = int(nodes[ni]["feature_idx"])
        ni = int(nodes[ni]["left"]) if x0[f] <= nodes[ni]["num_threshold"] else int(nodes[ni]["right"])
    path.add(ni)
    return path

def render_tree_structure(axes: list[Axes], predictors: list, x0: np.ndarray, feature_names: list[str], *, feature_labels: list[str] | None = None, learning_rate: float = 0.1, n_trees: int = 3, subject_id: str = "") -> None:
    """Render the first *n_trees* tree structures with subject path highlighted."""
    for ti in range(n_trees):
        ax = axes[ti]
        nodes = predictors[ti][0].nodes
        all_nodes = _collect_nodes(nodes, 0)
        path = _subject_path_nodes(nodes, 0, x0)

        ax.text(0.5, 0.95, f"Tree {ti+1}", fontsize=7.5, fontweight="bold", ha="center", va="top", transform=ax.transAxes)
        ax.axis("off")

        # Layout: position nodes by depth and index
        by_depth = {}
        md = max(r[1] for r in all_nodes)
        for r in all_nodes:
            by_depth.setdefault(r[1], []).append(r[0])
        pos = {}
        for d, ns in by_depth.items():
            for j, ni in enumerate(ns):
                pos[ni] = ((j + 0.5) / len(ns), 1.0 - (d + 0.5) / (md + 1))

        # Edges (diagonal, subject path in orange)
        for r in all_nodes:
            ni, _, leaf, _, _, _, left, right = r
            if not leaf:
                for child in (left, right):
                    if child in pos and ni in pos:
                        on_path = (child in path) and (ni in path)
                        lw = 1.5 if on_path else 0.6
                        alpha = 0.9 if on_path else 0.4
                        ax.plot([pos[ni][0], pos[child][0]],
                                [pos[ni][1], pos[child][1]],
                                color="#E65100" if on_path else "#999999",
                                linewidth=lw, alpha=alpha, zorder=1)

        # Nodes
        for r in all_nodes:
            ni, _, leaf, feat, thresh, val, _, _ = r
            x, y = pos[ni]
            on_path = ni in path

            if leaf:
                display_val = f"{val * learning_rate:.4f}"
                if on_path:
                    label = display_val
                    fc, ec, fs, lw = "#E8F5E9", "#2E7D32", 5, 0.8
                else:
                    label = display_val
                    fc, ec, fs, lw = "#F5F5F5", "#BDBDBD", 4.5, 0.5
                ax.annotate(label, (x, y), ha="center", va="center",
                            fontsize=fs, fontweight="bold" if on_path else "normal",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor=fc,
                                      edgecolor=ec, linewidth=lw))
            else:
                fnm = feature_labels[feat] if (feature_labels and feat < len(feature_labels)) else (feature_names[feat] if feat < len(feature_names) else f"f{feat}")
                feat_val = f"{x0[feat]:.3f}" if feat < len(x0) else "?"
                if on_path:
                    label = f"{fnm}\n≤ {thresh:.2f}  [{feat_val}]"
                    fc, ec, fs, lw = "#FFF3E0", "#E65100", 4.5, 0.8
                else:
                    label = f"{fnm}\n≤ {thresh:.2f}"
                    fc, ec, fs, lw = "#FAFAFA", "#BDBDBD", 4, 0.5
                ax.annotate(label, (x, y), ha="center", va="center",
                            fontsize=fs, fontweight="bold" if on_path else "normal",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor=fc,
                                      edgecolor=ec, linewidth=lw))

    if subject_id:
        axes[0].set_title(f"First {n_trees} trees: {subject_id}",
                          fontsize=7.5, fontweight="bold", pad=6)
    for ax in axes:
        ax.set_ylim(-0.05, 1.15)
