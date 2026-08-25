"""预测 vs 真实散点图组件（piecewise_log48 变换后）。

按 TN/FP/FN/TP 分类着色，绘制预测值 vs 真实值散点图。
x/y 轴均经过 piecewise_log48 变换，尾部离群点被压缩。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

PIECEWISE_THRESHOLD = 48.0
MIN_POINTS = 2  # 有效点不足时提示数据不足
MIN_FIT_POINTS = 4  # 拟合渐近线所需最少点数

# 3 分类严重度分级（0-20° / 20-40° / 40+°）—— 报告 3 类评估用
SEV3_BINS = [20, 40]
SEV3_LABELS = ["0-20°", "20-40°", "40+°"]


def _tr48(v: np.ndarray) -> np.ndarray:
    """piecewise_log48 变换 — >48° 的部分取 log 压缩。"""
    y = v.copy()
    m = y > PIECEWISE_THRESHOLD
    y[m] = PIECEWISE_THRESHOLD + np.log(y[m] - PIECEWISE_THRESHOLD + 1)
    return y


def render_scatter(
    axes: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cm: list[int],
    *,
    threshold: float = 20.0,
    title: str = "Predicted vs True Cobb Angle",
    show_fit_line: bool = False,
) -> None:
    """在给定 axes 上绘制分类着色散点图（log48 变换后）。

    Args:
        axes: matplotlib Axes 对象
        y_true: 真实 Cobb 角数组 (n,)
        y_pred: 预测 Cobb 角数组 (n,)
        cm: 混淆矩阵四元素 [TN, FP, FN, TP]（仅用于着色参考）
        threshold: 临床阈值，默认 20°
        title: 图标题
    """
    # NaN/Inf 保护
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < MIN_POINTS:
        axes.text(0.5, 0.5, "数据不足", ha="center", va="center", transform=axes.transAxes)
        axes.set_title(title)
        return
    y_true, y_pred = y_true[valid], y_pred[valid]

    clinical = 20.0

    # 变换到 log48 空间
    yt_t = _tr48(y_true)
    yp_t = _tr48(y_pred)

    true_bin = y_true > clinical
    pred_bin = y_pred > threshold

    # 分类着色
    colors = np.where(
        ~true_bin & ~pred_bin,
        "#22c55e",  # TN 绿色
        np.where(
            ~true_bin & pred_bin,
            "#ef4444",  # FP 红色
            np.where(
                true_bin & ~pred_bin,
                "#f97316",  # FN 橙色
                "#3b82f6",
            ),
        ),
    )  # TP 蓝色

    axes.scatter(yt_t, yp_t, c=colors, alpha=0.7, edgecolors="white", linewidth=0.5, s=30)

    # 对角线参考线（log48 空间）
    max_val = max(yt_t.max(), yp_t.max()) + 3
    axes.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=1)
    axes.set_xlim(0, max_val)
    axes.set_ylim(0, max_val)

    # 标记临床阈值线（在 log48 空间中，≤48° 不变，所以 20° 仍是 20）
    if threshold <= PIECEWISE_THRESHOLD:
        axes.axhline(threshold, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        axes.axvline(threshold, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

    # 拟合渐近线（淡红虚线）
    if show_fit_line and len(yt_t) >= MIN_FIT_POINTS:
        try:
            z = np.polyfit(yt_t, yp_t, 1)
            p = np.poly1d(z)
            x_line = np.linspace(yt_t.min(), yt_t.max(), 50)
            axes.plot(x_line, p(x_line), "--", color="#e74c3c", alpha=0.5, linewidth=1.5)
        except np.linalg.LinAlgError:
            pass

    axes.set_xlabel("True Cobb (log48 °)")
    axes.set_ylabel("Predicted Cobb (log48 °)")
    axes.set_title(title)

    # 图例
    legend_data = [
        ("TN (真阴)", "#22c55e"),
        ("FP (误报)", "#ef4444"),
        ("FN (漏报)", "#f97316"),
        ("TP (正确>20°)", "#3b82f6"),
    ]
    for label, color in legend_data:
        axes.scatter([], [], c=color, label=label)
    axes.legend(fontsize=7, loc="lower right")


def render_scatter_4class(
    axes: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_true: np.ndarray,
    class_pred: np.ndarray,
    *,
    title: str = "Predicted vs True Cobb Angle",
) -> None:
    """4 分类着色散点图 — 正确=绿，错误=红，画 3 条严重度边界线。

    Args:
        axes: matplotlib Axes 对象
        y_true: 真实 Cobb 角 (n,)
        y_pred: 预测 Cobb 角 (n,)
        class_true: 真实分类标签 (n,)
        class_pred: 预测分类标签 (n,)
        title: 图标题
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < MIN_POINTS:
        axes.text(0.5, 0.5, "数据不足", ha="center", va="center", transform=axes.transAxes)
        axes.set_title(title)
        return

    y_true, y_pred = y_true[valid], y_pred[valid]
    class_true, class_pred = np.array(class_true)[valid], np.array(class_pred)[valid]

    correct = class_true == class_pred
    colors = np.where(correct, "#22c55e", "#ef4444")

    axes.scatter(y_true, y_pred, c=colors, alpha=0.7, edgecolors="white", linewidth=0.5, s=30)

    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    axes.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

    # 3 条严重度边界线 (10°, 20°, 40°)
    boundaries = [10, 20, 40]
    labels = ["Normal", "Mild", "Moderate", "Severe"]
    for i, b in enumerate(boundaries):
        axes.axhline(b, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        axes.axvline(b, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        axes.text(lims[0] + 0.5, b + 0.5, f"{labels[i]}", fontsize=7, color="gray", alpha=0.6)

    axes.set_xlim(lims)
    axes.set_ylim(lims)
    axes.set_xlabel("True Cobb Angle (°)")
    axes.set_ylabel("Predicted Cobb Angle (°)")
    axes.set_title(title)

    for label, color in [("Correct", "#22c55e"), ("Incorrect", "#ef4444")]:
        axes.scatter([], [], c=color, label=label)
    axes.legend(fontsize=8, loc="lower right")


def render_scatter_3class(
    axes: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "",
) -> None:
    """3 分类着色散点图 — 正确=绿，错误=红，画 2 条边界线 (20°, 40°)。

    Args:
        axes: matplotlib Axes 对象
        y_true: 真实 Cobb 角 (n,)
        y_pred: 预测 Cobb 角 (n,)
        title: 图标题
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < MIN_POINTS:
        axes.text(0.5, 0.5, "数据不足", ha="center", va="center", transform=axes.transAxes)
        axes.set_title(title)
        return
    y_true, y_pred = y_true[valid], y_pred[valid]

    tc_true = np.digitize(y_true, SEV3_BINS)
    tc_pred = np.digitize(y_pred, SEV3_BINS)
    correct = tc_true == tc_pred
    colors = np.where(correct, "#22c55e", "#ef4444")

    axes.scatter(y_true, y_pred, c=colors, alpha=0.7, edgecolors="white", linewidth=0.5, s=30)
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    axes.plot(lims, lims, "k--", alpha=0.3, linewidth=1)

    for i, b in enumerate(SEV3_BINS):
        axes.axhline(b, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        axes.axvline(b, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
        axes.text(lims[0] + 0.5, b + 0.5, f"{SEV3_LABELS[i]}", fontsize=7, color="gray", alpha=0.6)

    axes.set_xlim(lims)
    axes.set_ylim(lims)
    axes.set_xlabel("True Cobb Angle (°)")
    axes.set_ylabel("Predicted Cobb Angle (°)")
    axes.set_title(title)
    for lbl, c in [("Correct", "#22c55e"), ("Incorrect", "#ef4444")]:
        axes.scatter([], [], c=c, label=lbl)
    axes.legend(fontsize=8, loc="lower right")
