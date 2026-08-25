#!/usr/bin/env python3
"""编排层：特征重要性三图 — Top15 + 解剖分区 + 测量类型。

用法:
    uv run python -m commands.export.charts_feature_importance
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from commands.export.config import EXPORT_SHAP_DIR, FEATURE_IMPORTANCE_DIR
from features.utils.ci_decompose import _assign_group, feature_display_name
from utils.logger import logger
from visualization.feature_importance_panels import (
    render_importance_pie,
    render_top15_barh,
)

# ── Academic muted colors ──
C_NORMALVEC = "#4A72A0"
C_CURVATURE = "#5A8C5A"
C_ROUGHNESS = "#C88A4A"
C_HEIGHT = "#D49A4A"
C_MORPH = "#6B5B8A"
C_CLINICAL = "#999999"
C_MEASURE_COLORS = {
    "Normal Angle": C_NORMALVEC,
    "Curvature": C_CURVATURE,
    "Roughness": C_ROUGHNESS,
    "Height": C_HEIGHT,
    "Morph": C_MORPH,
    "Clinical": C_CLINICAL,
}
C_UV_COLORS = {
    "Shoulder": C_NORMALVEC,
    "Scapula": C_CURVATURE,
    "Axilla": C_ROUGHNESS,
    "Waist": C_HEIGHT,
    "Pelvis": "#B06060",
    "Morph": C_MORPH,
    "Clinical": C_CLINICAL,
}

# ── Typography ──
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
})
MM = 1 / 25.4


def _strip_uv_prefix(label: str) -> str:
    name = label.split(": ", 1)[1] if ": " in label else label
    name = {"Neck": "Shoulder", "Shoulder/Scapula": "Scapula"}.get(name, name)
    return name


def _save_fig(fig, name: str) -> None:
    out_dir = EXPORT_SHAP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    fig.savefig(out_dir / f"{name}.pdf", dpi=300)
    logger.info(f"  Saved: {name}.png + .pdf")
    plt.close(fig)


def main():
    EXPORT_SHAP_DIR.mkdir(parents=True, exist_ok=True)

    top15 = pd.read_csv(FEATURE_IMPORTANCE_DIR / "feature_importance_decomposed.csv")
    uv = pd.read_csv(FEATURE_IMPORTANCE_DIR / "importance_by_horizontal_band.csv")
    mt = pd.read_csv(FEATURE_IMPORTANCE_DIR / "importance_by_group.csv")

    # ── Figure 1: Top 15 ──
    logger.info("\n[Figure 1] Top 15 features")
    top = top15.head(15).copy()
    vals = top["importance"].values[::-1]
    lbls = [feature_display_name(f) for f in top["feature"].values[::-1]]
    msrs = list(top["feature"].apply(_assign_group).values[::-1])
    colors = [C_MEASURE_COLORS.get(m, C_CLINICAL) for m in msrs]
    fig, ax = plt.subplots(figsize=(183 * MM, 120 * MM))
    render_top15_barh(ax, vals, lbls, colors, msrs)
    _save_fig(fig, "特征重要性_Top15")

    # ── Figure 2: Anatomical region ──
    logger.info("\n[Figure 2] UV horizontal band")
    df_uv = uv.sort_values("importance", ascending=False)
    lbls_uv = [_strip_uv_prefix(label) for label in df_uv["horizontal_band"].values]
    vals_uv = df_uv["importance"].values
    colors_uv = [C_UV_COLORS[label] for label in lbls_uv]
    fig = plt.figure(figsize=(155 * MM, 100 * MM))
    render_importance_pie(fig, vals_uv, lbls_uv, colors_uv,
                          "Feature Importance by Anatomical Region")
    _save_fig(fig, "特征重要性_按解剖分区")

    # ── Figure 3: Measurement type ──
    logger.info("\n[Figure 3] Measurement type")
    df_mt = mt.sort_values("importance", ascending=False)
    lbls_mt = df_mt["new_group"].values
    vals_mt = df_mt["importance"].values
    colors_mt = [C_MEASURE_COLORS.get(label, C_CLINICAL) for label in lbls_mt]
    fig = plt.figure(figsize=(155 * MM, 100 * MM))
    render_importance_pie(fig, vals_mt, lbls_mt, colors_mt,
                          "Feature Importance by Measurement Type")
    _save_fig(fig, "特征重要性_按测量类型")

    logger.info(f"\nAll figures saved to {EXPORT_SHAP_DIR}/")


if __name__ == "__main__":
    main()
