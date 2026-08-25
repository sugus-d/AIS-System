"""Cobb 角预测 — 模型对比报告。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from modeling.training.result_paths import scheme_results_path
from reports.pages.model_evaluation.data import (
    load_best_results,
    load_ensemble_results,
    load_latest_schemes,
    load_scheme_results,
)
from reports.pages.model_evaluation.render import render_ensemble, render_singles
from reports.pages.model_evaluation.style import CLINICAL
from visualization.evaluation_panels import render_confusion_matrix_4class
from visualization.scatter_plot import (
    render_scatter_3class,
    render_scatter_4class,
    SEV3_BINS,
    SEV3_LABELS,
)

SEVERITY_LABELS = ["Normal", "Mild", "Moderate", "Severe"]
TOP_N = 5


def _compute_3class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """从连续预测值计算 3 分类 (0-20°, 20-40°, 40+°) 指标。"""
    tc_true = np.digitize(y_true, SEV3_BINS)
    tc_pred = np.digitize(y_pred, SEV3_BINS)

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(tc_true, tc_pred, strict=False):
        cm[t, p] += 1

    total = len(y_true)
    correct = (tc_true == tc_pred).sum()
    accuracy = correct / total

    per_class = {}
    for i, label in enumerate(SEV3_LABELS):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[label] = {"precision": round(prec, 4), "recall": round(rec, 4),
                            "f1": round(f1, 4), "support": int(cm[i, :].sum())}

    # Macro F1
    macro_f1 = np.mean([v["f1"] for v in per_class.values()])

    return {
        "macro_f1": round(macro_f1, 4),
        "total_accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def _render_detail_expander(rank: int, scheme_name: str, model_name: str,
                            best_model: dict, pred_path: str, tab_label: str) -> None:
    """渲染一个模型的详情面板（4 类 + 3 类指标和图表）。"""
    pc4 = best_model.get("per_class", {})
    cm4 = best_model.get("confusion_matrix", [])

    prefix = f"#{rank}  " if rank else ""
    with st.expander(f"{prefix}{tab_label}  {scheme_name} / {model_name}  "
                     f"Macro-F1={best_model['macro_f1']:.4f}  "
                     f"RMSE={best_model['rmse']:.1f}°", expanded=rank == 1):
        # ── 4 类指标 ──
        st.markdown("**4 分类 (Normal/Mild/Moderate/Severe)**")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Macro-F1", f"{best_model['macro_f1']:.4f}")
        mc2.metric("RMSE", f"{best_model['rmse']:.2f}°")
        mc3.metric("Accuracy", f"{best_model['total_accuracy']:.3f}")
        mc4.metric("r (w/ Cobb)", f"{best_model.get('r', 0):.3f}")

        cols = st.columns(4)
        for i, label in enumerate(SEVERITY_LABELS):
            p = pc4.get(label, {})
            with cols[i]:
                st.metric(f"{label} F1", f"{p.get('f1', 0):.3f}")
                st.caption(f"prec={p.get('precision', 0):.3f}  rec={p.get('recall', 0):.3f}")

        # 4 类图表：散点图 + 混淆矩阵
        p = Path(pred_path)
        if p.exists():
            pred_df = pd.read_csv(p)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
            render_scatter_4class(ax1, pred_df["max_cobb_true"].values,
                                  pred_df["max_cobb_pred"].values,
                                  pred_df["class_true"].values,
                                  pred_df["class_pred"].values,
                                  title=f"{model_name} (4-class)")
            if cm4:
                render_confusion_matrix_4class(ax2, cm4, SEVERITY_LABELS)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            plt.close("all")

        # ── 3 类指标 ──
        st.markdown("---")
        st.markdown("**3 分类 (0-20° / 20-40° / 40+°)**")
        if p.exists():
            pred_df = pd.read_csv(p)
            y_t = pred_df["max_cobb_true"].values
            y_p = pred_df["max_cobb_pred"].values
            m3 = _compute_3class_metrics(y_t, y_p)
            cm3 = m3["confusion_matrix"]
            pc3 = m3["per_class"]

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Macro-F1", f"{m3['macro_f1']:.4f}")
            mc2.metric("Accuracy", f"{m3['total_accuracy']:.3f}")
            mc3.metric("RMSE", f"{best_model.get('rmse', 0):.2f}°")

            cols = st.columns(3)
            for i, label in enumerate(SEV3_LABELS):
                p = pc3.get(label, {})
                with cols[i]:
                    st.metric(f"{label} F1", f"{p.get('f1', 0):.3f}")
                    st.caption(f"prec={p.get('precision', 0):.3f}  rec={p.get('recall', 0):.3f}")

            # 3 类图表：散点图 + 混淆矩阵
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
            render_scatter_3class(ax1, y_t, y_p, title=f"{model_name} (3-class)")
            render_confusion_matrix_4class(ax2, cm3, SEV3_LABELS)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            plt.close("all")


def _top5_details(df: pd.DataFrame, schemes: dict, pred_base: str, tab_label: str) -> None:
    """渲染 Top 5 方案详情。"""
    seen = set()
    top5 = []
    for _, row in df.iterrows():
        key = (row["方案"], row["模型"])
        if key not in seen:
            seen.add(key)
            top5.append(row)
        if len(top5) >= TOP_N:
            break

    for rank, row in enumerate(top5, 1):
        sn, mn = row["方案"], row["模型"]
        best_model = None
        for sname, models in schemes.items():
            if sname == sn:
                for m in models:
                    if m["model"] == mn:
                        best_model = m
                        break
        if best_model is None:
            continue
        pred_path = f"{pred_base}/{sn}/{mn}/predictions.csv"
        _render_detail_expander(rank, sn, mn, best_model, pred_path, tab_label)


def render_scheme_table(schemes: dict, title: str, pred_base: str) -> None:
    """通用的方案对比渲染。"""
    st.subheader(title)

    all_rows = []
    for sname, models in schemes.items():
        for m in models:
            pc = m.get("per_class", {})
            all_rows.append({
                "方案": sname, "模型": m["model"],
                "Macro-F1": m["macro_f1"], "RMSE": m["rmse"],
                "Normal-F1": pc.get("Normal", {}).get("f1", 0),
                "Severe-F1": pc.get("Severe", {}).get("f1", 0),
            })
    if not all_rows:
        st.info(f"暂无 {title} 训练结果")
        return

    df = pd.DataFrame(all_rows).sort_values("Macro-F1", ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 最佳方案", df.iloc[0]["方案"])
    c2.metric("最佳模型", df.iloc[0]["模型"])
    c3.metric("Macro-F1", f"{df.iloc[0]['Macro-F1']:.4f}")
    c4.metric("RMSE", f"{df.iloc[0]['RMSE']:.2f}°")

    st.dataframe(df.style.format("{:.4f}", subset=["Macro-F1"])
                 .format("{:.2f}", subset=["RMSE"])
                 .format("{:.3f}", subset=["Normal-F1", "Severe-F1"])
                 .highlight_max(subset=["Macro-F1"]), use_container_width=True)

    st.subheader("Top 5 方案详情")
    _top5_details(df, schemes, pred_base, title.split("—")[0].strip())


def show():
    st.title("AIS Cobb 角预测 — 模型对比")
    st.caption(f"5×5-fold CV · 阈值 {CLINICAL:.0f}° · piecewise_log48 · 独立内层 HP 搜索")

    singles = load_best_results()
    ens = load_ensemble_results()
    split_results = load_scheme_results("results_schemeB_split.json")
    small_results = load_scheme_results("results_schemeB_small.json")
    dual_40d = load_latest_schemes(scheme_results_path("morph_region_ci_40d"))
    dual_27d = load_latest_schemes(scheme_results_path("morph_region_ci_27d"))
    ensemble = load_latest_schemes(scheme_results_path("ensemble_composite_v7_ai60"))
    latest = load_latest_schemes(scheme_results_path("train_back_v1"))

    tabs = []
    if ensemble:
        tabs.append("集成 CompositeV7+AI (α=0.60)")
    if dual_40d:
        tabs.append("morph_region_ci_40d")
    if dual_27d:
        tabs.append("morph_region_ci_27d")
    if latest:
        tabs.append("Back v1 最新")
    if singles:
        tabs.append("旧 pipeline")
    if split_results:
        tabs.append("schemeB-split 45D")
    if small_results:
        tabs.append("schemeB-small 31D")
    if singles:
        tabs.append("集成模型")

    if not tabs:
        st.info("暂无训练结果")
        return

    tab_views = st.tabs(tabs)
    idx = 0

    if ensemble:
        with tab_views[idx]:
            render_scheme_table(ensemble, "集成 CompositeV7+AI (α=0.60) — MAE=4.53  r=0.85", str(scheme_results_path("ensemble_composite_v7_ai60")))
        idx += 1
    if dual_40d:
        with tab_views[idx]:
            render_scheme_table(dual_40d, "morph_region_ci_40d — 最佳结果 (MF1=0.745)", str(scheme_results_path("morph_region_ci_40d")))
        idx += 1
    if dual_27d:
        with tab_views[idx]:
            render_scheme_table(dual_27d, "morph_region_ci_27d — 方案对比", str(scheme_results_path("morph_region_ci_27d")))
        idx += 1
    if latest:
        with tab_views[idx]:
            render_scheme_table(latest, "Back v1 最新 — 方案对比", str(scheme_results_path("train_back_v1")))
        idx += 1
    if singles:
        with tab_views[idx]:
            render_singles(singles, "旧 pipeline")
        idx += 1
    if split_results:
        with tab_views[idx]:
            render_singles(split_results, "schemeB-split 45D")
        idx += 1
    if small_results:
        with tab_views[idx]:
            render_singles(small_results, "schemeB-small 31D")
        idx += 1
    if singles:
        with tab_views[idx]:
            render_ensemble(ens, singles)


if __name__ == "__main__":
    show()
