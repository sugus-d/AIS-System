"""模型评估渲染 — 单模型 / 集成模型表格与详情。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .data import load_y_true
from .detail import render_model_detail
from .style import F1_THRESHOLD, passes, SENS_THRESHOLD, SPEC_THRESHOLD, style_table


def render_singles(results: list[dict], label: str = "单模型") -> None:
    st.subheader(f"{label}（{len(results)} 个）")
    y_true = load_y_true()

    passed = sum(1 for r in results if passes(r))
    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 最佳 F1", f"{max(results, key=lambda r: r['f1'])['f1']:.3f}")
    c2.metric("✅ 三项达标", f"{passed}/{len(results)}",
              help=f"F1≥{F1_THRESHOLD} & Sens≥{SENS_THRESHOLD} & Spec≥{SPEC_THRESHOLD}")
    c3.metric("📉 最低 RMSE", f"{min(results, key=lambda r: r['rmse'])['rmse']:.1f}°")

    sorted_r = sorted(results, key=lambda r: (not passes(r), -r.get("f1", 0)))
    rows = [
        {
            "算法": r["algo"], "达标": "✅" if passes(r) else "❌",
            "F1": round(r["f1"], 3), "Sens": round(r["sens"], 3),
            "Spec": round(r["spec"], 3), "RMSE": round(r["rmse"], 2),
            "r": round(r.get("r", 0), 3),
        }
        for r in sorted_r
    ]
    st.dataframe(style_table(pd.DataFrame(rows)).hide(axis="index"), width="stretch")

    st.subheader("模型详情")
    for r in sorted_r:
        flag = "✅" if passes(r) else "⚠️"
        with st.expander(f"{flag} {r['algo']}"):
            render_model_detail(r["algo"], r, y_true)


def render_ensemble(ens: list[dict], singles: list[dict]) -> None:
    if not ens:
        st.info("集成模型尚未运行")
        return

    st.subheader("集成模型 vs 最佳单模型")
    sorted_ens = sorted(ens, key=lambda r: (not passes(r), -r.get("f1", 0)))

    rows = []
    for r in sorted_ens:
        rows.append({
            "算法": r["algo"], "达标": "✅" if passes(r) else "❌",
            "F1": round(r["f1"], 3), "Sens": round(r["sens"], 3),
            "Spec": round(r["spec"], 3), "RMSE": round(r["rmse"], 2),
            "r": round(r.get("r", 0), 3),
        })
    if singles:
        best = max(singles, key=lambda r: r["f1"])
        rows.append({
            "算法": f"★ {best['algo']}（最佳单模型）",
            "达标": "✅" if passes(best) else "❌",
            "F1": round(best["f1"], 3), "Sens": round(best["sens"], 3),
            "Spec": round(best["spec"], 3), "RMSE": round(best["rmse"], 2),
            "r": round(best.get("r", 0), 3),
        })
    st.dataframe(style_table(pd.DataFrame(rows)).hide(axis="index"), width="stretch")

    y_true = load_y_true()
    if y_true is not None and len(y_true) > 0:
        st.subheader("集成模型详情")
        for r in sorted_ens:
            flag = "🤝" if passes(r) else "⚠️"
            with st.expander(f"{flag} {r['algo']}"):
                render_model_detail(r["algo"], r, y_true)
