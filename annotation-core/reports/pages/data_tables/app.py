"""论文表格展示 — 数据加载 + 渲染分离。"""

from __future__ import annotations

import streamlit as st

from reports.pages.data_tables.data import load_all
from reports.pages.data_tables.render import (
    render_table1,
    render_table2,
    render_table3,
    render_table4,
    render_table5,
    render_table6,
)


def _render(title: str, html: str) -> None:
    st.subheader(title)
    st.markdown(html, unsafe_allow_html=True)


def show():
    st.title("Paper Data Tables")
    d = load_all()

    inv = d.get("inventory")
    if inv is not None:
        valid = inv[inv["has_landmark"]]
        sev = valid["severity"].value_counts()
        st.subheader("Data Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(valid))
        ais = sum(sev.get(s, 0) for s in ["Mild", "Moderate", "Severe"])
        c2.metric("AIS / Normal", f"{ais} / {sev.get('Normal', 0)}")
        ct = valid["curve_type"].value_counts()
        c3.metric("Thoracic / Lumbar / Double",
                  f"{ct.get('Thoracic', 0)} / {ct.get('Lumbar', 0)} / {ct.get('Double', 0)}")

    # Formulas
    st.subheader("Asymmetry Index Formulas")
    cols = st.columns(3)
    for col, label, key in [
        (cols[0], "Asymmetric Index (AI)", "ai_formula"),
        (cols[1], "Curvature Index", "curvature_formula"),
        (cols[2], "Height Index", "height_formula"),
        (cols[3], "Normal Vector Index", "nvi_formula"),
        (cols[4], "Roughness Index", "ri_formula"),
    ]:
        fm = d.get(key)
        with col:
            if fm and fm.get("feats"):
                st.markdown(f"**{label}**  (r={fm.get('final_r', fm.get('r', 0)):.4f})")
                for f, c in zip(fm["feats"], fm["coefs"], strict=False):
                    st.caption(f"{'+' if c >= 0 else ''}{c:.4f} × {f}")

    # Tables
    _render("Table 1. Demographic and clinical characteristics", render_table1(d["table1"], d.get("clinical")))
    _render("Table 2. Extraction of Cosmetic Parameters", render_table2(d.get("table2")))
    _render("Table 3. Core Asymmetry Indices by AIS severity", render_table3(d["table3"]))
    _render("Table 4. Correlation: Indices vs Cobb Angle", render_table4(d["table4"]))
    _render("Table 5. Cobb angle prediction performance", render_table5(d["table5"]))
    _render("Table 6. Classification Performance", render_table6(d["table6"]))


if __name__ == "__main__":
    show()
