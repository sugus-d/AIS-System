"""Composite Index 搜索结果报告。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from reports.pages.composite_index.data import load_data
from reports.pages.composite_index.sections import render_sections
from reports.pages.composite_index.style import fmt_table

CV5_PASS = 0.4  # cv5_r 通过阈值


def show() -> None:
    st.title("Composite Index 搜索结果")
    st.markdown(
        "16 组独立搜索 · 重复 5 次 5-fold CV · 每组 3–10 特征 · "
        "样本量 N=60 · 特征维度 3,600 · P95=48° 压缩阈值"
    )

    mode = st.radio(
        "模式",
        ["baseline", "compressed"],
        horizontal=True,
        format_func=lambda x: {"baseline": "基线", "compressed": "压缩 (48°+log)"}.get(x, x),
    )

    data = load_data(mode)
    if not data:
        st.warning("数据未找到")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("总组数", len(data))
    passed = sum(1 for d in data if d["cv5_r"] > CV5_PASS)
    col2.metric("通过 (cv5_r > 0.4)", f"{passed}/{len(data)}")
    col3.metric("平均 gap", f"{float(np.mean([d['gap'] for d in data])):.4f}")

    st.subheader("全量结果表")
    df = pd.DataFrame(data).rename(
        columns={"group": "Group", "n_feats": "特征数", "cv5_r": "cv5_r", "full_r": "full_r", "gap": "gap"}
    )
    st.dataframe(fmt_table(df.set_index("Group")), width="stretch")

    st.subheader("分组详情")
    render_sections(data, mode)


if __name__ == "__main__":
    show()
