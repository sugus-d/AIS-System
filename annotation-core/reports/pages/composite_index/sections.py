"""Composite Index 分组详情配置与渲染。"""

from __future__ import annotations

import streamlit as st

from .data import COMPOSITE_DIR

CV5_GOOD = 0.6  # cv5_r 良好阈值
CV5_PASS = 0.4  # cv5_r 通过阈值

SECTION_CONFIG = [
    {"id": "height", "title": "📏 高度 Height", "prefix": "height"},
    {"id": "mean_curv", "title": "🌀 平均曲率 Mean Curv", "prefix": "mean_curv"},
    {"id": "gauss_curv", "title": "🌋 高斯曲率 Gauss Curv", "prefix": "gauss_curv"},
    {"id": "roughness", "title": "🏔️ 粗糙度 Roughness", "prefix": "roughness"},
    {"id": "normal_angle", "title": "📐 法向量夹角 Normal Angle", "prefix": "normal_angle"},
    {
        "id": "normal_vector",
        "title": "🧭 法向量矢量差 Vector",
        "prefix": "normal_vector",
        "exclude_suffixes": ["cos", "sin"],
    },
    {"id": "normal_vector_cos", "title": "🧮 法向量 Cos", "prefix": "normal_vector_cos"},
]


def render_sections(data: list[dict], mode: str) -> None:
    fig_dir = COMPOSITE_DIR / ("figures_compressed" if mode == "compressed" else "figures")
    tabs = st.tabs([sec["title"] for sec in SECTION_CONFIG])

    for tab, sec in zip(tabs, SECTION_CONFIG, strict=False):
        with tab:
            items = [d for d in data if d["group"].startswith(sec["prefix"])]
            excl = sec.get("exclude_suffixes", [])
            if excl:
                items = [d for d in items if not any(d["group"].endswith(f"_{s}") for s in excl)]
            if not items:
                st.info("无数据")
                continue

            for item in items:
                cv5 = item["cv5_r"]
                tag = "🟢" if cv5 > CV5_GOOD else ("🟡" if cv5 > CV5_PASS else "🔴")

                with st.container(border=True):
                    cols = st.columns([1.2, 2.5])
                    with cols[0]:
                        img_path = fig_dir / f"{item['group']}.png"
                        if img_path.exists():
                            st.image(str(img_path))
                        else:
                            st.caption("无配图")
                    with cols[1]:
                        st.markdown(f"**{item['group']}** {tag}")
                        mc = st.columns(4)
                        mc[0].metric("特征数", item["n_feats"])
                        mc[1].metric("cv5_r", f"{cv5:.4f}")
                        mc[2].metric("full_r", f"{item['full_r']:.4f}")
                        mc[3].metric("gap", f"{item['gap']:.4f}")
                        if item.get("formula"):
                            with st.expander("公式"):
                                st.code(item["formula"], language="text")
