"""Streamlit 报告入口 — 使用原生页面导航。

子页面放在 pages/ 目录，自动发现，侧边栏原生导航。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAGES_DIR = Path(__file__).resolve().parent / "pages"


def main():
    st.set_page_config(page_title="AIS 评估报告", layout="wide")

    # 全宽布局 CSS
    st.markdown("""
        <style>
        .main > div, .block-container, .stApp > header, section.main > div {
            max-width: 100% !important;
            width: 100% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        section[data-testid="stSidebarContent"] + div section.main {
            max-width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # 自动发现 pages/ 下的页面（支持单页 xxx.py 和模块 xxx/app.py）
    page_entries: list[Path] = []
    for p in sorted(PAGES_DIR.iterdir()):
        if p.suffix == ".py" and p.stem != "__init__":
            page_entries.append(p)
        elif p.is_dir() and (p / "app.py").exists():
            page_entries.append(p / "app.py")
    if not page_entries:
        st.error("pages/ 目录下未发现页面文件。")
        return

    pages = []
    for pf in page_entries:
        if pf.parent.stem != "pages":
            # 子目录模块，用目录名做标题 + URL path
            title = pf.parent.stem.replace("_", " ").title()
            url_path = pf.parent.stem
        else:
            title = pf.stem.replace("_", " ").title()
            url_path = pf.stem
        pages.append(st.Page(str(pf), title=title, url_path=url_path))

    pg = st.navigation(pages, position="sidebar")
    pg.run()


if __name__ == "__main__":
    main()
