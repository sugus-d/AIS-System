"""ROI 提取评估页面 — 87 例全量验证。"""

from __future__ import annotations

import streamlit as st

from reports.pages.roi.detail import show as show_detail
from reports.pages.roi.overview import show as show_overview


def show() -> None:
    target = st.query_params.get("subject", "")
    if target:
        show_detail(target)
    else:
        show_overview()


if __name__ == "__main__":
    show()
