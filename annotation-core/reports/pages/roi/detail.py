"""ROI 详情页 — 单个 subject 的指标 + 对比图。"""

from __future__ import annotations

import numpy as np
import streamlit as st

from .comparison import render
from .data import _ply_header, EXPORT_DIR, gt_path, load_eval, load_regions, roi_path

REGION_ORDER = ["neck", "hem", "side_L", "side_R"]
REGION_LABELS = {"neck": "脖子", "hem": "下摆", "side_L": "左侧", "side_R": "右侧"}
PIPELINE = "BFS(r=0.20) → 边界侵蚀(3) → 混合裤子切割"
DELTA_OK = 10  # 区域偏差正常阈值（%）
DELTA_WARN = 25  # 区域偏差警告阈值（%）


def show(sid: str) -> None:
    eval_data = load_eval()
    region_data = load_regions()
    ev = eval_data.get(sid, {})
    reg = region_data.get(sid, {})
    has_eval = bool(ev) and not ev.get("error")
    has_gt = gt_path(sid).exists()

    # ── 标题行 ──
    oc = _ply_header(EXPORT_DIR / sid / "original.ply")
    rc = _ply_header(roi_path(sid))
    ov = oc[0] if oc else 0
    rv = rc[0] if rc else 0

    st.title(f"ROI 详情 — {sid}")
    st.caption(f"{PIPELINE}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("原始顶点", f"{ov:,}")
    col2.metric("ROI 顶点", f"{rv:,}")
    col3.metric("保留率", f"{rv / max(ov, 1):.1%}")

    if has_eval:
        ch = ev.get("chamfer", float("nan"))
        col4.metric("Chamfer", f"{ch:.1f}mm" if isinstance(ch, float) and not np.isnan(ch) else "—")
    else:
        col4.metric("GT", "有" if has_gt else "无")

    # ── 核心指标 ──
    if has_eval:
        st.subheader("评估指标")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("覆盖率", f"{ev.get('coverage', 0):.1%}")
        mc2.metric("多余率", f"{ev.get('excess', 0):.1%}")
        mc3.metric("Chamfer", f"{ch:.1f}mm")
        mc4.metric("GT 三角面", f"{ev.get('gt_t', 0):,}")

        # ── 区域 Δtri ──
        keys = [(f"{r}_delta_pct", REGION_LABELS.get(r, r)) for r in REGION_ORDER if reg.get(f"{r}_delta_pct") is not None]
        if keys:
            st.subheader("区域偏差 Δtri")
            c = st.columns(len(keys))
            for col, (k, lbl) in zip(c, keys, strict=False):
                dp = reg[k]
                emoji = "🟢" if abs(dp) < DELTA_OK else ("🟡" if abs(dp) < DELTA_WARN else "🔴")
                col.metric(lbl, f"{emoji} {dp:+.1f}%")
    else:
        if not has_gt:
            st.warning("该 subject 无 GT 数据")

    # ── 对比图 ──
    img = render(sid)
    if img:
        st.image(str(img))
    else:
        st.error("无法生成对比图（原始网格或算法结果缺失）")

    st.button("← 返回全部 subject", on_click=lambda: (_ := st.query_params.clear(), st.rerun()))


