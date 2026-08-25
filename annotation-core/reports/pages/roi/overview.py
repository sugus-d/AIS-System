"""ROI 概览页 — 全部 subject 表格 + 汇总统计。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .data import clear_cache, load_eval, load_regions, subject_list

PIPELINE = "BFS(r=0.20) → 边界侵蚀(3) → 混合裤子切割"
REGION_ORDER = ["neck", "hem", "side_L", "side_R"]
REGION_LABELS = {"neck": "脖子", "hem": "下摆", "side_L": "左侧", "side_R": "右侧"}
EXCESS_OK_RATIO = 0.08  # 多余率正常阈值
LEGEND_BASE_FIELDS = 4  # 图例基础字段数（不含平均值）


def _style(v: float, t1: float, t2: float) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "color: #999"
    av = abs(v)
    return "" if av < t1 else ("color: #ca8a04" if av < t2 else "color: #dc2626; font-weight: bold")


def show() -> None:
    st.title("ROI 提取评估 — 87 例全量验证")
    st.caption(f"管线：{PIPELINE}")

    eval_data = load_eval()
    region_data = load_regions()
    subjects = subject_list()
    has_eval = len(eval_data) > 0

    # ── 汇总行 ──
    metrics = {"总数": len(subjects), "含GT": f"{sum(s['has_gt'] for s in subjects)}/{len(subjects)}"}
    if has_eval:
        covs = [eval_data[s["subject"]]["coverage"] for s in subjects if s["subject"] in eval_data]
        excs = [eval_data[s["subject"]]["excess"] for s in subjects if s["subject"] in eval_data]
        if covs:
            metrics["平均覆盖率"] = f"{np.mean(covs):.1%}"
            metrics["平均多余率"] = f"{np.mean(excs):.1%}"
            metrics["多余率<8%"] = f"{sum(1 for e in excs if e < EXCESS_OK_RATIO)}/{len(excs)}"

    cols = st.columns(len(metrics))
    for i, (k, v) in enumerate(metrics.items()):
        cols[i].metric(k, v)

    st.button("🔄 重算全部 GT 指标", type="primary", on_click=clear_cache)

    st.divider()
    st.subheader("全部 subject 明细")

    # ── 图例 ──
    legend_data = {
        "覆盖率": ("GT 面质心在算法 5mm 内", "≥90%", "80~90%", "<80%"),
        "多余率": ("算法结果离 GT >5mm", "<8%", "8~15%", "≥15%"),
        "Chamfer": ("双向边界距离", "<5mm", "5~10mm", "≥10mm"),
        "区域 Δtri": ("与 GT 三角面数偏差", "|x|<10%", "10~25%", "|x|≥25%"),
    }
    if has_eval:
        for k in legend_data:
            vals = [eval_data[s["subject"]]["coverage"] for s in subjects if s["subject"] in eval_data] if k == "覆盖率" else (
                   [eval_data[s["subject"]]["excess"] for s in subjects if s["subject"] in eval_data] if k == "多余率" else (
                   [eval_data[s["subject"]].get("chamfer", float("nan")) for s in subjects if s["subject"] in eval_data] if k == "Chamfer" else []))
            legend_data[k] = (*legend_data[k], f"{np.mean(vals):.1%}" if vals and k in ("覆盖率", "多余率") else f"{np.nanmean(vals):.1f}mm" if vals and k == "Chamfer" else "—")

    st.dataframe(
        pd.DataFrame([{"指标": k, "说明": v[0], "正常": v[1], "警告": v[2], "超标": v[3], "平均值": v[4] if len(v) > LEGEND_BASE_FIELDS else "—"} for k, v in legend_data.items()]).set_index("指标"),
        width="stretch",
    )

    # ── 明细表 ──
    rows = []
    for s in subjects:
        sid = s["subject"]
        ev = eval_data.get(sid, {})
        reg = region_data.get(sid, {})
        rows.append({
            "Subject": sid,
            "覆盖率": ev.get("coverage", float("nan")),
            "多余率": ev.get("excess", float("nan")),
            "Chamfer": ev.get("chamfer", float("nan")),
            "颈": reg.get("neck_delta_pct", float("nan")),
            "摆": reg.get("hem_delta_pct", float("nan")),
            "左": reg.get("side_L_delta_pct", float("nan")),
            "右": reg.get("side_R_delta_pct", float("nan")),
        })

    styled = pd.DataFrame(rows).style \
        .format({"覆盖率": "{:.1%}", "多余率": "{:.1%}", "Chamfer": "{:.1f}",
                 "颈": "{:+.1f}%", "摆": "{:+.1f}%", "左": "{:+.1f}%", "右": "{:+.1f}%"}) \
        .map(lambda v: _style(v, 0.9, 0.8), subset=["覆盖率"]) \
        .map(lambda v: _style(v, 0.08, 0.15), subset=["多余率"]) \
        .map(lambda v: _style(v, 5, 10), subset=["Chamfer"]) \
        .map(lambda v: _style(v, 10, 25), subset=["颈", "摆", "左", "右"])

    selection = st.dataframe(
        styled,
        column_config={"Subject": st.column_config.TextColumn("Subject")},
        on_select="rerun",
        selection_mode="single-row",
        hide_index=True,
        use_container_width=True,
        height=1056,
    )

    if selection and len(selection.selection.rows) > 0:
        idx = selection.selection.rows[0]
        if idx < len(subjects):
            st.query_params["subject"] = subjects[idx]["subject"]
            st.rerun()
