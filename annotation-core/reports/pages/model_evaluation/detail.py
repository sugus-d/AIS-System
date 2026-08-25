"""统一的模型详情渲染组件。

垂直布局：散点图 → 混淆矩阵 → 指标表 → 参数表。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from visualization.scatter_plot import render_scatter


def _kv_df(items: list[tuple[str, str]]) -> pd.DataFrame:
    """(key, value) 列表转两列 DataFrame。"""
    return pd.DataFrame(items, columns=["项目", "值"])


def render_model_detail(
    label: str,
    model_data: dict,
    y_true: np.ndarray,
    ensemble_weights: list[tuple[str, float]] | None = None,
) -> None:
    """统一的模型详情面板 - 1图+3表 垂直布局。"""
    # ── 1. 散点图 ──
    st.markdown("**散点图**")
    preds = np.array(model_data.get("preds", []))
    if len(preds) == len(y_true):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        render_scatter(
            ax, y_true, preds,
            cm=model_data.get("cm", [0, 0, 0, 0]),
            threshold=20.0, title=label, show_fit_line=True,
        )
        st.pyplot(fig)
        plt.close(fig)
        plt.close("all")
    else:
        st.caption("预测数据不完整")

    # ── 2. 混淆矩阵 ──
    st.markdown("**混淆矩阵**")
    cm = model_data.get("cm", [0, 0, 0, 0])
    st.dataframe(
        pd.DataFrame(
            [[cm[0], cm[1]], [cm[2], cm[3]]],
            index=["正常 (≤20°)", "异常 (>20°)"],
            columns=["预测正常", "预测异常"],
        ),
        use_container_width=True,
    )

    # ── 3. 指标表 ──
    st.markdown("**指标**")
    st.dataframe(
        _kv_df([
            ("F1", f"{model_data.get('f1', 0):.3f}"),
            ("Sens", f"{model_data.get('sens', 0):.3f}"),
            ("Spec", f"{model_data.get('spec', 0):.3f}"),
            ("RMSE", f"{model_data.get('rmse', 0):.1f}°"),
            ("r", f"{model_data.get('r', 0):.3f}"),
        ]),
        use_container_width=True, hide_index=True,
    )

    # ── 4. 参数表 ──
    st.markdown("**参数**")
    bp = model_data.get("best_params", {})
    if bp:
        items = []
        for k, v in bp.items():
            if isinstance(v, float):
                items.append((k, f"{v:.4f}"))
            else:
                items.append((k, str(v)))
        st.dataframe(_kv_df(items), use_container_width=True, hide_index=True)
    else:
        st.markdown("集成模型，无独立参数")

    # ── 5. 集成权重（可选）──
    if ensemble_weights:
        st.markdown("**子模型权重**")
        wdf = pd.DataFrame(ensemble_weights, columns=["模型", "权重"])
        wdf["权重"] = wdf["权重"].apply(lambda x: f"{x:.4f}")
        st.dataframe(wdf, use_container_width=True, hide_index=True)
