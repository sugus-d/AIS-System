"""Composite Index（压缩/调优版）计算。

加载预训练的 Ridge 回归系数（来自 ``search_orchestrator.py`` 的搜索结果），
对指定受试者的特征做标准化后线性组合得到 Composite Index。

支持两种模式：
- 原始模式：直接标准化后线性组合。
- 压缩模式（tuned）：对目标值做分段对数压缩后拟合，最终 CI 在压缩空间计算。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 公式加载
# ---------------------------------------------------------------------------


def load_formulas(path: str) -> dict[str, tuple[list[str], list[float]]]:
    """从 CSV 文件加载 Composite Index 公式。

    CSV 格式（与 ``search_orchestrator.py`` 输出一致）：
        group, feats, coefs, n_feats, ...
    其中 feats 和 coefs 是以 ``|`` 分隔的字符串。

    Args:
        path: CSV 文件路径（如 ``results/modeling/composite/results_compressed.csv``）。

    Returns:
        {group: (feat_names_list, coefs_list)} 的字典。
    """
    formulas: dict[str, tuple[list[str], list[float]]] = {}
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        if r["n_feats"] == 0:
            continue
        feats = r["feats"].split("|")
        coefs = [float(x) for x in r["coefs"].split("|")]
        formulas[r["group"]] = (feats, coefs)
    return formulas


# ---------------------------------------------------------------------------
# CI 计算
# ---------------------------------------------------------------------------


def _parse_group(group: str) -> tuple[str, str]:
    """从 group 名解析 (measure, method)。

    ``normal_angle_dm`` → (``normal_angle``, ``dm``)
    ``height_pw`` → (``height``, ``pw``)
    """
    method = "pw" if group.endswith("_pw") else "dm"
    measure = group.replace(f"_{method}", "")
    return measure, method


def compute_ci(
    subject_id: str,
    group: str,
    feats: list[str],
    coefs: list[float],
    features_df: pd.DataFrame,
) -> float:
    """计算单个 Composite Index。

    Args:
        subject_id: 受试者 ID。
        group: 公式组名（如 ``normal_angle_dm``）。
        feats: 区域名列表（如 ``["nr_p0_p1", "st_p0_p2"]``）。
        coefs: 对应 Ridge 回归系数。
        features_df: 全量特征 DataFrame（列如 ``{name}_normal_angle``）。

    Returns:
        Composite Index 标量值。
    """
    measure, method = _parse_group(group)
    suffix = f"_{measure}__pw" if method == "pw" else f"_{measure}"

    row = features_df[features_df["subject_id"] == subject_id]
    if len(row) == 0:
        return 0.0

    vals: list[float] = []
    for fn in feats:
        col = f"{fn}{suffix}"
        if col in row.columns:
            v = row[col].values[0]
            vals.append(v if np.isfinite(v) else 0.0)
        else:
            vals.append(0.0)

    if not vals:
        return 0.0

    X = np.array(vals, dtype=np.float64).reshape(1, -1)
    Xs = (X - np.mean(X)) / np.maximum(np.std(X), 1e-8)
    return float(Xs @ np.array(coefs))


def compute_all_ci(
    formulas: dict[str, tuple[list[str], list[float]]],
    subject_ids: list[str],
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """计算多个受试者在多组公式下的全部 Composite Index。

    Args:
        formulas: ``load_formulas()`` 的结果。
        subject_ids: 受试者 ID 列表。
        features_df: 全量特征 DataFrame。

    Returns:
        DataFrame，每行一个受试者，列为 ``subject_id`` + 各 group 的 CI。
    """
    rows: list[dict[str, float]] = []
    for sid in subject_ids:
        row: dict[str, float] = {"subject_id": sid}
        for group, (feats, coefs) in formulas.items():
            row[group] = compute_ci(sid, group, feats, coefs, features_df)
        rows.append(row)
    return pd.DataFrame(rows)
