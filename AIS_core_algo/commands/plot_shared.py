"""编排层共享辅助 — 图形保存、连通分量拆分、组件分类、面板数据准备。

被 commands/plot_*.py 各编排脚本复用，避免重复实现。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.logger import logger


def _save_figure(
    fig: plt.Figure, out_path: str | Path, facecolor: str = "white", pad_inches: float | None = None
) -> None:
    """统一保存 figure：dpi/裁剪参数一致，保存后关闭并记录日志。"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=pad_inches, facecolor=facecolor)
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


def _split_into_full_components(
    removed_tri_indices: list[int],
    original_triangles: np.ndarray,
    full_adjacency: list[set[int]] | None = None,
) -> list[list[int]]:
    """将切除的三角面按连通性拆分，每个连通分量作为一个独立组件返回。

    组件拆分后，每个组件可以独立判断是否为有效切除（valid），避免将整个切除区域
    一刀切地标记为有效或无效——同一切除操作的不同连通分量可能有不同分类。
    """
    from mesh.roi._mesh_graph import build_triangle_adjacency, find_connected_components

    if not removed_tri_indices:
        return []
    removed_set = set(removed_tri_indices)
    if full_adjacency is None:
        full_adjacency = build_triangle_adjacency(original_triangles)
    local_to_global = list(removed_set)
    global_to_local = {g: li for li, g in enumerate(local_to_global)}
    local_adj: list[set[int]] = []
    for gidx in local_to_global:
        local_adj.append({global_to_local[n] for n in full_adjacency[gidx] if n in removed_set})
    raw = find_connected_components(local_adj)
    return [[local_to_global[li] for li in comp] for comp in raw]


def _classify_component_by_analysis(
    component: list[int],
    original_triangles: np.ndarray,
    analysis_removals: list[dict],
) -> bool:
    """判断一个切除连通分量是否属于有效切除。

    遍历所有 analysis 中标记为 valid 的切除操作，若该分量中有三角面出现在
    任一 valid 切除的 triangle_indices 列表中，则该分量被视为有效切除。
    """
    comp_set = set(component)
    for rem in analysis_removals:
        if not rem.get("valid"):
            continue
        for ti in rem.get("triangle_indices", []):
            if ti in comp_set:
                return True
    return False


def _prepare_panel_data(
    original_triangles: np.ndarray,
    analysis: dict,
    removed_tris: list[int],
) -> dict:
    """准备面板数据：将切除三角面拆分为有效区域（valid）和回填区域（invalid）。

    有效区域 = 与 analysis 中 valid 切除操作有交集的连通分量
    无效区域 = analysis.restored_tris（被回填的三角面）
    返回 dict 含 valid_tris 和 invalid_tris 两个 set。
    """
    from mesh.roi._mesh_graph import build_triangle_adjacency

    full_adj = build_triangle_adjacency(original_triangles)
    analysis_rems = analysis.get("removals", [])
    components = _split_into_full_components(removed_tris, original_triangles, full_adj)
    valid_tri_set: set[int] = set()
    for comp in components:
        if _classify_component_by_analysis(comp, original_triangles, analysis_rems):
            valid_tri_set.update(comp)
    invalid_tri_set: set[int] = set(analysis.get("restored_tris", []))
    return {"valid_tris": valid_tri_set, "invalid_tris": invalid_tri_set}
