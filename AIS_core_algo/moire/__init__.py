"""Moiré 显示模块（仅 prediction 报告显示，不进主干流程）。

算法层在 `moire.moire.compute_moire_distances`；渲染在
`visualization.moire_panels.render_moire`（三层分离）。
"""

from moire.moire import compute_moire_distances

__all__ = ["compute_moire_distances"]
