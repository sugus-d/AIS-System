"""切割边界分析与无效切割修复（_cut_analysis 目录包）。

内部职责拆分：
  - core.py            公开 API 主逻辑
  - boundary.py        切割边界边查找与分段
  - regions.py         移除三角区域与连通分量
  - classification.py  区域/分段有效性分类与结果构建
"""

from .core import analyze_cut_boundary, compute_removed_triangles, restore_invalid_cuts

__all__: list[str] = [
    "analyze_cut_boundary",
    "compute_removed_triangles",
    "restore_invalid_cuts",
]
