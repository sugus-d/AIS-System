"""统一学术样式（matplotlib rcParams）——渲染层定义，编排层应用。

所有入口（predict.py 生产报告 + commands/export 论文图）共用同一套 rcParams，
消除"生产报告 vs 论文图"风格割裂。

约定（2026-08-15 用户决策）：
- font.size=8 中号——所有图通用折中（瀑布图 panel 内部显式字号主导，7→8 影响有限）
- Arial/Liberation Sans 论文字体链（Linux 下 Liberation Sans 为度量兼容 fallback）
- **保留坐标轴边框**（不设 axes.spines.top/right，保持默认有框）
- figure.titlesize=9（多子图总标题）
- hatch.linewidth=3.0（瀑布图斜线阴影，panel 内部默认值，统一无害）
"""

from __future__ import annotations

ACADEMIC_STYLE: dict[str, object] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "hatch.linewidth": 3.0,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
}
