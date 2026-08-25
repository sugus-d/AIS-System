"""路径工具函数——`_get_latest_edited` 等跨服务共享函数。"""

from __future__ import annotations

from ..constants import CACHE_DIR, PLATFORM_CACHE_DIR


def _get_latest_edited(subject_id: str) -> str | None:
    """找到最新的 edited PLY（按时间戳后缀排序，优先 platform cache，回退外部缓存）。"""
    for base in (PLATFORM_CACHE_DIR, CACHE_DIR):
        for subdir in ("extract_roi", "align"):
            d = base / subject_id / subdir
            if not d.exists():
                continue
            edited = sorted(d.glob("roi_edited_*.ply"))
            if edited:
                return str(edited[-1])
    return None
