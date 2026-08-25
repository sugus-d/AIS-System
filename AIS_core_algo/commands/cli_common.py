"""CLI 通用工具 — 查找原始 mesh 路径。

其他 CLI 装饰器（plot_cli / app_cli / batch_cli）已随 run_parameterization
移除；当前 CLI 入口统一走 commands/plot.py 的 argparse 框架。
"""

from __future__ import annotations

from pathlib import Path


def find_mesh_path(subject_id: str, mesh_dir: Path) -> Path | None:
    """根据 subject ID 在 mesh_dir 下查找 STD_fuse_mesh PLY 文件路径。

    模板文件可能不存在或尚未生成，需要先检查目录是否存在。
    取排序后的第一个 PLY 文件（通常只有一个）。

    Args:
        subject_id: Subject ID（作为子目录名）。
        mesh_dir: 原始 mesh 存放目录（调用方按自身运行 cwd 提供）。

    Returns:
        匹配的 PLY 文件路径；目录或文件不存在时返回 None。
    """
    subject_dir = mesh_dir / subject_id
    if not subject_dir.is_dir():
        return None
    ply_files = sorted(subject_dir.glob("STD_fuse_mesh_*.ply"))
    return ply_files[0] if ply_files else None
