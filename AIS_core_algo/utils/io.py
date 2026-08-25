"""文件 / 缓存 I/O 工具。

- 路径构建：get_output_path
- landmark 缓存读写：load_landmarks

图像落盘（save_img）已移入 visualization/_render_utils.py（渲染编排工具，
基础层不携带 matplotlib）。

注意：pipeline 缓存读取（load_cached_mesh / load_landmarks）统一走本模块，
CLI 层不直接拼缓存路径。
"""

import pickle as pkl
from pathlib import Path


def get_output_path(
    file_path: str,
    output_type: str = "",
    output_dir: str = "results",
    file_type: str = "png",
) -> str:
    """Build an output path from an input file path."""
    output_root = Path(output_dir)
    if output_type:
        output_root = output_root / output_type

    output_root.mkdir(parents=True, exist_ok=True)

    input_path = Path(file_path)
    parent_name = input_path.parent.name
    stem = input_path.stem
    output_name = (
        f"{parent_name}_{stem}.{file_type}" if parent_name else f"{stem}.{file_type}"
    )
    return str(output_root / output_name)


def load_landmarks(subject: str, cache_dir: str = "results/cache") -> dict | None:
    """Load cached landmarks for a subject."""
    path = Path(cache_dir) / subject / "landmarks" / "landmarks.pkl"
    if not path.exists():
        return None
    with path.open("rb") as file_handle:
        return pkl.load(file_handle)
