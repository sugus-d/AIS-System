"""文件 / 图像 / 缓存 I/O 工具。

- 路径构建：get_output_path
- 图像落盘：save_img
- landmark 缓存读写：load_landmarks

注意：pipeline 缓存读取（load_cached_mesh / load_landmarks）统一走本模块，
CLI 层不直接拼缓存路径。
"""

import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt

from utils.logger import logger


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


def save_img(
    fig: plt.Figure, image_path: str, dpi: int = 500, pad_inches: float = 0, bbox_inches: str = "tight"
) -> None:
    """Save a matplotlib figure to disk."""
    Path(image_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(image_path, dpi=dpi, pad_inches=pad_inches, bbox_inches=bbox_inches)
    plt.close(fig)
    logger.success(f"Saved image to {image_path}")


def load_landmarks(subject: str, cache_dir: str = "results/cache") -> dict | None:
    """Load cached landmarks for a subject."""
    path = Path(cache_dir) / subject / "landmarks" / "landmarks.pkl"
    if not path.exists():
        return None
    with path.open("rb") as file_handle:
        return pkl.load(file_handle)
