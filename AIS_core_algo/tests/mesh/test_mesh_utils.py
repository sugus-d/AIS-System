from pathlib import Path

from utils.io import get_output_path


def test_get_output_path_uses_parent_and_stem() -> None:
    output_path = get_output_path(
        "/home/nnb/code/python/AIS/data/mesh/S0006/STD_fuse_mesh001.ply",
        output_type="diag",
        output_dir="results",
        file_type="png",
    )

    assert Path(output_path) == Path("results/diag/S0006_STD_fuse_mesh001.png")


def test_get_output_path_falls_back_to_stem() -> None:
    output_path = get_output_path(
        "sample.ply",
        output_type="",
        output_dir="results",
        file_type="png",
    )

    assert Path(output_path) == Path("results/sample.png")
