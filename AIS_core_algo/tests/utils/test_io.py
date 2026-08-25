"""io 工具单元测试 — get_output_path / load_landmarks（save_img 已迁 visualization）。"""

from __future__ import annotations

import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import pytest


class TestGetOutputPath:
    def test_output_type_subdir(self, tmp_path: Path) -> None:
        from utils.io import get_output_path

        result = get_output_path(
            "data/subject/scan.ply",
            output_type="figures",
            output_dir=str(tmp_path),
        )
        expected = str(tmp_path / "figures" / "subject_scan.png")
        assert result == expected

    def test_no_output_type(self, tmp_path: Path) -> None:
        from utils.io import get_output_path

        result = get_output_path(
            "data/subject/scan.ply",
            output_dir=str(tmp_path),
        )
        expected = str(tmp_path / "subject_scan.png")
        assert result == expected

    def test_creates_directory(self, tmp_path: Path) -> None:
        from utils.io import get_output_path

        nested = tmp_path / "a" / "b"
        result = get_output_path(
            "data/subject/scan.ply",
            output_type="figures",
            output_dir=str(nested),
        )
        assert Path(result).parent.exists()


class TestSaveImg:
    def test_saves_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from visualization._render_utils import save_img

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        img_path = str(tmp_path / "test.png")

        called = False
        original_savefig = fig.savefig

        def _tracking_savefig(*args: object, **kwargs: object) -> None:
            nonlocal called
            called = True
            original_savefig(*args, **kwargs)

        monkeypatch.setattr(fig, "savefig", _tracking_savefig)
        save_img(fig, img_path)
        assert called, "fig.savefig 应被调用"

    def test_file_is_written(self, tmp_path: Path) -> None:
        from visualization._render_utils import save_img

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        img_path = str(tmp_path / "test.png")
        save_img(fig, img_path)
        assert Path(img_path).exists()
        assert Path(img_path).stat().st_size > 0


class TestLoadLandmarks:
    def test_cache_missing_returns_none(self, tmp_path: Path) -> None:
        from utils.io import load_landmarks

        result = load_landmarks("subject_xyz", cache_dir=str(tmp_path))
        assert result is None

    def test_cache_exists_returns_dict(self, tmp_path: Path) -> None:
        from utils.io import load_landmarks

        cache_dir = tmp_path / "cache"
        cache_path = cache_dir / "subject_xyz" / "landmarks"
        cache_path.mkdir(parents=True)
        data = {"landmark_A": [1.0, 2.0, 3.0], "landmark_B": [4.0, 5.0, 6.0]}
        with (cache_path / "landmarks.pkl").open("wb") as f:
            pkl.dump(data, f)

        result = load_landmarks("subject_xyz", cache_dir=str(cache_dir))
        assert result == data
