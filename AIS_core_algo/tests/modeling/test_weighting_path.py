"""回归测试：加权训练入口（modeling.train --weighting）不因错误导入路径崩溃。

历史：_run_weighted 曾从空目录 pipeline.training 导入 Trainer，
导致 `python -m modeling.train --scheme X --weighting inv_freq` 抛 ModuleNotFoundError。
"""

from unittest import mock

import pytest

import modeling.train as train_module


def test_weighting_path_imports_ml_training_trainer(monkeypatch) -> None:
    """加权模式可走到 Trainer 层（import 路径正确），不抛 ModuleNotFoundError。

    用 mock 替换 Trainer.train 返回空列表，避免真实训练；
    真实加载 v0.1.0 scheme 数据（约 122 subject，秒级）。

    scheme 加载从 ``results/extraction/...`` 相对 cwd 读特征 CSV。
    仓库内测试数据位于 ``tests/data/numerics/features/``（结构镜像 results/），
    测试时 chdir 过去；数据缺失时 skip（对齐 golden 测试的数据缺失处理）。
    """
    from tests.numerics.conftest import DATA_DIR

    feature_root = DATA_DIR / "features"
    if not (feature_root / "results" / "extraction" / "features_extraction" / "v0.1.0" / "basic.csv").exists():
        pytest.skip(f"v0.1.0 特征数据缺失（本地放置后运行）: {feature_root}")

    monkeypatch.chdir(feature_root)

    from modeling.training.trainer import Trainer

    with mock.patch.object(Trainer, "train", return_value=[]):
        results = train_module.run(
            scheme_name="v0.1.0",
            weighting="inv_freq",
            algo_filter="HistGBRT",
            hp_n_iter=5,
        )

    assert results == []


def test_weighting_path_margin_trainer_importable() -> None:
    """margin 分支的导入路径同样指向 modeling.training（曾为坏路径）。"""
    from modeling.training.trainer_margin import MarginTrainer

    assert MarginTrainer is not None


def test_weighting_path_rejects_unknown_scheme() -> None:
    """未知方案名仍报错（壳层行为不变）。"""
    with pytest.raises(KeyError):
        train_module.run(
            scheme_name="no_such_scheme_xyz",
            weighting="inv_freq",
            algo_filter="HistGBRT",
        )
