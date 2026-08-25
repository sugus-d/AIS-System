"""测试训练组件 — DataSplitter、HPSearcher、Trainer。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from modeling.contracts import FeatureSet, TrainingConfig
from modeling.training.data_splitters import (
    KFoldSplitter,
    SPLITTERS,
    StratifiedKFoldSplitter,
)
from modeling.training.hp_searchers import (
    _inject_weight_params,
    _narrow_grid,
    _r2_score,
    GridSearch,
    RandomSearch,
    SEARCHERS,
)
from modeling.training.trainer import Trainer

# ===========================================================================
# DataSplitters
# ===========================================================================


class TestKFoldSplitter:
    """KFoldSplitter 的常规路径、边界、异常。"""

    def test_normal_path(self) -> None:
        splitter = KFoldSplitter(n_splits=3, n_repeats=2, random_state=42)
        y = np.random.rand(30)
        folds = list(splitter.split(y))
        assert len(folds) == 6  # 3 × 2
        train_idx, test_idx = folds[0]
        assert len(train_idx) + len(test_idx) == 30

    def test_single_repeat(self) -> None:
        splitter = KFoldSplitter(n_splits=5, n_repeats=1, random_state=42)
        y = np.random.rand(50)
        folds = list(splitter.split(y))
        assert len(folds) == 5
        # 每折无重叠
        used = set()
        for _, te_idx in folds:
            assert len(used & set(te_idx.tolist())) == 0
            used.update(te_idx.tolist())

    def test_insufficient_samples(self) -> None:
        """n_splits > len(y) 时 sklearn 应抛出 ValueError。"""
        splitter = KFoldSplitter(n_splits=5, n_repeats=1, random_state=42)
        y = np.random.rand(3)
        with pytest.raises(ValueError, match="Cannot have number of splits"):
            list(splitter.split(y))


class TestStratifiedKFoldSplitter:
    """StratifiedKFoldSplitter 的测试。"""

    def test_normal_path(self) -> None:
        splitter = StratifiedKFoldSplitter(n_splits=3, n_repeats=2, random_state=42)
        y = np.random.rand(30) * 40
        folds = list(splitter.split(y))
        assert len(folds) == 6

    def test_single_repeat(self) -> None:
        splitter = StratifiedKFoldSplitter(n_splits=4, n_repeats=1, random_state=42)
        y = np.random.rand(40) * 40
        folds = list(splitter.split(y))
        assert len(folds) == 4

    def test_insufficient_samples(self) -> None:
        splitter = StratifiedKFoldSplitter(n_splits=5, n_repeats=1, random_state=42)
        y = np.random.rand(3)
        with pytest.raises(ValueError):
            list(splitter.split(y))

    def test_degrades_to_kfold_when_class_too_small(self) -> None:
        """某类样本数 < n_splits 时退化为 KFold，不发 sklearn UserWarning。"""
        import warnings

        splitter = StratifiedKFoldSplitter(n_splits=3, n_repeats=1, random_state=42)
        # 构造极端不平衡：某类只有 2 个样本 < n_splits=3
        y = np.concatenate([np.full(28, 5.0), np.full(2, 50.0)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            folds = list(splitter.split(y))
        assert len(folds) == 3
        assert not any("least populated" in str(w.message) for w in caught)
        # 退化为 KFold：每折 train+test 覆盖全部样本
        tr_idx, te_idx = folds[0]
        assert len(tr_idx) + len(te_idx) == 30
        assert set(tr_idx) | set(te_idx) == set(range(30))


# ===========================================================================
# HPSearchers
# ===========================================================================


class _TestModel:
    """简单的模型桩 — 用于测试搜索逻辑。"""

    name = "test_model"

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}
        self.coef_ = 1.0

    def get_param_space(self) -> dict:
        return {"alpha": [0.1, 1.0, 10.0]}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # 简单线性拟合
        self.coef_ = np.dot(X[:, 0], y) / np.dot(X[:, 0], X[:, 0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0] * self.coef_


class TestRandomSearch:
    """RandomSearch 的测试。"""

    def test_normal_path(self) -> None:
        searcher = RandomSearch()
        rng = np.random.default_rng(42)
        X = rng.random((30, 3))
        # y 是 X[:,0] 的线性函数 + 噪声
        y = X[:, 0] * 2.0 + rng.normal(0, 0.1, 30)
        model = _TestModel()
        inner = KFoldSplitter(n_splits=3, n_repeats=1, random_state=42)
        best_model, best_params = searcher.search(
            model, X, y, inner, n_iter=5, score_metric="r2"
        )
        assert best_model is not None
        assert "alpha" in best_params
        # 预测合理
        preds = best_model.predict(X)
        assert np.isfinite(preds).all()

    def test_empty_grid(self) -> None:
        class _EmptyGridModel(_TestModel):
            def get_param_space(self) -> dict:
                return {}

        searcher = RandomSearch()
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        model = _EmptyGridModel()
        inner = KFoldSplitter(n_splits=2, n_repeats=1, random_state=42)
        best_model, best_params = searcher.search(
            model, X, y, inner, n_iter=5
        )
        assert best_model is not None
        # 空网格会注入 class_weight + dist_k
        assert "class_weight" in best_params
        assert "dist_k" in best_params


class TestGridSearch:
    """GridSearch 的测试。"""

    def test_normal_path(self) -> None:
        searcher = GridSearch()
        rng = np.random.default_rng(42)
        X = rng.random((30, 3))
        y = X[:, 0] * 2.0 + rng.normal(0, 0.1, 30)
        model = _TestModel()
        inner = KFoldSplitter(n_splits=2, n_repeats=1, random_state=42)
        best_model, best_params = searcher.search(
            model, X, y, inner, score_metric="r2"
        )
        assert best_model is not None
        assert "alpha" in best_params

    def test_full_coverage(self) -> None:
        """GridSearch 应遍历所有 3 种 alpha 值。"""
        searcher = GridSearch()
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        model = _TestModel()
        inner = KFoldSplitter(n_splits=2, n_repeats=1, random_state=42)
        best_model, best_params = searcher.search(
            model, X, y, inner
        )
        assert best_model is not None
        assert best_params["alpha"] in [0.1, 1.0, 10.0]

    def test_empty_grid(self) -> None:
        class _EmptyGridModel(_TestModel):
            def get_param_space(self) -> dict:
                return {}

        searcher = GridSearch()
        X = np.random.rand(20, 3)
        y = np.random.rand(20)
        model = _EmptyGridModel()
        inner = KFoldSplitter(n_splits=2, n_repeats=1, random_state=42)
        best_model, best_params = searcher.search(
            model, X, y, inner
        )
        assert best_model is not None
        # 空网格会注入 class_weight + dist_k
        assert "class_weight" in best_params
        assert "dist_k" in best_params


class TestSearchHelpers:
    """搜索辅助函数的测试。"""

    def test_r2_score_perfect(self) -> None:
        score = _r2_score(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert score == 1.0

    def test_r2_score_poor(self) -> None:
        score = _r2_score(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert score <= 0

    def test_r2_score_constant_y(self) -> None:
        score = _r2_score(np.array([5, 5, 5]), np.array([1, 2, 3]))
        assert score == -np.inf

    def test_inject_weight_params_empty(self) -> None:
        result = _inject_weight_params({})
        assert "class_weight" in result
        assert "dist_k" in result
        assert result["class_weight"] == [3, 5, 8, 12]

    def test_inject_weight_params_existing(self) -> None:
        result = _inject_weight_params({"class_weight": [1, 2]})
        assert result["class_weight"] == [1, 2]  # 不覆盖已存在的
        assert "dist_k" in result

    def test_inject_weight_params_wide(self) -> None:
        result = _inject_weight_params({}, wide=True)
        assert 15 in result["class_weight"]
        assert 0.8 in result["dist_k"]

    def test_narrow_grid_int(self) -> None:
        narrowed = _narrow_grid({"alpha": 10}, {"alpha": [1, 5, 10, 20, 50]})
        assert "alpha" in narrowed
        # 7 ~ 13 范围内: 10
        assert set(narrowed["alpha"]).issubset({7, 8, 9, 10, 11, 12, 13})

    def test_narrow_grid_non_numeric(self) -> None:
        """非数值参数应保持原样。"""
        narrowed = _narrow_grid({"solver": "auto"}, {"solver": ["auto", "svd", "cholesky"]})
        assert narrowed["solver"] == ["auto", "svd", "cholesky"]  # 保持原样


# ===========================================================================
# Trainer
# ===========================================================================


class TestTrainer:
    """Trainer 编排器的测试。"""

    def test_build_splitter(self) -> None:
        config = TrainingConfig(
            models=["Ridge"],
            data_splitter="kfold",
            data_splitter_params={"n_splits": 3, "n_repeats": 1},
        )
        trainer = Trainer(config)
        splitter = trainer._build_splitter()
        assert splitter.name == "kfold"
        assert splitter.n_splits == 3

    def test_build_searcher(self) -> None:
        config = TrainingConfig(
            models=["Ridge"],
            hp_searcher="random",
            hp_searcher_params={"n_iter": 10},
        )
        trainer = Trainer(config)
        searcher = trainer._build_searcher()
        assert searcher is not None
        assert searcher.name == "random"

    def test_build_searcher_none(self) -> None:
        config = TrainingConfig(models=["Ridge"], hp_searcher="none")
        trainer = Trainer(config)
        assert trainer._build_searcher() is None

    def test_unknown_splitter_raises(self) -> None:
        config = TrainingConfig(
            models=["Ridge"],
            data_splitter="nonexistent",
        )
        trainer = Trainer(config)
        with pytest.raises(ValueError, match="未知 data_splitter"):
            trainer._build_splitter()

    def test_train_single_model(self) -> None:
        """端到端：用最小配置训练一个模型。"""
        rng = np.random.default_rng(42)
        n = 30
        X = rng.random((n, 3))
        y = X[:, 0] * 2.0 + X[:, 1] * (-1.0) + rng.normal(0, 0.5, n)

        feature_set = FeatureSet(
            name="test_scheme",
            y=y,
            X=X,
            feature_names=["f1", "f2", "f3"],
        )
        config = TrainingConfig(
            models=["Ridge"],
            data_splitter="kfold",
            data_splitter_params={"n_splits": 3, "n_repeats": 1},
            hp_searcher="none",
        )
        trainer = Trainer(config)
        results = trainer.train(feature_set)
        assert len(results) == 1
        result = results[0]
        assert result.model_name == "Ridge"
        assert result.scheme == "test_scheme"
        assert "r" in result.metrics
        assert "rmse" in result.metrics
        assert len(result.predictions) == n
        assert np.all(np.isfinite(result.predictions))

    def test_single_label_confusion_matrix_4x4(self) -> None:
        """预测全落单一类时 confusion_matrix 仍为 4×4（labels 显式传入）。"""
        import json
        import warnings

        rng = np.random.default_rng(42)
        n = 30
        X = rng.random((n, 3))
        y = np.full(n, 5.0)  # 全 5°→单一严重度 bin

        feature_set = FeatureSet(
            name="single_label_scheme",
            y=y,
            X=X,
            feature_names=["f1", "f2", "f3"],
        )
        config = TrainingConfig(
            models=["Ridge"],
            data_splitter="kfold",
            data_splitter_params={"n_splits": 3, "n_repeats": 1},
            hp_searcher="none",
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = Trainer(config).train(feature_set)[0]
        # 不触发 sklearn 的 single-label confusion matrix UserWarning
        assert not any("single label" in str(w.message).lower() for w in caught)
        # 落盘混淆矩阵为 4×4（labels=[0,1,2,3] 修复）
        with open(Path(result.session_dir) / "confusion_matrix.json") as f:
            cm_json = json.load(f)
        matrix = cm_json["matrix"]
        assert len(matrix) == 4
        assert all(len(row) == 4 for row in matrix)
        assert np.asarray(result.predictions).shape == (n,)

    def test_multiple_models(self) -> None:
        """多模型训练。"""
        rng = np.random.default_rng(42)
        n = 30
        X = rng.random((n, 3))
        y = X[:, 0] + rng.normal(0, 0.3, n)

        feature_set = FeatureSet(
            name="multi_test",
            y=y,
            X=X,
            feature_names=["f1", "f2", "f3"],
        )
        config = TrainingConfig(
            models=["Ridge", "DecisionTree"],
            data_splitter="kfold",
            data_splitter_params={"n_splits": 3, "n_repeats": 1},
            hp_searcher="none",
        )
        trainer = Trainer(config)
        results = trainer.train(feature_set)
        assert len(results) == 2
        assert results[0].model_name == "Ridge"
        assert results[1].model_name == "DecisionTree"

    def test_apply_calibration_per_class_bias(self) -> None:
        """per-class 偏差从 fold 预测统计并应用校正。"""
        y = np.array([5.0, 6.0, 15.0, 16.0, 25.0, 45.0])  # 覆盖 4 类
        preds = y + 2.0  # 全部偏 +2°
        fold_preds = [(np.arange(len(y)), preds.copy())]
        trainer = Trainer(TrainingConfig(models=["Ridge"]))
        corrected, details = trainer._apply_calibration(preds, y, fold_preds)
        # 每类 pred−true 均值 = 2.0
        for c in range(4):
            assert details["bias"][c] == pytest.approx(2.0)
        # 校正后 ≈ 真实值（钳制到类范围内）
        assert np.allclose(corrected, y, atol=1e-6)

    def test_apply_calibration_empty_class_zero_bias(self) -> None:
        """无样本的类别 bias 为 0（不污染校正）。"""
        y = np.array([5.0, 6.0])  # 仅 class 0
        preds = np.array([7.0, 8.0])
        fold_preds = [(np.arange(2), preds.copy())]
        trainer = Trainer(TrainingConfig(models=["Ridge"]))
        corrected, details = trainer._apply_calibration(preds, y, fold_preds)
        assert details["bias"][0] == pytest.approx(2.0)
        assert details["bias"][1] == 0.0
        assert details["bias"][2] == 0.0
        assert details["bias"][3] == 0.0


# ===========================================================================
# SPLITTERS / SEARCHERS 注册表
# ===========================================================================


class TestRegistry:
    """注册表完整性。"""

    def test_all_splitters_registered(self) -> None:
        assert "kfold" in SPLITTERS
        assert "stratified_kfold" in SPLITTERS

    def test_all_searchers_registered(self) -> None:
        assert "random" in SEARCHERS
        assert "grid" in SEARCHERS

    def test_components_instantiable(self) -> None:
        for name, cls in SPLITTERS.items():
            inst = cls(n_splits=2, n_repeats=1)
            assert inst.name == name

        for name, cls in SEARCHERS.items():
            inst = cls()
            assert inst.name == name
