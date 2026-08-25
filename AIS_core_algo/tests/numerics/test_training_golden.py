"""M4 训练内核黄金测试 — 特征筛选/CI/HP 搜索纯函数 + 模型参数空间 + 端到端迷你 CV。

注意：端到端测试用 fixed_params 模式保持黄金值稳定（HP 搜索路径含
随机采样，不适合逐位断言；其纯函数由下方测试覆盖）。class_weight
注入 bug 已修复（_inner_search 构造时过滤，见 _hp_search.py）。
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.numerics.conftest import assert_golden, RNG_SEED

GOLDEN_SELECTOR = {
    "hybrid_scores": ("(12,)", "3.9821057888", "866419f4cb38d71cf9c02400766cd043"),
    "dedup_by_corr": ("(12,)", "66.0000000000", "fab8abc5aa12ff95bc06b4e3553f967a"),
    "select_morph": ("(10,)", "53.0000000000", "a217b968f131ffb9e29547e56c9e58b2"),
    "select_region": ("(12,)", "66.0000000000", "fab8abc5aa12ff95bc06b4e3553f967a"),
}

GOLDEN_CI = {
    "ci_tr": ("(60, 3)", "-0.0000000000", "5921dd91f5f918c656a5eaa0fa994c01"),
    "ci_te": ("(60, 3)", "-0.0000000000", "5921dd91f5f918c656a5eaa0fa994c01"),
}

GOLDEN_HP = {
    "cv_inject": ("dict", "-", "1bf66d251b3270bad390398bed3ff55f"),
    "cv_inject_wide": ("dict", "-", "605bc729e823b3fe6f63999fe216f656"),
    "tr_narrow": ("dict", "-", "c1fe332ec7c958a7b4ab26b6adc7cee5"),
}

GOLDEN_MODELS = {
    "BaggingEN": "6793af44066445873022b8e6c792c480",
    "CatBoost": "e01b79a0b855f9f2ef7bb53c5ea77754",
    "DecisionTree": "fd011ad818c44a1b2b64d6252c966ce6",
    "ElasticNet": "800d8a9cb861137903b52c102e5ac4a1",
    "ExtraTrees": "5f8ae34377b57a5da06f6c58bfe204a4",
    "GBRT": "de1f765bdf45180750b0a496f150f2d2",
    "HistGBRT": "ea3611d64334a6515209740287c4b70c",
    "Huber": "f61bb757156cdec1353baac0cfaf966a",
    "KNN": "cc3e03c07f45767743d8226fd7e1ee67",
    "LightGBM": "0555f4eca6649affef1fc1c328249e12",
    "MLP": "7ca19fa67e3d5ea05376e773e7c01463",
    "RF": "2cb4bc64756683b0f8e5fdbac0e63b29",
    "Ridge": "f4f2f8f693d5171d20c607cd3f4dd498",
    "SGD": "17f0f1133a9ac3c1f81a918ac6489da5",
    "SVR": "b2f295cfb31c69ad13d6b46b419f7364",
    "XGBoost": "600e80400d9893307e4366fb24fd6a43",
}

GOLDEN_E2E = ("(30,)", "1205.7202548020", "6e621c6b037e7dc11a828be1a9fa9fef")


def test_selectors_golden() -> None:
    """每折特征筛选纯函数（合成 X/y）与黄金值一致。"""
    from modeling.training.feature_selector import _dedup_by_corr, _hybrid_scores, _select_morph, _select_region

    rng = np.random.default_rng(RNG_SEED)
    X = rng.normal(size=(60, 12))
    y = rng.uniform(5, 60, size=60)
    assert_golden("hybrid_scores", _hybrid_scores(X, y), *GOLDEN_SELECTOR["hybrid_scores"])
    assert_golden("dedup_by_corr", _dedup_by_corr(X, y), *GOLDEN_SELECTOR["dedup_by_corr"])
    assert_golden("select_morph", _select_morph(X, y), *GOLDEN_SELECTOR["select_morph"])
    assert_golden("select_region", _select_region(X, y), *GOLDEN_SELECTOR["select_region"])


def test_ci_per_fold_golden() -> None:
    """per-fold CI 计算（合成分组特征）与黄金值一致。"""
    from modeling.training.feature_selector import _compute_ci_per_fold

    rng = np.random.default_rng(RNG_SEED)
    X = rng.normal(size=(60, 12))
    y = rng.uniform(5, 60, size=60)
    groups = {f"m{i}|dm": list(range(i * 3, i * 3 + 3)) for i in range(4)}
    ci_tr, ci_te_fn = _compute_ci_per_fold(X, y, groups)
    assert_golden("ci_tr", ci_tr, *GOLDEN_CI["ci_tr"])
    assert_golden("ci_te", ci_te_fn(X), *GOLDEN_CI["ci_te"])


def test_corrcoef_zero_variance_no_nan_warning() -> None:
    """特征含零方差列时 corrcoef 不产生 NaN divide RuntimeWarning（nan→0 修复）。"""
    import warnings

    from modeling.training.feature_selector import _select_morph, _select_region

    rng = np.random.default_rng(RNG_SEED)
    X = rng.normal(size=(60, 12))
    X[:, 3] = 5.0  # 人为制造零方差列
    y = rng.uniform(5, 60, size=60)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _select_morph(X, y)
        _select_region(X, y)
    divide_warnings = [w for w in caught if "invalid value" in str(w.message)]
    assert divide_warnings == [], f"corrcoef 仍触发 divide warning: {divide_warnings}"


def test_ci_corrcoef_zero_variance_no_nan_warning() -> None:
    """CI 值含零方差列时 corrcoef 不产生 NaN divide RuntimeWarning。"""
    import warnings

    from modeling.training.feature_selector_ci import _compute_ci_per_fold

    rng = np.random.default_rng(RNG_SEED)
    X = rng.normal(size=(60, 12))
    X[:, :3] = 1.0  # 第一组全常数 → Lasso 系数 0 → CI 零方差
    y = rng.uniform(5, 60, size=60)
    groups = {f"m{i}|dm": list(range(i * 3, i * 3 + 3)) for i in range(4)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ci_tr, _ = _compute_ci_per_fold(X, y, groups)
    divide_warnings = [w for w in caught if "invalid value" in str(w.message)]
    assert divide_warnings == [], f"corrcoef 仍触发 divide warning: {divide_warnings}"
    assert ci_tr.shape == (60, 3)


def test_hp_search_golden() -> None:
    """HP 搜索纯函数（固定网格）与黄金值一致。"""
    from modeling.training.hp_searchers import _narrow_grid as tr_narrow
    from modeling.training.hp_searchers._search_utils import _inject_weight_params as cv_inject

    grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 8, 12], "learning_rate": [0.05, 0.1, 0.2, 0.5]}
    ref = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}
    assert_golden("cv_inject", cv_inject(grid), *GOLDEN_HP["cv_inject"])
    assert_golden("cv_inject_wide", cv_inject(grid, wide=True), *GOLDEN_HP["cv_inject_wide"])
    assert_golden("tr_narrow", tr_narrow(ref, grid), *GOLDEN_HP["tr_narrow"])


def test_model_param_spaces_golden() -> None:
    """全部注册模型的参数空间与黄金值一致。"""
    from modeling.models import REGISTRY

    for name, expected_md5 in GOLDEN_MODELS.items():
        cls = REGISTRY[name]
        params = cls().get_param_space()
        assert_golden(f"model_{name}_space", params, "dict", "-", expected_md5)


@pytest.mark.slow
def test_trainer_e2e_golden() -> None:
    """端到端迷你训练（Trainer + per-fold 筛选）：折划分→筛选→训练→预测→聚合。"""
    from modeling.contracts import FeatureSet, TrainingConfig
    from modeling.training.trainer import Trainer

    rng = np.random.default_rng(RNG_SEED + 1)
    n = 30
    Xb = rng.normal(size=(n, 5))
    Xm = rng.normal(size=(n, 12))
    Xr = rng.normal(size=(n, 60))
    y = rng.uniform(5, 60, size=n)
    col_names = [f"m{i}|dm" for i in range(20)]
    config = TrainingConfig(
        models=["Ridge"],
        data_splitter="kfold",
        data_splitter_params={"n_splits": 2, "n_repeats": 2},
        hp_searcher="none",
        transform_target=True,
        feature_selector="per_fold",
    )
    feature_set = FeatureSet(
        name="e2e", y=y, X=Xb, feature_names=[],
        X_raw_blocks={"basic": Xb, "morph": Xm, "region": Xr},
        region_column_names=col_names,
    )
    result = Trainer(config).train(feature_set)[0]
    assert_golden("trainer_e2e_preds", np.array(result.predictions), *GOLDEN_E2E)
