"""HP 搜索回归测试 — 权重参数不泄漏给回归模型。

历史 bug：_inject_weight_params 无条件注入 class_weight/dist_k，
旧 cross_verification 路径未过滤就传给模型构造器，Ridge/SVR 等
回归模型不接受 → TypeError 崩溃（修复见 modeling/training/hp_searchers）。
"""

from __future__ import annotations

import numpy as np

from modeling.training.hp_searchers import RandomSearch


def test_random_search_ridge_with_injected_weights() -> None:
    """Ridge 走默认搜索（_inject_weight_params 注入权重参数）不崩溃、可预测。"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = rng.uniform(5, 60, size=30)

    model = __import__("modeling.models", fromlist=["REGISTRY"]).REGISTRY["Ridge"]()
    searcher = RandomSearch()
    best_model, _ = searcher.search(model, X, y, n_iter=5)

    assert best_model is not None
    preds = best_model.predict(X[:3])
    assert preds.shape == (3,)
    assert np.isfinite(preds).all()
