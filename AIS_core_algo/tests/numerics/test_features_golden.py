"""M1 特征工程黄金测试 — 核心方案特征加载与选择器数值。

黄金值来自 tests/numerics/_generate_golden.py（已验证正确的 HEAD 输出）。
"""

from __future__ import annotations

from tests.numerics.conftest import assert_golden, DATA_DIR

GOLDEN = {
    "v0.1.0": {
        "X_basic": ("(122, 40)", "84743.7941930000", "ff1699c054bbcfd59d4158c9cccb7c3e"),
        "y": ("(122,)", "3102.0000000000", "c8c3626aeeb3217019f6050a6222baa3"),
    },
    "archived/morph_region_ci_37d": {
        "X_basic": ("(122, 37)", "101587.7113210000", "3bffdce2519967ee13d340ebd6352d1e"),
        "y": ("(122,)", "3102.0000000000", "c8c3626aeeb3217019f6050a6222baa3"),
    },
    "archived/morph_region_ci_36d": {
        "X_basic": ("(122, 36)", "84607.7862820000", "c9247e3995e0fcff039e05ad890281fd"),
        "y": ("(122,)", "3102.0000000000", "c8c3626aeeb3217019f6050a6222baa3"),
    },
    "archived/morph_region_ci_35d": {
        "X_basic": ("(122, 35)", "84485.9487480000", "e7ef8ce0994d58bd55ac21d901127b57"),
        "y": ("(122,)", "3102.0000000000", "c8c3626aeeb3217019f6050a6222baa3"),
    },
    "archived/morph_region_ci_27d": {
        # 名为 27d 但 X_basic 为 42 维——历史命名残留，黄金锁现状不纠名
        "X_basic": ("(122, 42)", "83472.6384100000", "268ffa8608ef0e3feb59497dd333941c"),
        "y": ("(122,)", "3102.0000000000", "c8c3626aeeb3217019f6050a6222baa3"),
    },
    "archived/canonical_union_64d": {
        "X_basic": ("(122, 64)", "117882.2431840000", "bc6533aef541cab70163eadb19dabfe5"),
        "y": ("(122,)", "3102.0000000000", "c8c3626aeeb3217019f6050a6222baa3"),
    },
    "archived/canonical_44d": {
        "X_basic": ("(122, 44)", "75530.3824840000", "56c11502d6853443b82f1231d5f4ede4"),
        "y": ("(122,)", "3102.0000000000", "c8c3626aeeb3217019f6050a6222baa3"),
    },
}


def test_scheme_features_load(monkeypatch) -> None:
    """核心特征方案加载：X/y 与黄金值逐位一致（chdir 镜像目录命中相对路径）。"""
    monkeypatch.chdir(DATA_DIR / "features")
    from features.selectors.schemes import SELECTION_REGISTRY

    for scheme, keys in GOLDEN.items():
        data = SELECTION_REGISTRY[scheme].load()
        for key, expected in keys.items():
            assert_golden(f"feat_{scheme}_{key}", data[key], *expected)
