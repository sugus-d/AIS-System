"""Verify feature schemes can be loaded without errors (shape + no crash)."""
import pytest

from features.selectors.schemes import SELECTION_REGISTRY


class TestFeatureSchemes:
    """Smoke tests: load each scheme, check basic structure."""

    @pytest.mark.parametrize("name", list(SELECTION_REGISTRY.keys()))
    def test_load(self, name):
        """Loading a scheme returns dict with y, feature_names."""
        scheme = SELECTION_REGISTRY[name]
        assert scheme.name == name
        assert callable(scheme.load)

    @pytest.mark.parametrize("name", list(SELECTION_REGISTRY.keys()))
    def test_load_returns_data_dict(self, name):
        """调用 load() 不抛异常，且返回含 y 的数据 dict。

        依赖 results/ 下的数据文件；CI 无数据时跳过而非失败。
        """
        scheme = SELECTION_REGISTRY[name]
        try:
            data = scheme.load()
        except FileNotFoundError as exc:
            pytest.skip(f"缺少数据文件，跳过 {name}: {exc}")
        assert isinstance(data, dict)
        assert "y" in data
