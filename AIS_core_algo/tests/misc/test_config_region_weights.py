"""Tests for config.yaml region_indices — weight validation and structure.

验证 region_indices 配置节的 active_regions、region_weights 和
quantity_weights 字段结构正确，权重和为 ≈1。
"""

import yaml

_CONFIG_PATH = "config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_region_indices_section_exists():
    """region_indices section must be present in config."""
    cfg = _load_config()
    assert "region_indices" in cfg, "Missing region_indices section"


def test_active_regions_structure():
    """active_regions must be a list of 3 integers (region IDs)."""
    cfg = _load_config()
    active = cfg["region_indices"]["active_regions"]
    assert isinstance(active, list), "active_regions must be a list"
    assert len(active) == 3, f"active_regions must have 3 elements, got {len(active)}"
    for rid in active:
        assert isinstance(rid, int), f"region ID must be int, got {type(rid).__name__}"


def test_region_weights_length():
    """region_weights must be a list of 3 floats."""
    cfg = _load_config()
    region_weights = cfg["region_indices"]["region_weights"]
    assert isinstance(region_weights, list), "region_weights must be a list"
    assert len(region_weights) == 3, \
        f"region_weights must have 3 elements, got {len(region_weights)}"


def test_quantity_weights_length():
    """quantity_weights must be a list of 3 floats."""
    cfg = _load_config()
    quantity_weights = cfg["region_indices"]["quantity_weights"]
    assert isinstance(quantity_weights, list), "quantity_weights must be a list"
    assert len(quantity_weights) == 3, \
        f"quantity_weights must have 3 elements, got {len(quantity_weights)}"


def test_region_weights_sum_to_one():
    """region_weights must sum to approximately 1.0 (within 1e-6)."""
    cfg = _load_config()
    weights = cfg["region_indices"]["region_weights"]
    total = sum(weights)
    assert abs(total - 1.0) < 1e-6, \
        f"region_weights sum to {total}, expected ~1.0"


def test_quantity_weights_sum_to_one():
    """quantity_weights must sum to approximately 1.0 (within 1e-6)."""
    cfg = _load_config()
    weights = cfg["region_indices"]["quantity_weights"]
    total = sum(weights)
    assert abs(total - 1.0) < 1e-6, \
        f"quantity_weights sum to {total}, expected ~1.0"


def test_region_weights_non_negative():
    """All region_weights must be non-negative."""
    cfg = _load_config()
    weights = cfg["region_indices"]["region_weights"]
    for i, w in enumerate(weights):
        assert w >= 0, f"region_weights[{i}] = {w} is negative"


def test_quantity_weights_non_negative():
    """All quantity_weights must be non-negative."""
    cfg = _load_config()
    weights = cfg["region_indices"]["quantity_weights"]
    for i, w in enumerate(weights):
        assert w >= 0, f"quantity_weights[{i}] = {w} is negative"
