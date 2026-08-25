"""特征方案注册表 — 从 features/selectors/schemes.py 加载。

当前方案（语义化命名）:
  - morph_region_ci_40d: 默认方案（文档最优 🏆）
  - archived/*:    历史保留方案（可加载复现，不再推荐）
"""

from features.selectors.schemes import SELECTION_REGISTRY


def get_selector(name: str) -> object:
    if name in SELECTION_REGISTRY:
        return SELECTION_REGISTRY[name]
    raise KeyError(f"未知选择器: {name}，可选: {list(SELECTION_REGISTRY.keys())}")

def list_selectors() -> list[str]:
    return list(SELECTION_REGISTRY.keys())
