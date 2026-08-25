"""特征提取器注册表 — 每个提取器为纯函数，注册后可按名称调用。"""

from features.extractors.asymmetry import extract_asymmetry
from features.extractors.basic import extract_basic
from features.extractors.morphology import extract_morphology

_EXTRACTORS: dict[str, object] = {}

def register(name: str, fn: object) -> None:
    _EXTRACTORS[name] = fn

def get(name: str) -> object:
    if name not in _EXTRACTORS:
        raise KeyError(f"未知提取器: {name}，可选: {list(_EXTRACTORS.keys())}")
    return _EXTRACTORS[name]

def list_extractors() -> list[str]:
    return list(_EXTRACTORS.keys())

register("basic", extract_basic)
register("morphology", extract_morphology)
register("asymmetry", extract_asymmetry)
