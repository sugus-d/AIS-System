"""特征方案注册表 — 从 features/selectors/schemes.py 加载。

当前方案（版本号 = 注册表 key）:
  - v1.0.0 (alias=production): 人工 ROI 生产路径（30D）
  - v0.1.0 (alias=beta):       算法 ROI 路径（40D）
  - 0.0.x:                     归档实验（archived/*）
"""

from features.selectors.schemes import (
    EXTRACTION_REGISTRY,
    SCHEME_VERSIONS,
    SELECTION_REGISTRY,
)


def _resolve_version(name: str) -> str | None:
    """name → 版本号（精确 version / alias 匹配）。"""
    for _key, (version, alias) in SCHEME_VERSIONS.items():
        if name in (version, alias):
            return version
    return None


def get_selector(name: str) -> object:
    """按 name / version / alias 查方案注册表。

    version/alias 先解析到版本号，再返回该版本对应的 **selection 方案**；
    精确名匹配（key/name/alias）落到对应注册表（含 EXTRACTION 单定义）。
    """
    if name in SELECTION_REGISTRY:
        return SELECTION_REGISTRY[name]
    if name in EXTRACTION_REGISTRY:
        return EXTRACTION_REGISTRY[name]
    # EXTRACTION 单定义仅按 name 精确匹配（alias=production 是展示标注，不参与解析，
    # 避免与 SELECTION 的 v1.0.0→production 冲突）
    for scheme in EXTRACTION_REGISTRY.values():
        if name == scheme.name:
            return scheme
    target_version = _resolve_version(name)
    if target_version is not None:
        for scheme in SELECTION_REGISTRY.values():
            if scheme.version == target_version:
                return scheme
    raise KeyError(f"未知选择器: {name}，可选: {list(SCHEME_VERSIONS.keys())}")


def list_selectors() -> list[str]:
    return list(SELECTION_REGISTRY.keys())
