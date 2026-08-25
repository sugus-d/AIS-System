"""特征提取器 — 纯函数提取器，按名称导出。

（原 register/get/list_extractors 注册表核查为零消费方死代码，2026-08-16 删除；
extract_all 直接 import 子模块函数，不走注册表。）
"""

from features.extractors.asymmetry import extract_asymmetry
from features.extractors.basic import extract_basic
from features.extractors.morphology import extract_morphology

__all__ = ["extract_asymmetry", "extract_basic", "extract_morphology"]
