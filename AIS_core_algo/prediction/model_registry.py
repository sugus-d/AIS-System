"""预测模型注册与解析 — 版本名/别名 → joblib 模型包路径。

版本→路径映射是**部署配置**，唯一来源是 config.yaml 的 `models:` 段（训练产出新包
后在此登记即可，无需改预测代码）。config 缺失或缺少 `models:` 段 → 导入即报错
（fail-fast：配置错误应暴露而非静默回退）。

加载/缓存/结构校验不在本模块——复用 modeling.model_package 的共享加载器
（训练写、预测读、导出复读同一套模型包协议）。
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _load_model_config() -> dict:
    """读取 config.yaml 的 `models:` 段；缺失抛错（fail-fast）。"""
    try:
        raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise RuntimeError("缺少 config.yaml（模型注册表唯一来源），无法解析预测模型") from exc
    models = raw.get("models")
    if not isinstance(models, dict) or "registry" not in models:
        raise ValueError("config.yaml 缺少 models.registry 段（模型注册表）")
    return models


_model_config = _load_model_config()
# 版本名 → 模型包路径
MODEL_REGISTRY = _model_config["registry"]
# 别名（v1.0.0→production / v0.1.0→beta），唯一来源 config.yaml models.aliases
_MODEL_ALIASES = _model_config.get("aliases") or {}
# 缺省模型版本名（生产 = v1.0.0）
_DEFAULT_MODEL_NAME = _model_config.get("default", "v1.0.0")
DEFAULT_MODEL = MODEL_REGISTRY[_DEFAULT_MODEL_NAME]


def _lookup_registered(model_spec: str) -> str | None:
    """注册表查表：别名 → 版本名 → 路径；未注册返回 None。"""
    spec = _MODEL_ALIASES.get(model_spec, model_spec)
    return MODEL_REGISTRY.get(spec)


def resolve_model_path(model_spec: str) -> str:
    """CLI 用：版本名/别名 → 注册路径；未注册视为直接 joblib 路径（调试用）。

    未知命名且非存在的路径抛 ValueError（CLI → 报错退出）。
    """
    path = _lookup_registered(model_spec) or model_spec
    if not Path(path).exists():
        available = " / ".join(MODEL_REGISTRY)
        raise ValueError(f"未知模型: {model_spec}。可用: {available}，或指向存在的 joblib 文件路径")
    return path


def resolve_registered_model(model_spec: str) -> str:
    """API 用：仅接受注册版本名/别名（白名单），禁止任意路径。

    joblib 即 pickle，反序列化不可信文件有 RCE 风险——API 的 `model` 字段
    不接受任意路径，只允许注册表内的版本名/别名。
    """
    path = _lookup_registered(model_spec)
    if path is None:
        available = " / ".join(MODEL_REGISTRY)
        raise ValueError(f"未知模型: {model_spec}。可用: {available}")
    return path


def _resolve_model_id(model_path: str) -> str:
    """由解析后的模型路径推导对外 model_id（registry 已知路径 → 命名版本，否则用文件名）。"""
    for name, path in MODEL_REGISTRY.items():
        if model_path == path:
            return name
    return Path(model_path).stem
