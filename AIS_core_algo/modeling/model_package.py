"""模型包加载协议 — joblib 加载 + mtime 缓存 + 按 kind 结构校验。

训练写（save_trained_model 落盘后 round-trip 自检）、预测读（prediction）、
导出复读（commands/export）三方共用同一套加载/校验契约，消除各处裸 joblib.load。
"""

from __future__ import annotations

from pathlib import Path

# 模型包缓存 {路径: (文件 mtime, 模型包)}：mtime 变化时自动重载（重训/恢复模型后无需重启服务）
_pkg_cache: dict[str, tuple[float, dict]] = {}

# 标准模型包必填字段（save_trained_model 落盘结构）
_STANDARD_KEYS = ("model", "scaler", "feature_names", "transform_target")


def validate_model_package(pkg: dict, model_path: str) -> None:
    """按标准字段校验模型包结构契约，缺字段抛 ValueError。

    所有包类型统一校验标准字段——v1.0.0 边界 Ensemble 包在训练落盘时已把
    composite 分量展平到顶层（见 migrate_model_package_flat），与 v0.1.0
    结构对齐，无需按 kind 分支校验。
    """
    for key in _STANDARD_KEYS:
        if key not in pkg:
            raise ValueError(f"模型包缺少字段 {key}: {model_path}")


def load_model_package(model_path: str, use_cache: bool = True) -> dict:
    """加载 joblib 模型包（按文件 mtime 缓存，模型文件更新后自动重载）并校验结构。"""
    import joblib

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"模型权重不存在: {path}。请先训练: python -m modeling.train --scheme v0.1.0"
        )
    mtime = path.stat().st_mtime
    cached = _pkg_cache.get(model_path)
    if use_cache and cached is not None and cached[0] == mtime:
        return cached[1]
    pkg = joblib.load(path)
    validate_model_package(pkg, model_path)
    if use_cache:
        _pkg_cache[model_path] = (mtime, pkg)
    return pkg
