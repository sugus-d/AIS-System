"""modeling.model_package — 共享模型包加载/缓存/结构校验测试。

训练写、预测读、导出复读共用同一加载器；本测试验证加载契约与 mtime 缓存。
"""

from __future__ import annotations

import joblib
import pytest

from modeling.model_package import load_model_package


class TestLoadModelPackage:
    def test_load_and_cache(self, tmp_path, monkeypatch):
        model_file = tmp_path / "m.joblib"
        model_file.write_bytes(b"x")
        fake_pkg = {"model": object(), "scaler": object(), "feature_names": ["a"], "transform_target": False}
        calls: list = []

        def fake_load(path):
            calls.append(path)
            return fake_pkg

        monkeypatch.setattr(joblib, "load", fake_load)
        assert load_model_package(str(model_file)) is fake_pkg
        assert load_model_package(str(model_file)) is fake_pkg  # 命中 mtime 缓存，不重复 load
        assert len(calls) == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="模型权重不存在"):
            load_model_package(str(tmp_path / "nope.joblib"))

    def test_boundary_ensemble_requires_standard_keys(self, tmp_path, monkeypatch):
        """展平后 boundary 包与标准包统一校验顶层字段（缺 model 即拒绝）。"""
        model_file = tmp_path / "b.joblib"
        model_file.write_bytes(b"x")
        fake_pkg = {"kind": "ridge_boundary_ensemble"}  # 缺 model/scaler/feature_names/transform_target
        monkeypatch.setattr(joblib, "load", lambda path: fake_pkg)
        with pytest.raises(ValueError, match="model"):
            load_model_package(str(model_file))

    def test_standard_package_requires_transform_target(self, tmp_path, monkeypatch):
        model_file = tmp_path / "s.joblib"
        model_file.write_bytes(b"x")
        fake_pkg = {"model": object(), "scaler": object(), "feature_names": ["a"]}  # 缺 transform_target
        monkeypatch.setattr(joblib, "load", lambda path: fake_pkg)
        with pytest.raises(ValueError, match="transform_target"):
            load_model_package(str(model_file))
