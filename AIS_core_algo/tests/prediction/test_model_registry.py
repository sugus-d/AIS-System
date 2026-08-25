"""prediction.model_registry — 模型路径解析测试（注册表来自 config.yaml，单源 fail-fast）。"""

from __future__ import annotations

import pytest

import prediction.model_registry as registry


class TestResolveModelPath:
    def test_version_name(self, tmp_path, monkeypatch):
        model_file = tmp_path / "v9.joblib"
        model_file.write_bytes(b"x")
        monkeypatch.setattr(registry, "MODEL_REGISTRY", {"v9": str(model_file)})
        assert registry.resolve_model_path("v9") == str(model_file)

    def test_alias(self, tmp_path, monkeypatch):
        model_file = tmp_path / "v9.joblib"
        model_file.write_bytes(b"x")
        monkeypatch.setattr(registry, "MODEL_REGISTRY", {"v9": str(model_file)})
        monkeypatch.setattr(registry, "_MODEL_ALIASES", {"old_name": "v9"})
        assert registry.resolve_model_path("old_name") == str(model_file)

    def test_direct_path(self, tmp_path):
        """CLI 调试用：未注册字符串视为直接 joblib 路径（存在即返回）。"""
        model_file = tmp_path / "direct.joblib"
        model_file.write_bytes(b"x")
        assert registry.resolve_model_path(str(model_file)) == str(model_file)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="未知模型"):
            registry.resolve_model_path("nonexistent_version_xyz")


class TestResolveRegisteredModel:
    """API 白名单：仅接受注册版本名/别名，禁止任意路径（joblib=pickle，防 RCE）。"""

    def test_registered(self, tmp_path, monkeypatch):
        model_file = tmp_path / "v9.joblib"
        model_file.write_bytes(b"x")
        monkeypatch.setattr(registry, "MODEL_REGISTRY", {"v9": str(model_file)})
        assert registry.resolve_registered_model("v9") == str(model_file)

    def test_alias(self, tmp_path, monkeypatch):
        model_file = tmp_path / "v9.joblib"
        model_file.write_bytes(b"x")
        monkeypatch.setattr(registry, "MODEL_REGISTRY", {"v9": str(model_file)})
        monkeypatch.setattr(registry, "_MODEL_ALIASES", {"old_name": "v9"})
        assert registry.resolve_registered_model("old_name") == str(model_file)

    def test_arbitrary_path_rejected(self):
        """任意路径字符串（即使存在）不被当作模型路径解析。"""
        with pytest.raises(ValueError, match="未知模型"):
            registry.resolve_registered_model("/etc/passwd")


class TestResolveModelId:
    def test_registered_path_returns_name(self, tmp_path, monkeypatch):
        model_file = tmp_path / "v9.joblib"
        model_file.write_bytes(b"x")
        monkeypatch.setattr(registry, "MODEL_REGISTRY", {"v9": str(model_file)})
        assert registry._resolve_model_id(str(model_file)) == "v9"

    def test_unknown_path_returns_stem(self, tmp_path):
        assert registry._resolve_model_id(str(tmp_path / "custom.joblib")) == "custom"
