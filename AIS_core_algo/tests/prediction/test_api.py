"""prediction.api — HTTP 端点测试（mock 管线，验证契约：双状态返回差异 + 校验分支）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import prediction.api as api_module
from landmarks.constants import FLAT_KEYS
from prediction.api import app


def _complete_flat_landmarks() -> dict:
    """完整 18 键扁平 landmarks（landmarks 响应模型要求全键）。"""
    return {key: [float(i), 0.0, 0.0] for i, key in enumerate(FLAT_KEYS)}


@pytest.fixture
def client(monkeypatch):
    """Monkeypatch 管线重依赖，返回 TestClient。"""

    def fake_run_landmarks(ply_path: str, subject_id: str, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        (out_dir / "landmarks.json").write_text(json.dumps(_complete_flat_landmarks()), encoding="utf-8")
        (out_dir / "roi.ply").write_bytes(b"roi")
        return out_dir / "landmarks.json"

    def fake_predict_flow(ply_path, subject_id, clinical_path, landmarks_path, model_path, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prediction.json").write_text(
            json.dumps(
                {
                    "subject_id": subject_id,
                    "cobb": 30.0,
                    "severity": "Moderate",
                    "model_id": "v1.0.0",
                    "clinical": {"gender": "Female", "height_cm": 150.0, "weight_kg": 40.0},
                    "indices": {
                        "curvature_index": 1.0,
                        "height_index": 2.0,
                        "normal_angle_index": 3.0,
                        "roughness_index": 4.0,
                        "asymmetric_index": 5.0,
                    },
                    "body_params": {},
                }
            ),
            encoding="utf-8",
        )
        if landmarks_path is None:  # auto 模式：预测流程落盘 landmarks
            (out_dir / "landmarks.json").write_text(json.dumps(_complete_flat_landmarks()), encoding="utf-8")
        report = out_dir / "report"
        report.mkdir(parents=True, exist_ok=True)
        (report / "waterfall.png").write_bytes(b"img")

    monkeypatch.setattr(api_module, "run_landmarks", fake_run_landmarks)
    monkeypatch.setattr(api_module, "_predict_flow", fake_predict_flow)
    return TestClient(app)


_CLINICAL = json.dumps({"gender": "Female", "height_cm": 150.0, "weight_kg": 40.0})
_PLY = {"file": ("s.ply", b"ply-data", "application/octet-stream")}


class TestLandmarksEndpoint:
    def test_ok(self, client):
        resp = client.post("/api/landmarks", files=_PLY, data={"subject_id": "sid"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["subject_id"] == "sid"
        assert len(body["landmarks"]) == 18  # 完整 18 键
        assert body["outputs"]["roi"].endswith("roi.ply")

    def test_non_ply_422(self, client):
        resp = client.post("/api/landmarks", files={"file": ("s.txt", b"x", "text/plain")}, data={"subject_id": "sid"})
        assert resp.status_code == 422

    def test_path_traversal_subject_400(self, client):
        resp = client.post("/api/landmarks", files=_PLY, data={"subject_id": "../evil"})
        assert resp.status_code == 400


class TestPredictEndpoint:
    def test_auto_returns_landmarks_and_roi(self, client):
        resp = client.post("/api/predict", files=_PLY, data={"clinical": _CLINICAL})
        assert resp.status_code == 200
        body = resp.json()
        assert body["cobb"] == 30.0
        assert "landmarks" in body  # auto 模式返回
        assert body["outputs"]["roi"].endswith("roi.ply")
        assert body["outputs"]["waterfall"].endswith("waterfall.png")

    def test_predict_mode_excludes_landmarks_roi(self, client):
        resp = client.post(
            "/api/predict",
            files=_PLY,
            data={"clinical": _CLINICAL, "landmarks": json.dumps(_complete_flat_landmarks())},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "landmarks" not in body  # predict 模式不回显
        assert "roi" not in body["outputs"]
        assert body["outputs"]["waterfall"].endswith("waterfall.png")

    def test_clinical_non_json_400(self, client):
        resp = client.post("/api/predict", files=_PLY, data={"clinical": "not-json"})
        assert resp.status_code == 400

    def test_clinical_missing_field_422(self, client):
        resp = client.post(
            "/api/predict",
            files=_PLY,
            data={"clinical": json.dumps({"gender": "Female"})},  # 缺 height/weight
        )
        assert resp.status_code == 422

    def test_incomplete_landmarks_422(self, client):
        resp = client.post(
            "/api/predict",
            files=_PLY,
            data={"clinical": _CLINICAL, "landmarks": json.dumps({"neck_root_L": [1, 2, 3]})},
        )
        assert resp.status_code == 422

    def test_arbitrary_model_path_rejected_422(self, client):
        """API model 字段白名单：任意路径（joblib=pickle）被拒绝，防反序列化 RCE。"""
        resp = client.post(
            "/api/predict",
            files=_PLY,
            data={"clinical": _CLINICAL, "model": "/etc/passwd"},
        )
        assert resp.status_code == 422
