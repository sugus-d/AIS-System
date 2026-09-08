"""prediction.render — 轻量标注图渲染测试（真实 mesh 渲染 + API 契约）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from fastapi.testclient import TestClient

import prediction.api as api_module
from landmarks.constants import FLAT_KEYS
from prediction.api import app
from prediction.render import render_landmarks_image


def _sphere_ply(tmp_path: Path) -> Path:
    """生成球面 ROI PLY，供渲染测试用（与真实 ROI 同为三角网格）。"""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    ply = tmp_path / "roi.ply"
    o3d.io.write_triangle_mesh(str(ply), mesh)
    return ply


def _complete_landmarks() -> dict:
    """完整 18 键扁平 landmarks（仅需坐标结构合法，渲染不校验解剖正确性）。"""
    rng = np.random.default_rng(0)
    return {key: rng.uniform(-1.0, 1.0, 3).tolist() for key in FLAT_KEYS}


class TestRenderLandmarksImage:
    def test_renders_png(self, tmp_path):
        ply = _sphere_ply(tmp_path)
        out = render_landmarks_image(str(ply), _complete_landmarks(), tmp_path / "out")
        assert out.name == "landmarks.png"
        assert out.exists()
        assert out.stat().st_size > 0

    def test_missing_key_raises(self, tmp_path):
        ply = _sphere_ply(tmp_path)
        landmarks = _complete_landmarks()
        landmarks.pop(FLAT_KEYS[0])
        with pytest.raises(ValueError, match="不完整"):
            render_landmarks_image(str(ply), landmarks, tmp_path / "out")

    def test_empty_mesh_raises(self, tmp_path):
        empty = tmp_path / "empty.ply"
        empty.write_bytes(b"")
        with pytest.raises(ValueError, match="无法加载"):
            render_landmarks_image(str(empty), _complete_landmarks(), tmp_path / "out")


class TestRenderEndpoint:
    @pytest.fixture
    def client(self, monkeypatch):
        """Mock 重依赖（真实渲染在 TestRenderLandmarksImage 已覆盖）。"""

        def fake_render(ply_path: str, landmarks_flat: dict, out_dir: Path) -> Path:
            report = Path(out_dir) / "report"
            report.mkdir(parents=True, exist_ok=True)
            png = report / "landmarks.png"
            png.write_bytes(b"png")
            return png

        monkeypatch.setattr(api_module, "render_landmarks_image", fake_render)
        return TestClient(app)

    def test_ok(self, client):
        resp = client.post(
            "/api/render",
            files={"file": ("roi.ply", b"ply", "application/octet-stream")},
            data={"subject_id": "sid", "landmarks": json.dumps(_complete_landmarks())},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["subject_id"] == "sid"
        assert body["outputs"]["landmarks"].endswith("landmarks.png")

    def test_incomplete_landmarks_422(self, client):
        resp = client.post(
            "/api/render",
            files={"file": ("roi.ply", b"ply", "application/octet-stream")},
            data={"landmarks": json.dumps({"neck_root_L": [1, 2, 3]})},
        )
        assert resp.status_code == 422

    def test_landmarks_non_json_400(self, client):
        resp = client.post(
            "/api/render",
            files={"file": ("roi.ply", b"ply", "application/octet-stream")},
            data={"landmarks": "not-json"},
        )
        assert resp.status_code == 400

    def test_non_ply_422(self, client):
        resp = client.post(
            "/api/render",
            files={"file": ("roi.txt", b"x", "text/plain")},
            data={"landmarks": json.dumps(_complete_landmarks())},
        )
        assert resp.status_code == 422

    def test_path_traversal_subject_400(self, client):
        resp = client.post(
            "/api/render",
            files={"file": ("roi.ply", b"ply", "application/octet-stream")},
            data={"subject_id": "../evil", "landmarks": json.dumps(_complete_landmarks())},
        )
        assert resp.status_code == 400
