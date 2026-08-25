"""prediction.predict._predict_flow — mock 集成测试（不依赖真实 mesh/模型/网络）。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import prediction.predict as predict_module
from landmarks.constants import FLAT_KEYS


def _complete_flat_landmarks() -> dict:
    """18 键扁平 landmarks（值不重要，complete_landmarks_flat 对完整输入直接返回）。"""
    return {key: [float(i), float(i), float(i)] for i, key in enumerate(FLAT_KEYS)}


def _install_mocks(monkeypatch, tmp_path, persist_roi_lm: bool) -> dict:
    """Monkeypatch 全部重依赖，返回记录调用/写入的哨兵对象。"""
    out_dir = tmp_path / "out"
    ply_path = tmp_path / "input.ply"
    ply_path.write_bytes(b"ply")
    landmarks_path = tmp_path / "landmarks.json"
    landmarks_path.write_text(json.dumps(_complete_flat_landmarks()), encoding="utf-8")

    sentinel = {"visualize": 0, "waterfall": 0, "mesh": object()}
    feature_df = pd.DataFrame({"f1": [1.0]})

    monkeypatch.setattr(predict_module, "_load_clinical", lambda clinical_path, subject_id: {"height_cm": 150})
    monkeypatch.setattr(
        predict_module,
        "_run_parameterization",
        lambda subject_id, roi_path, lm_path, out_dir: (sentinel["mesh"], np.zeros((1, 2))),
    )
    monkeypatch.setattr(predict_module, "extract_all", lambda *args, **kwargs: feature_df)
    monkeypatch.setattr(
        predict_module,
        "load_model_package",
        lambda model_path: {"model": object(), "scaler": object(), "feature_names": ["f1"], "transform_target": False},
    )
    monkeypatch.setattr(
        predict_module, "_predict", lambda feature_df, model_pkg: {"cobb": 25.0, "severity": "Moderate"}
    )
    monkeypatch.setattr(predict_module, "_compute_indices", lambda feature_df, model_pkg: {"asymmetric_index": 1.0})
    monkeypatch.setattr(
        predict_module,
        "_compute_body_params",
        lambda gt, subject_id: {"Sh.W": {"info": "肩宽(mm)", "value": 200.0}},
    )
    monkeypatch.setattr(predict_module, "_resolve_model_id", lambda model_path: "v1.0.0")

    def fake_visualize(roi_mesh, flat, out_dir):
        sentinel["visualize"] += 1

    def fake_waterfall(feature_df, model_pkg, out_dir, subject_id, severity, cobb):
        sentinel["waterfall"] += 1

    monkeypatch.setattr(predict_module, "_visualize", fake_visualize)
    monkeypatch.setattr(predict_module, "_render_waterfall", fake_waterfall)
    monkeypatch.setattr(
        predict_module.o3d.io,
        "read_triangle_mesh",
        lambda path: sentinel["mesh"],
    )
    return {"out_dir": out_dir, "ply_path": ply_path, "landmarks_path": landmarks_path, "sentinel": sentinel}


class TestPredictFlow:
    def test_auto_mode_persists_roi_landmarks(self, tmp_path, monkeypatch):
        mocks = _install_mocks(monkeypatch, tmp_path, persist_roi_lm=True)
        predict_module._predict_flow(
            str(mocks["ply_path"]),
            "S001",
            str(tmp_path / "clinical.json"),
            str(mocks["landmarks_path"]),
            "v1.0.0",
            mocks["out_dir"],
            persist_roi_lm=True,
        )
        # 产物齐全 + prediction.json 内容正确
        assert (mocks["out_dir"] / "roi.ply").exists()
        assert (mocks["out_dir"] / "landmarks.json").exists()
        assert (mocks["out_dir"] / "features.csv").exists()
        result = json.loads((mocks["out_dir"] / "prediction.json").read_text(encoding="utf-8"))
        assert result["cobb"] == 25.0
        assert result["severity"] == "Moderate"
        assert result["model_id"] == "v1.0.0"
        assert result["indices"] == {"asymmetric_index": 1.0}
        assert result["body_params"]["Sh.W"]["value"] == 200.0
        assert mocks["sentinel"]["visualize"] == 1
        assert mocks["sentinel"]["waterfall"] == 1

    def test_predict_mode_skips_roi_landmarks(self, tmp_path, monkeypatch):
        mocks = _install_mocks(monkeypatch, tmp_path, persist_roi_lm=False)
        predict_module._predict_flow(
            str(mocks["ply_path"]),
            "S001",
            str(tmp_path / "clinical.json"),
            str(mocks["landmarks_path"]),
            "v1.0.0",
            mocks["out_dir"],
            persist_roi_lm=False,
        )
        # predict 模式不重复产出 roi/landmarks（输入已是 ROI + landmarks）
        assert not (mocks["out_dir"] / "roi.ply").exists()
        assert not (mocks["out_dir"] / "landmarks.json").exists()
        assert (mocks["out_dir"] / "prediction.json").exists()

    def test_incomplete_landmarks_raises(self, tmp_path, monkeypatch):
        """landmarks 缺键且补全无法修复（已知点 < 3）→ 报错。"""
        out_dir = tmp_path / "out"
        ply_path = tmp_path / "input.ply"
        ply_path.write_bytes(b"ply")
        landmarks_path = tmp_path / "landmarks.json"
        landmarks_path.write_text(json.dumps({"neck_root_L": [1, 2, 3]}), encoding="utf-8")
        monkeypatch.setattr(predict_module, "_load_clinical", lambda clinical_path, subject_id: {"height_cm": 150})
        monkeypatch.setattr(
            predict_module.o3d.io,
            "read_triangle_mesh",
            lambda path: type("Mesh", (), {"vertices": np.zeros((3, 3))})(),
        )
        with pytest.raises(ValueError, match="landmarks 不完整"):
            predict_module._predict_flow(
                str(ply_path),
                "S001",
                str(tmp_path / "clinical.json"),
                str(landmarks_path),
                "v1.0.0",
                out_dir,
            )
