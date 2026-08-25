"""黄金值生成器 — 核心算法数值指纹。

在已验证正确的 HEAD 上运行，输出各模块数值指纹（shape + 总和 + md5）。
生成结果粘贴进 test_*_golden.py 的断言常量。

仅在有意的算法变更后重跑（重新确认新值正确后替换断言）。

用法:
  python -m tests.numerics._generate_golden   # 项目根目录运行
"""

# ruff: noqa: T201  # CLI 工具，print 输出指纹即其功能

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "numerics"
RNG_SEED = 2026


def digest(name: str, val: object) -> None:
    """输出统一格式的指纹行：name shape=... sum=... md5=...。"""
    if isinstance(val, dict):
        payload = repr(sorted(val.items())).encode()
        print(f"{name} dict md5={hashlib.md5(payload).hexdigest()}")
        return
    arr = np.ascontiguousarray(np.asarray(val))
    total = float(arr.sum())
    md5 = hashlib.md5(arr.tobytes()).hexdigest()
    print(f"{name} shape={arr.shape} sum={total:.10f} md5={md5}")


def walk_dict(name: str, val: object) -> None:
    if isinstance(val, dict):
        for k, v in val.items():
            walk_dict(f"{name}_{k}", v)
    elif isinstance(val, (list, tuple, np.ndarray)):
        digest(name, np.asarray(val, dtype=float))


def run_features() -> None:
    """M1 特征方案加载（chdir 到镜像目录命中相对路径）。"""
    os.chdir(DATA / "features")
    from features.selectors.schemes import SELECTION_REGISTRY

    for scheme in sorted(SELECTION_REGISTRY.keys()):
        data = SELECTION_REGISTRY[scheme].load()
        for key in ["X_basic", "X_morph", "X_region_full", "y"]:
            arr = data.get(key)
            if arr is not None:
                digest(f"feat_{scheme}_{key}", arr)


def run_roi() -> object:
    """M2 ROI 提取端到端；返回 ROI 网格供 M3/M5 使用。"""
    import open3d as o3d

    from mesh.roi.pipeline import run_roi_pipeline

    mesh = o3d.io.read_triangle_mesh(str(DATA / "mesh" / "STD_fuse_mesh_20250619.ply"))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles)
    roi_v, roi_t = run_roi_pipeline(vertices, triangles)
    digest("roi_v", roi_v)
    digest("roi_t", roi_t)
    roi_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(roi_v),
        o3d.utility.Vector3iVector(roi_t),
    )
    out = DATA / "mesh" / "roi_S0006.ply"
    if not out.exists():
        o3d.io.write_triangle_mesh(str(out), roi_mesh)
    return roi_mesh


def run_landmarks(roi_mesh: object) -> dict:
    """M3 Landmark 端到端。"""
    from landmarks.extract import extract_landmarks

    lms = extract_landmarks(roi_mesh)
    for key in sorted(lms):
        if not key.endswith("_debug"):
            walk_dict(f"lm_{key}", lms[key])
    return lms


def run_training() -> None:
    """M4 训练内核：纯函数 + 端到端迷你 CV。"""
    rng = np.random.default_rng(RNG_SEED)
    X = rng.normal(size=(60, 12))
    y = rng.uniform(5, 60, size=60)

    from modeling.training.feature_selector import (
        _compute_ci_per_fold,
        _dedup_by_corr,
        _hybrid_scores,
        _select_morph,
        _select_region,
    )
    from modeling.training.hp_searchers._search_utils import _inject_weight_params

    digest("hybrid_scores", _hybrid_scores(X, y))
    digest("dedup_by_corr", _dedup_by_corr(X, y))
    digest("select_morph", _select_morph(X, y))
    digest("select_region", _select_region(X, y))

    groups = {f"m{i}|dm": list(range(i * 3, i * 3 + 3)) for i in range(4)}
    ci_tr, ci_te_fn = _compute_ci_per_fold(X, y, groups)
    digest("ci_tr", ci_tr)
    digest("ci_te", ci_te_fn(X))

    grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 8, 12], "learning_rate": [0.05, 0.1, 0.2, 0.5]}
    digest("cv_inject", _inject_weight_params(grid))
    digest("cv_inject_wide", _inject_weight_params(grid, wide=True))

    from modeling.training.hp_searchers import _narrow_grid as tr_narrow

    digest("tr_narrow", tr_narrow({"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}, grid))

    from modeling.models import REGISTRY

    for name, cls in sorted(REGISTRY.items()):
        try:
            params = cls().get_param_space()
        except Exception:
            continue
        digest(f"model_{name}_space", params)

    # 端到端迷你 CV（确定性合成数据）
    rng2 = np.random.default_rng(RNG_SEED + 1)
    n = 30
    Xb = rng2.normal(size=(n, 5))
    Xm = rng2.normal(size=(n, 12))
    Xr = rng2.normal(size=(n, 60))
    y2 = rng2.uniform(5, 60, size=n)
    col_names = [f"m{i}|dm" for i in range(20)]

    from modeling.contracts import FeatureSet, TrainingConfig
    from modeling.training.trainer import Trainer

    # Trainer + per-fold 筛选（无 HP 搜索，保持确定性），
    # 覆盖折划分→特征筛选→标准化→训练→预测→聚合全流程。
    config = TrainingConfig(
        models=["Ridge"],
        data_splitter="kfold",
        data_splitter_params={"n_splits": 2, "n_repeats": 2},
        hp_searcher="none",
        transform_target=True,
        feature_selector="per_fold",
    )
    feature_set = FeatureSet(
        name="e2e", y=y2, X=Xb, feature_names=[],
        X_raw_blocks={"basic": Xb, "morph": Xm, "region": Xr},
        region_column_names=col_names,
    )
    result = Trainer(config).train(feature_set)[0]
    digest("trainer_e2e_preds", result.predictions)


_BILATERAL_KEYS = {"neck_root", "shoulder_transition", "scapular_peaks", "axilla", "waist"}


def flatten_landmarks(lms: dict) -> dict:
    """extract_landmarks 已输出扁平语义键；过滤出纯 landmark 键（FLAT_KEYS）。"""
    from landmarks.constants import FLAT_KEYS

    return {key: lms[key] for key in FLAT_KEYS if key in lms}


def run_parameterization(roi_mesh: object, landmarks: dict) -> None:
    """M5 参数化：地标匹配 + 调和参数化 + 测地边界。"""
    from parameterization.geodesic_cut import geodesic_boundary
    from parameterization.harmonic import harmonic_parameterize
    from parameterization.landmark_io import find_landmark_vertices
    from parameterization.template import TEMPLATE_LANDMARKS

    flat = flatten_landmarks(landmarks)
    k, y_uv = find_landmark_vertices(roi_mesh, flat, TEMPLATE_LANDMARKS)
    digest("param_k", k)
    digest("param_y", y_uv)

    uv_mesh, uv = harmonic_parameterize(roi_mesh, k, y_uv)
    digest("param_uv", uv)

    # 简化网格后跑测地边界（生产路径同样先简化，地标在简化网格上重新匹配）

    simple = roi_mesh.simplify_quadric_decimation(target_number_of_triangles=3000)
    V = np.asarray(simple.vertices, dtype=np.float64)
    F = np.asarray(simple.triangles)
    k_simple, y_simple = find_landmark_vertices(simple, flat, TEMPLATE_LANDMARKS)
    # geodesic_boundary 要求 k 与 TEMPLATE_LANDMARKS 全量对齐：
    # 真实匹配的用地标索引，缺失的用确定性合法顶点补齐（保证测地路径计算有效）
    matched_names = [name for name in TEMPLATE_LANDMARKS if name in flat]
    k_map = dict(zip(matched_names, k_simple, strict=False))
    y_map = dict(zip(matched_names, y_simple, strict=False))
    k_full, y_full = [], []
    for offset, (name, uv) in enumerate(TEMPLATE_LANDMARKS.items()):
        if name in k_map:
            k_full.append(k_map[name])
            y_full.append(y_map[name])
        else:
            k_full.append(int(np.argmax(V[:, 1] - offset * 1e-6)))
            y_full.append(np.asarray(uv, dtype=float))
    k_full = np.array(k_full)
    y_full = np.array(y_full)
    outer_names = [name for name in TEMPLATE_LANDMARKS if not name.endswith("_spine_point")]
    boundary_v, boundary_f = geodesic_boundary(V, F, k_full, y_full, outer_names)
    digest("param_boundary_v", boundary_v)
    digest("param_boundary_f", boundary_f)


def run_utils(roi_mesh: object) -> None:
    """M6 utils 数值地基：合成轮廓 + 真实网格。"""
    rng = np.random.default_rng(RNG_SEED)
    # 确定性合成轮廓：半圆 + 噪声
    theta = np.linspace(0, np.pi, 200)
    contour = np.column_stack([100 * np.cos(theta), 100 * np.sin(theta)]) + rng.normal(0, 0.1, size=(200, 2))

    from landmarks.angle import compute_lateral_angle_profile
    from landmarks.contour import extract_lower_boundary_per_integer_x, resample_polyline_uniform
    from landmarks.geometry import is_contour_ccw
    from landmarks.signal_ops import compute_derivatives_from_xy, smooth_contour

    digest("u_resample", resample_polyline_uniform(contour, step=5.0))
    digest("u_lower", extract_lower_boundary_per_integer_x(contour))
    smoothed = smooth_contour(contour)
    digest("u_smooth", smoothed)
    derivative = compute_derivatives_from_xy(smoothed[:, 0], smoothed[:, 1])
    digest("u_derivative", derivative)
    digest("u_ccw", np.array([is_contour_ccw(contour)]))
    sampled_pts, angle_values = compute_lateral_angle_profile(contour, 10.0)
    digest("u_angle_sampled", sampled_pts)
    digest("u_angle_values", angle_values)

    from utils.mesh import estimate_vertex_radius, lift_2d_to_vertex

    vertices = np.asarray(roi_mesh.vertices, dtype=np.float64)
    radius = estimate_vertex_radius(vertices, nb_neighbors=8)
    digest("u_radius", radius)
    pts2d = np.array([[0.0, 0.0], [100.0, -100.0]])
    lifted = lift_2d_to_vertex(vertices, pts2d)
    digest("u_lift", lifted)


def main() -> None:
    roi_mesh = run_roi()
    landmarks = run_landmarks(roi_mesh)
    run_features()
    run_training()
    run_utils(roi_mesh)
    run_parameterization(roi_mesh, landmarks)


if __name__ == "__main__":
    main()
