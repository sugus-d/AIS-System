"""轻量标注图渲染 — ROI mesh + landmarks → landmarks.png。

独立于预测管线：不加载模型、不读 features.csv / prediction.json、不做参数化、
不生成 waterfall。仅复用报告渲染里「mesh 度量 + 标注连线图」这一步，保证
标注完成后的即时标注图与最终预测报告图风格一致（同一渲染函数与参数）。
"""

from __future__ import annotations

from pathlib import Path

import open3d as o3d

from landmarks.complete import complete_landmarks_flat
from landmarks.constants import FLAT_KEYS
from prediction.measures import _compute_measures


def render_landmarks_image(ply_path: str, landmarks_flat: dict, out_dir: Path) -> Path:
    """渲染标注连线图 landmarks.png（不加载模型/特征/参数化）。

    Args:
        ply_path: ROI PLY 路径（编辑后 ROI 或算法 roi.ply）。
        landmarks_flat: 扁平 18 键 landmarks（与标注平台 ground_truth.json 展平后的
            18 键一致；键名见 landmarks.constants.FLAT_KEYS）。
        out_dir: 输出根目录，landmarks.png 写到 out_dir/report/。

    Returns:
        输出 PNG 路径（out_dir/report/landmarks.png）。

    Raises:
        ValueError: mesh 为空 / landmarks 不完整（缺 FLAT_KEYS 键）。
    """
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty():
        raise ValueError("无法加载 ROI mesh")

    missing = [key for key in FLAT_KEYS if key not in landmarks_flat]
    if missing:
        raise ValueError(f"landmarks 不完整，缺少: {missing[:5]}")
    # 完整 18 键时 complete_landmarks_flat 走直通路径；保留调用以与预测链路对齐，
    # 并对可能的 numpy 类型做 JSON 序列化安全转换。
    flat = complete_landmarks_flat(landmarks_flat, mesh)

    measures = _compute_measures(mesh)  # 曲率/粗糙度/法向，仅 mesh 级计算

    import matplotlib as mpl
    from matplotlib import pyplot as plt

    from visualization._render_utils import save_img
    from visualization._style import ACADEMIC_STYLE
    from visualization.back_panels import render_back_landmarks

    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 与 prediction/report.py 的 landmarks.png 渲染参数一致（6×8、dpi 150）
    with mpl.rc_context(ACADEMIC_STYLE):
        fig, ax = plt.subplots(figsize=(6, 8))
        render_back_landmarks(
            ax,
            measures["vertices"],
            measures["faces"],
            measures["normals"],
            flat,
        )
        fig.tight_layout()
        out_path = report_dir / "landmarks.png"
        save_img(fig, str(out_path), dpi=150)

    return out_path
