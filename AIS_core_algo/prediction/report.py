"""报告图渲染 — 热力图/landmark/背部光照/莫尔条纹（物理空间）。

底图用 ROI 网格（完整背部区域），非 mesh_cut（参数化切割后的局部网格）；
landmark 用 flat 物理坐标直线连接（对齐标注平台配色）。
热力图统一 academic 样式（rc_context 局部生效，不污染进程全局 rcParams）；
色限用 `_adaptive_clim` 按分布形态自动选择策略，避免长尾离群拉坏色带。
SHAP 瀑布图见 `prediction.report_waterfall`（独立模块，解耦 shap/waterfall 重依赖）。
"""

from __future__ import annotations

import numpy as np
import open3d as o3d

from prediction.measures import _compute_measures
from utils.logger import logger


def _adaptive_clim(
    values: np.ndarray,
    k: float = 1.5,
    low: float | None = None,
    high: float | None = None,
) -> tuple[float, float]:
    """通用自适应色限：按分布形态自动选择策略（统一各热力图）。

    依据四类热力图实测结论：
      - 尖峰分布（IQR 相对 5/95 跨度极小，如高斯曲率）→ 中位数居中 ± k×IQR，
        主体（≈0）绿色居中，避免长尾把色带拉开成一片冷色；
      - 近对称分布（如平均曲率）→ 中位数居中 ± k×IQR，凸/凹冷暖对称；
      - 右偏分布（如粗糙度）→ Tukey Q1/Q3 ± k×IQR，主体铺满色带，
        长尾离群 clip 成高值色；
    最后 clamp 到物理上下限（如法向角 [0,90]）与数据实际范围，避免色带浪费。
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    q1, q3 = np.percentile(finite, [25, 75])
    p5, p50, p95 = np.percentile(finite, [5, 50, 95])
    iqr = q3 - q1
    spread = p95 - p5
    if spread <= 0:
        vmin, vmax = float(finite.min()), float(finite.max())
    elif iqr < 0.1 * spread:
        # 尖峰：中位数居中（gauss 大部分≈0 → 绿色居中）
        limit = max(k * iqr, 1e-9)
        vmin, vmax = p50 - limit, p50 + limit
    else:
        skew = (p95 - p50) - (p50 - p5)
        if abs(skew) < 0.3 * spread:
            # 近对称：中位数居中
            vmin, vmax = p50 - k * iqr, p50 + k * iqr
        else:
            # 右偏：Tukey 单侧
            vmin, vmax = q1 - k * iqr, q3 + k * iqr
    if low is not None:
        vmin = max(vmin, low)
    if high is not None:
        vmax = min(vmax, high)
    vmin = max(float(finite.min()), float(vmin))
    vmax = min(float(finite.max()), float(vmax))
    if vmax <= vmin:
        return (float(finite.min()), float(finite.max()))
    return (vmin, vmax)


def _visualize(roi_mesh: o3d.geometry.TriangleMesh, flat: dict, out_dir) -> None:
    """曲率 / 粗糙度 / 法向角热力图 + landmark 连线图 + 背部光照 + 莫尔条纹（物理空间渲染）。

    底图用 ROI 网格（完整背部区域），非 mesh_cut（参数化切割后的局部网格）；
    landmark 用 flat 物理坐标直线连接（对齐标注平台配色）。
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    from visualization._render_utils import save_img
    from visualization._style import ACADEMIC_STYLE
    from visualization.back_panels import render_back, render_back_landmarks
    from visualization.heatmap_panels import render_heatmap

    report = out_dir / "report"
    report.mkdir(parents=True, exist_ok=True)
    measures = _compute_measures(roi_mesh)
    vertices = measures["vertices"]
    faces = measures["faces"]

    # 报告图统一 academic 样式——rc_context 局部生效，不污染进程全局 rcParams
    with mpl.rc_context(ACADEMIC_STYLE):
        for name, data, clim_range in [
            # 统一自适应色限：尖峰/对称/右偏自动选策略 + 物理上下限 clamp
            ("curvature_mean", measures["curv_mean"], _adaptive_clim(measures["curv_mean"])),
            ("curvature_gauss", measures["curv_gauss"], _adaptive_clim(measures["curv_gauss"])),
            ("roughness", measures["roughness"], _adaptive_clim(measures["roughness"], low=0)),
            ("normal_angle", measures["normal_angle"], _adaptive_clim(measures["normal_angle"], low=0, high=90)),
        ]:
            if data is None:
                continue
            values = np.asarray(data, dtype=np.float64)
            if not np.isfinite(values).any():
                logger.warning(f"热力图 {name}: 数据全为 NaN，跳过")
                continue
            fig, ax = plt.subplots(figsize=(6, 8))
            render_heatmap(ax, vertices, faces, values, float(clim_range[0]), float(clim_range[1]), name)
            # prediction 报告定制：隐藏坐标轴/外框（论文批量图不受影响，仍保留轴）
            ax.set_axis_off()
            fig.tight_layout()
            out_path = report / f"{name}.png"
            save_img(fig, str(out_path), dpi=150)
            logger.info(f"报告图已保存: {out_path}")

        fig, ax = plt.subplots(figsize=(6, 8))
        render_back_landmarks(ax, vertices, faces, measures["normals"], flat)
        fig.tight_layout()
        out_path = report / "landmarks.png"
        save_img(fig, str(out_path), dpi=150)
        logger.info(f"报告图已保存: {out_path}")

        # 原始背部图像：中国肤色 SSS 光照渲染（Blinn-Phong + 次表面散射近似），无 landmarks
        fig, ax = plt.subplots(figsize=(6, 8))
        render_back(ax, vertices, faces, measures["normals"])
        fig.tight_layout()
        out_path = report / "back.png"
        save_img(fig, str(out_path), dpi=150)
        logger.info(f"报告图已保存: {out_path}")

        # 莫尔条纹图：光照底图 + 奇数带黑纹（明暗体积感）
        from moire.moire import compute_moire_distances
        from visualization.back_panels import compute_phong_colors
        from visualization.moire_panels import render_moire

        fig, ax = plt.subplots(figsize=(6, 8))
        distances = compute_moire_distances(roi_mesh)
        render_moire(ax, vertices, faces, distances, colors=compute_phong_colors(measures["normals"], "pearl"))
        fig.tight_layout()
        out_path = report / "moire.png"
        save_img(fig, str(out_path), dpi=150)
        logger.info(f"报告图已保存: {out_path}")
