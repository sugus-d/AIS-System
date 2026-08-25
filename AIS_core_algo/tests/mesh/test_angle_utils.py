"""Tests for angle_utils: _compute_angle_and_cosine clockwise angle."""

# ruff: noqa: T201 — 测试故意 print 角度剖面摘要

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 非交互后端，测试环境无需显示
import matplotlib.pyplot as plt
import numpy as np
import pytest

from landmarks.angle import (
    _compute_angle_and_cosine,
    compute_lateral_angle_at_point,
)
from landmarks.lateral_profile import extract_split_contours
from mesh.preprocess import preprocess_back_scan_mesh
from mesh.roi_extract import extract_back_roi
from utils.mesh import load_mesh_by_project


class TestComputeAngleAndCosine:
    """_compute_angle_and_cosine: 0-360° 顺时针角测试。

    所有测试用例 cand_pt = (0,0)，pt_before / pt_after 的值直接就是向量，
    方便直观判断角度方向。
    """

    # ── 基本几何形状 ──────────────────────────────────────────────

    def test_concave_v_left(self):
        """左肩凹 V（有手臂）：clockwise < 180°"""
        _, cw, _, _ = _compute_angle_and_cosine(
            np.array([-15.0, -15.0]),  # before: 左下
            np.array([-10.0, 15.0]),  # after:  左上
            np.array([0.0, 0.0]),  # cand
        )
        assert 80 < cw < 175, f"expected concave (<180°), got {cw:.1f}°"

    def test_convex_lambda_left(self):
        """左肩凸 Λ（无手臂）：clockwise > 185°"""
        _, cw, _, _ = _compute_angle_and_cosine(
            np.array([12.0, -20.0]),  # before: 右下
            np.array([15.0, 20.0]),  # after:  右上
            np.array([0.0, 0.0]),  # cand
        )
        assert cw > 185, f"expected convex (>185°), got {cw:.1f}°"

    def test_concave_v_right(self):
        """右肩凹 V（有手臂）：clockwise > 185°（contour-order 凹角在右侧 > 180°）"""
        _, cw, _, _ = _compute_angle_and_cosine(
            np.array([12.0, -15.0]),  # before: 右下
            np.array([10.0, 15.0]),  # after:  右上
            np.array([0.0, 0.0]),  # cand（偏内侧）
        )
        assert cw > 185, f"expected concave (>185°), got {cw:.1f}°"

    def test_convex_lambda_right(self):
        """右肩凸 Λ（无手臂）：clockwise < 175°（contour-order 凸角在右侧 < 180°）"""
        _, cw, _, _ = _compute_angle_and_cosine(
            np.array([-15.0, -20.0]),  # before: 左下
            np.array([-18.0, 20.0]),  # after:  左上
            np.array([0.0, 0.0]),  # cand（凸顶）
        )
        assert cw < 175, f"expected convex (<175°), got {cw:.1f}°"

    # ── 边界情况 ──────────────────────────────────────────────

    def test_collinear(self):
        """三点共线 → clockwise ≈ 180°"""
        _, cw, _, _ = _compute_angle_and_cosine(
            np.array([-10.0, 0.0]),
            np.array([10.0, 0.0]),
            np.array([0.0, 0.0]),
        )
        assert abs(cw - 180) < 1e-6

    def test_collinear_vertical(self):
        """三点垂直共线 → clockwise ≈ 180°"""
        _, cw, _, _ = _compute_angle_and_cosine(
            np.array([0.0, 20.0]),
            np.array([0.0, -20.0]),
            np.array([0.0, 0.0]),
        )
        assert abs(cw - 180) < 1e-6

    def test_sharp_concave(self):
        """锐角凹（~60°）：clockwise ≈ 67°"""
        _, cw, _, _ = _compute_angle_and_cosine(
            np.array([-15.0, -8.0]),
            np.array([-10.0, 8.0]),
            np.array([0.0, 0.0]),
        )
        assert 30 < cw < 100, f"expected sharp concave, got {cw:.1f}°"

    def test_wide_convex(self):
        """接近直线的凸：clockwise ≈ 189°"""
        _, cw, _, _ = _compute_angle_and_cosine(
            np.array([5.0, -18.0]),
            np.array([-2.0, 18.0]),
            np.array([0.0, 0.0]),
        )
        assert 175 < cw < 200, f"expected wide convex near 180°, got {cw:.1f}°"

    # ── 接近水平（向量在 cand 两侧）────────────────────────────

    def test_near_horizontal_concave(self):
        """几乎水平，before/after 在 cand 两侧：cw 略 < 180°（凹侧）"""
        _, cw, _, _ = _compute_angle_and_cosine(
            np.array([-10.0, 2.0]),  # before: 左偏上
            np.array([10.0, -1.0]),  # after:  右偏下
            np.array([0.0, 0.0]),  # cand
        )
        assert 170 < cw < 178, f"expected near-horizontal concave, got {cw:.1f}°"

    def test_near_horizontal_convex(self):
        """几乎水平，before/after 在 cand 两侧：cw 略 > 180°（凸侧）"""
        _, cw, _, _ = _compute_angle_and_cosine(
            np.array([-10.0, -2.0]),  # before: 左偏下
            np.array([10.0, 1.0]),  # after:  右偏上
            np.array([0.0, 0.0]),  # cand
        )
        assert 182 < cw < 190, f"expected near-horizontal convex, got {cw:.1f}°"

    # ── 等边/等腰 ──────────────────────────────────────────────

    def test_right_angle(self):
        """直角（90°）"""
        _, cw, _, _ = _compute_angle_and_cosine(
            np.array([0.0, 10.0]),  # 正上方
            np.array([10.0, 0.0]),  # 正右方
            np.array([0.0, 0.0]),
        )
        assert abs(cw - 90) < 1e-6, f"expected 90°, got {cw:.1f}°"

    def test_symmetric_concave(self):
        """等腰凹 V：before/after 对称"""
        _, cw, _, _ = _compute_angle_and_cosine(
            np.array([-15.0, -15.0]),
            np.array([-15.0, 15.0]),
            np.array([0.0, 0.0]),
        )
        assert 80 < cw < 150, f"expected symmetric concave, got {cw:.1f}°"

    # ── 退化情况 ──────────────────────────────────────────────

    def test_before_degenerate(self):
        """前点与候选点重合 → 返回退化值"""
        cos, cw, before_d, after_d = _compute_angle_and_cosine(
            np.array([0.0, 0.0]),  # before == cand
            np.array([-10.0, 10.0]),
            np.array([0.0, 0.0]),
        )
        assert cos == 1.0
        assert cw == 0.0

    def test_after_degenerate(self):
        """后点与候选点重合 → 返回退化值"""
        cos, cw, before_d, after_d = _compute_angle_and_cosine(
            np.array([-10.0, 10.0]),
            np.array([0.0, 0.0]),  # after == cand
            np.array([0.0, 0.0]),
        )
        assert cos == 1.0
        assert cw == 0.0

    def test_both_degenerate(self):
        """前后点都与候选点重合 → 返回退化值"""
        cos, cw, _, _ = _compute_angle_and_cosine(
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
        )
        assert cos == 1.0
        assert cw == 0.0

    # ── 3D 坐标（Z 忽略）──────────────────────────────────────

    def test_with_3d_coords(self):
        """3D 坐标传入，只使用前两维"""
        _, cw, _, _ = _compute_angle_and_cosine(
            np.array([-15.0, -15.0, 30.0]),
            np.array([-10.0, 15.0, 28.0]),
            np.array([0.0, 0.0, 0.0]),
        )
        assert 80 < cw < 175, f"expected same as 2D, got {cw:.1f}°"

    # ── 值域完整性 ──────────────────────────────────────────────

    def test_clockwise_in_full_range(self):
        """顺时针角落在 [0, 360) 范围内"""
        pts = [
            (np.array([-15.0, -15.0]), np.array([-10.0, 15.0]), np.array([0.0, 0.0])),
            (np.array([12.0, -20.0]), np.array([15.0, 20.0]), np.array([0.0, 0.0])),
            (np.array([0.0, 10.0]), np.array([10.0, 0.0]), np.array([0.0, 0.0])),
            (np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])),
        ]
        for before, after, cand in pts:
            _, cw, _, _ = _compute_angle_and_cosine(before, after, cand)
            assert 0 <= cw < 360, f"clockwise={cw} out of range"


class TestAgainstRealContours:
    """在光滑密集轮廓上验证 concavity/convexity 区分（相邻点即稳定）。"""

    def _fake_contour_arm_left(self) -> np.ndarray:
        """光滑密集左肩凹轮廓（neck → shoulder dip → arm）。"""
        y = np.linspace(20, 140, 250)
        x_base = -30 - 35 * (y - 20) / 120
        dip = 22 * np.exp(-(((y - 75) / 12) ** 2))
        return np.column_stack([x_base + dip, y])

    def _fake_contour_noarm_left(self) -> np.ndarray:
        """光滑密集左肩凸轮廓（neck → shoulder peak → torso）。"""
        y = np.linspace(20, 140, 250)
        x_base = -30 - 35 * (y - 20) / 120
        bump = -22 * np.exp(-(((y - 80) / 12) ** 2))
        return np.column_stack([x_base + bump, y])

    def test_arm_left_shoulder_is_concave(self):
        """有手臂左肩：转点应为凹角（顺时针 < 175°）。"""
        contour = self._fake_contour_arm_left()
        # 凹底（X 最大处）
        di = int(np.argmax(contour[:, 0]))
        dip_pt = contour[di]
        cw = compute_lateral_angle_at_point(contour, dip_pt, distance=10.0)[1]
        assert cw < 175, f"arm-left dip expected concave, got {cw:.1f}°"

    def test_noarm_left_shoulder_is_convex(self):
        """无手臂左肩：转点应为凸角（顺时针 > 185°）。"""
        contour = self._fake_contour_noarm_left()
        # 峰顶（X 最小处）
        pi = int(np.argmin(contour[:, 0]))
        peak_pt = contour[pi]
        cw = compute_lateral_angle_at_point(contour, peak_pt, distance=10.0)[1]
        assert cw > 185, f"noarm-left peak expected convex, got {cw:.1f}°"


# ── 真实网格集成测试 ──────────────────────────────────────

MESH_ID = "17-10745"
# 测试已按领域分组（tests/mesh/），上溯两级到项目根
_ROOT = Path(__file__).resolve().parents[2]
_MESH_DIR = os.path.normpath(os.path.join(_ROOT, "data", "mesh", MESH_ID))
_HAS_MESH = len(glob.glob(os.path.join(_MESH_DIR, "STD_fuse_mesh*.ply"))) > 0

VIZ_DIR = os.path.join(_ROOT, "results", "test")
VIZ_DIR = os.path.normpath(VIZ_DIR)


def _build_angle_profile(contour: np.ndarray, distance: float = 10.0, margin: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """沿轮廓逐点计算顺时针角，返回 (y_vals, angle_vals) 去掉首尾各 margin 点。"""
    n = len(contour)
    y_vals: list[float] = []
    cw_vals: list[float] = []
    for i in range(margin, n - margin):
        pt = contour[i]
        cw = compute_lateral_angle_at_point(contour, pt, distance=distance)[1]
        y_vals.append(float(pt[1]))
        cw_vals.append(float(cw))
    return np.array(y_vals), np.array(cw_vals)


def _build_angle_profile_full(contour: np.ndarray, distance: float = 10.0) -> np.ndarray:
    """沿轮廓所有点算角（含端点退化值），用于 scatter 颜色映射。"""
    return np.array([compute_lateral_angle_at_point(contour, pt, distance=distance)[1] for pt in contour])


def _region_label(y: float) -> str:
    """根据 Y 坐标返回身体区域名称。"""
    if y > 40:
        return "shoulder"
    elif y > -30:
        return "torso/axilla"
    elif y > -80:
        return "waist"
    else:
        return "hip"


def _plot_angle_profile(
    left_c: np.ndarray,
    right_c: np.ndarray,
    mesh_vertices: np.ndarray,
    save_path: str,
    distance: float = 30.0,
) -> str:
    """生成角度剖面图：mesh 顶点云 + 轮廓叠加 与 angle-vs-Y 曲线。

    Args:
        left_c: 左轮廓 (M, 3)
        right_c: 右轮廓 (M, 3)
        mesh_vertices: 全网格顶点 (N, 3)，用于背景显示真实边界
        save_path: 输出路径
        distance: 角度计算弧距 (mm)

    Returns:
        保存后的图片路径
    """
    cwL_full = _build_angle_profile_full(left_c, distance=distance)
    cwR_full = _build_angle_profile_full(right_c, distance=distance)
    yL, cwL = _build_angle_profile(left_c, distance=distance)
    yR, cwR = _build_angle_profile(right_c, distance=distance)

    # 裁剪首尾不稳定区（端点处轮廓点较少）
    trim = 10
    left_t = left_c[trim:-trim]
    right_t = right_c[trim:-trim]
    cwL_t = cwL_full[trim:-trim]
    cwR_t = cwR_full[trim:-trim]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # ── 左图：mesh 顶点云（灰底，降采样）+ 轮廓（彩色）──
    step = max(1, len(mesh_vertices) // 5000)
    ax1.scatter(
        mesh_vertices[::step, 0], mesh_vertices[::step, 1], c="lightgray", s=1, alpha=0.3, label="mesh vertices"
    )
    sc1 = ax1.scatter(
        left_t[:, 0], left_t[:, 1], c=cwL_t, cmap="coolwarm", s=15, vmin=0, vmax=360, edgecolors="k", linewidth=0.3
    )
    ax1.scatter(
        right_t[:, 0], right_t[:, 1], c=cwR_t, cmap="coolwarm", s=15, vmin=0, vmax=360, edgecolors="k", linewidth=0.3
    )
    ax1.set_title("Mesh + Contour (color = angle°)")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.2)
    plt.colorbar(sc1, ax=ax1, label="cw°")

    # ── 右图：Angle vs Y 曲线 ──
    ax2.plot(cwL, yL, "b-", label="left (CW)", linewidth=1.5)
    ax2.plot(cwR, yR, "r-", label="right (CCW→swap)", linewidth=1.5)

    for angle, clr in [(175, "green"), (180, "black"), (185, "green")]:
        ax2.axvline(angle, color=clr, linestyle="--", alpha=0.5, linewidth=1)
    ax2.axvspan(175, 185, color="gray", alpha=0.08)
    ax2.text(180, yR.max(), " 180°", fontsize=9, alpha=0.5)

    y_min, y_max = yL.min(), yL.max()
    for y_lo, y_hi, label, color in [
        (y_min, -200, "hip", "mistyrose"),
        (-200, -80, "waist", "lightyellow"),
        (-80, 40, "torso/axilla", "honeydew"),
        (40, y_max, "shoulder", "lightcyan"),
    ]:
        ax2.axhspan(y_lo, y_hi, facecolor=color, alpha=0.3)
        ax2.text(365, (y_lo + y_hi) / 2, label, fontsize=8, va="center", ha="left", alpha=0.6)

    ax2.set_title("Lateral angle profile")
    ax2.set_xlabel("Clockwise angle (°)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_xlim(0, 360)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("S0006 — body contour lateral angle", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


@pytest.mark.skipif(not _HAS_MESH, reason=f"Mesh not found: {_MESH_DIR}")
class TestRealMeshAngleProfile:
    """在真实 3D 扫描网格上验证外侧角计算的稳定性与合理性。"""

    @classmethod
    @pytest.fixture(scope="class")
    def mesh_data(cls) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mesh = load_mesh_by_project(MESH_ID)

        # Step 1: ROI 提取——去衣、去手臂
        roi_mesh = extract_back_roi(mesh)

        # Step 2: Envelope 重建 + 平滑
        processed_mesh, _ = preprocess_back_scan_mesh(roi_mesh)

        # Step 3: 提取实际边界轮廓
        vertices = np.asarray(processed_mesh.vertices, dtype=np.float64)
        left_c, right_c = extract_split_contours(vertices)

        vertices = np.asarray(processed_mesh.vertices, dtype=np.float64)
        return left_c, right_c, vertices

    def test_contour_structure(self, mesh_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """左右轮廓均非空且有正确的 X 分布。"""
        left_c, right_c, _ = mesh_data
        assert len(left_c) > 10, f"left contour too short: {len(left_c)}"
        assert len(right_c) > 10, f"right contour too short: {len(right_c)}"
        # 左轮廓 X 中位值应小于右轮廓 X 中位值
        assert np.median(left_c[:, 0]) < np.median(right_c[:, 0]), "left contour should be left of right contour"

    def test_left_angle_profile_stable(self, mesh_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """左轮廓逐点外侧角：全部在 [0,360) 内，无 NaN，无退化。"""
        left_c, _, _ = mesh_data
        y, cw = _build_angle_profile(left_c, distance=10.0)
        assert len(cw) > 20, f"not enough valid points: {len(cw)}"
        assert np.all(np.isfinite(cw)), "NaN or Inf detected in angle profile"
        assert np.all((cw >= 0) & (cw < 360)), "angles out of [0, 360) range"

    def test_right_angle_profile_stable(self, mesh_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """右轮廓逐点外侧角：全部在 [0,360) 内，无 NaN，无退化。"""
        _, right_c, _ = mesh_data
        y, cw = _build_angle_profile(right_c, distance=10.0)
        assert len(cw) > 20, f"not enough valid points: {len(cw)}"
        assert np.all(np.isfinite(cw)), "NaN or Inf detected in angle profile"
        assert np.all((cw >= 0) & (cw < 360)), "angles out of [0, 360) range"

    def test_angle_smoothness(self, mesh_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """上半部分（y > -400）角度剖面应整体平滑：大部分相邻差 < 120°。

        允许少量孤立跳变（髋部转折、腰部极凹点），但如果超过 15% 的点
        有大幅跳变说明方向判断有问题。
        """
        left_c, right_c, _ = mesh_data
        for side, contour in [("left", left_c), ("right", right_c)]:
            y, cw = _build_angle_profile(contour, distance=10.0)
            upper = y > -400.0
            if upper.sum() < 10:
                continue
            diffs = np.abs(np.diff(cw[upper]))
            bad_ratio = (diffs > 120.0).sum() / len(diffs)
            if bad_ratio > 0.15:
                n_bad = (diffs > 120.0).sum()
                pytest.fail(
                    f"{side} contour (y > -400): {n_bad}/{len(diffs)} diffs > 120° "
                    f"({bad_ratio:.0%}), indicating systematic orientation error"
                )

    def test_shoulder_region_anatomically_plausible(self, mesh_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """肩部区域角度应与已知解剖结构一致（S0006 左有手臂、右无手臂）。"""
        left_c, right_c, _ = mesh_data
        distance = 10.0

        # 用 Y 值过滤肩部区域（轮廓为 CW 排列，不能按 index 取顶部）
        def _shoulder_points(contour: np.ndarray) -> np.ndarray:
            y_top: float = float(contour[:, 1].max())
            y_bot: float = float(contour[:, 1].min())
            threshold: float = y_top - 0.15 * (y_top - y_bot)
            return contour[contour[:, 1] >= threshold]

        # 左肩区域：有手臂，应 < 175°（凹角）
        shoulder_left = _shoulder_points(left_c)
        cw_left = np.array([compute_lateral_angle_at_point(left_c, pt, distance=distance)[1] for pt in shoulder_left])
        min_cw = cw_left.min()
        assert min_cw < 175, f"left shoulder (has arm) expected concave <175°, got min={min_cw:.1f}"

        # 右肩区域：无手臂，应 > 185°（凸角）
        shoulder_right = _shoulder_points(right_c)
        cw_right = np.array(
            [compute_lateral_angle_at_point(right_c, pt, distance=distance)[1] for pt in shoulder_right]
        )
        max_cw = cw_right.max()
        assert max_cw > 185, f"right shoulder (no arm) expected convex >185°, got max={max_cw:.1f}"

    def test_print_angle_summary(self, mesh_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        """打印角度剖面摘要 + 生成可视化。"""
        left_c, right_c, vertices = mesh_data
        save_path = os.path.join(VIZ_DIR, "angle_profile_S0006.png")
        _plot_angle_profile(left_c, right_c, vertices, save_path)
        print(f"\nVisualization saved: {save_path}")

        print("\n" + "=" * 70)
        print("Real-mesh angle profile — S0006")
        print("=" * 70)

        for side, contour in [("LEFT", left_c), ("RIGHT", right_c)]:
            y, cw = _build_angle_profile(contour, distance=10.0)
            print(f"\n--- {side} contour ({len(contour)} pts) ---")
            print(f"{'Y (mm)':>8} {'Angle(°)':>9} {'Region':>14}")
            print("-" * 35)

            # 等间隔采 8 个点打印
            n = len(y)
            for frac in np.linspace(0, 1, 8):
                i = int(frac * (n - 1))
                y_reg = y[i]
                print(f"{y_reg:8.1f} {cw[i]:9.1f} {_region_label(y_reg):>14}")

            # 统计
            print(f"  angle range: [{cw.min():.1f}, {cw.max():.1f}]")
            print(f"  mean ± std:  {cw.mean():.1f} ± {cw.std():.1f}")
            n_near_180 = int(((cw > 160) & (cw < 200)).sum())
            print(f"  points near 180° (160-200): {n_near_180}/{len(cw)}")
