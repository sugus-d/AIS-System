"""编排器：从背部表面网格提取全部双侧 landmark。"""

import numpy as np
import open3d as o3d

from mesh.curvature import compute_mean_curvature
from utils.logger import logger
from utils.mesh import lift_or_raise

from .axilla import detect_axilla_strips
from .lateral_profile import compute_width_profile, extract_split_contours
from .neck_root import detect_neck_root_strips
from .scapular_peak import detect_scapular_peak
from .shoulder_transition import detect_shoulder_transition
from .spine import derive_spine_points
from .waist import detect_waist

_MIN_LANDMARK_VERTS = 50  # 顶点数下限（太少无法提取 landmark）


def extract_landmarks(
    mesh: o3d.geometry.TriangleMesh,
    is_debug: bool = True,
) -> dict:
    """从背部表面网格提取双侧解剖学标志点（landmarks）。

    本函数为 orchestrator：按步骤构建横向剖面、检测各 bilateral landmark，
    再由双侧点推导脊柱点。函数只做坐标处理与流程调度，
    不实现任何复杂的几何运算（遵循可视化/计算分离原则）。

    Args:
        mesh: open3d TriangleMesh 对象，包含顶点与面信息。
        is_debug: 是否在结果中包含调试数据。

    Returns:
        dict: 包含 neck_root、shoulder_transition 等 landmark 的字典。
    """
    vertices: np.ndarray = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) < _MIN_LANDMARK_VERTS:
        logger.error("Too few vertices to extract landmarks.")
        raise ValueError("Too few vertices to extract landmarks.")

    y_min: float = float(vertices[:, 1].min())
    y_max: float = float(vertices[:, 1].max())
    x_min: float = float(vertices[:, 0].min())
    x_max: float = float(vertices[:, 0].max())
    y_range: float = y_max - y_min

    # 预计算曲率用于肩胛峰组合评分
    triangles: np.ndarray = np.asarray(mesh.triangles, dtype=np.int64)
    mean_curvature: np.ndarray = compute_mean_curvature(vertices, triangles)

    # -- Step 1: 构建侧向轮廓（lateral profiles） -----------------------------
    left_c: np.ndarray
    right_c: np.ndarray
    left_c, right_c = extract_split_contours(vertices)
    widths: np.ndarray
    y_cen: np.ndarray
    widths, y_cen = compute_width_profile(left_c, right_c)
    lateral_profiles: dict[str, np.ndarray] = {
        "left_contour": left_c,
        "right_contour": right_c,
        "widths": widths,
        "y_centers": y_cen,
    }

    # -- Step 2: 检测双侧 landmark（bilateral landmarks）-----------------------
    # 先检测腰部（最下层的 bilateral landmark），其他依赖腰部结果
    waist: np.ndarray
    waist_debug: dict
    waist, waist_debug = detect_waist(left_c, right_c, widths, y_cen, y_min, y_range)
    waist = lift_or_raise(vertices, waist, "waist")

    # 颈根依赖于腰部点确定搜索范围
    neck_root: np.ndarray
    neck_debug: dict
    neck_root, neck_debug = detect_neck_root_strips(
        waist,
        left_c,
        right_c,
        is_debug=is_debug,
    )
    neck_root = lift_or_raise(vertices, neck_root, "neck_root")

    # 腋窝依赖于颈根和腰部确定搜索带
    axilla: np.ndarray
    axilla_debug: dict
    axilla, axilla_debug = detect_axilla_strips(
        left_c,
        right_c,
        widths,
        y_cen,
        neck_root,
        y_min,
        y_range,
        waist,
    )
    axilla = lift_or_raise(vertices, axilla, "axilla")

    # 肩臂转点依赖于颈根和腋窝确定搜索框
    shoulder_transition: np.ndarray
    st_debug: dict
    shoulder_transition, st_debug = detect_shoulder_transition(
        left_c,
        right_c,
        neck_root,
        axilla,
        axilla_debug,
    )
    shoulder_transition = lift_or_raise(vertices, shoulder_transition, "shoulder_transition")

    # 肩胛峰（scapular peaks）依赖颈根和 Y 范围确定搜索区间
    scapular_peaks: np.ndarray
    scapular_debug: dict
    scapular_peaks, scapular_debug = detect_scapular_peak(
        vertices,
        y_min,
        y_range,
        np.zeros((0, 3)),  # spine_midline 不再预计算，回退到 band 中位数
        neck_root,
        curvature=mean_curvature,
    )
    scapular_peaks = lift_or_raise(vertices, scapular_peaks, "scapular_peaks")

    # -- Step 3: 从双侧点中点推导脊柱点 --------------------------------------
    # shoulder_transition 不参与：附近横截面为 ∩ 形凸起，选点不准
    bilateral_pairs: list[np.ndarray] = [neck_root, scapular_peaks, axilla, waist]
    spine_points: np.ndarray = derive_spine_points(
        vertices,
        bilateral_pairs,
        curvature=mean_curvature,
    )

    # -- Legacy 输出（为下游 pipeline 兼容保留）-------------------------------
    z_mean: float = float(vertices[:, 2].mean())
    shoulder_y: float = y_min + 0.85 * y_range
    pelvic_y: float = y_min + 0.15 * y_range
    shoulder_line: np.ndarray = np.array([[x_min, shoulder_y, z_mean], [x_max, shoulder_y, z_mean]])
    pelvic_line: np.ndarray = np.array([[x_min, pelvic_y, z_mean], [x_max, pelvic_y, z_mean]])

    return {
        "neck_root": neck_root,
        "shoulder_transition": shoulder_transition,
        "scapular_peaks": scapular_peaks,
        "axilla": axilla,
        "waist": waist,
        "spine_points": spine_points,
        "spine_midline": spine_points,  # pipeline 兼容名
        "lateral_profiles": lateral_profiles,
        "waist_debug": waist_debug,
        "neck_debug": neck_debug,
        "axilla_debug": axilla_debug,
        "shoulder_transition_debug": st_debug,
        "scapular_debug": scapular_debug,
        "spine_midline_debug": {},
        "shoulder_line": shoulder_line,
        "pelvic_line": pelvic_line,
    }
