"""编排器：从背部表面网格提取全部双侧 landmark。"""

import numpy as np
import open3d as o3d

from mesh.curvature import compute_mean_curvature
from utils.logger import logger
from utils.mesh import lift_or_raise

from .axilla import detect_axilla_strips
from .complete import complete_landmarks_flat
from .constants import FLAT_SPINE_KEYS
from .lateral_profile import compute_width_profile, extract_split_contours
from .neck_root import detect_neck_root_strips
from .scapular_peak import detect_scapular_peak
from .shoulder_transition import detect_shoulder_transition
from .spine import derive_spine_points
from .waist import detect_waist

_MIN_LANDMARK_VERTS = 50  # 顶点数下限（太少无法提取 landmark）


def extract_landmarks(
    mesh: o3d.geometry.TriangleMesh,
) -> dict:
    """从背部表面网格提取双侧解剖学标志点（landmarks）。

    本函数为 orchestrator：按步骤构建横向剖面、检测各 bilateral landmark，
    再由双侧点推导脊柱点。函数只做坐标处理与流程调度，
    不实现任何复杂的几何运算（遵循可视化/计算分离原则）。

    Args:
        mesh: open3d TriangleMesh 对象，包含顶点与面信息。

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
    waist = detect_waist(left_c, right_c, widths, y_cen, y_min, y_range)
    waist = lift_or_raise(vertices, waist, "waist")

    # 颈根依赖于腰部点确定搜索范围
    neck_root = detect_neck_root_strips(
        waist,
        left_c,
        right_c,
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
    shoulder_transition = detect_shoulder_transition(
        left_c,
        right_c,
        neck_root,
        axilla,
        axilla_debug,
    )
    shoulder_transition = lift_or_raise(vertices, shoulder_transition, "shoulder_transition")

    # 肩胛峰（scapular peaks）依赖颈根和 Y 范围确定搜索区间
    scapular_peaks = detect_scapular_peak(
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

    # -- 扁平 18 键输出（全链路统一契约；debug/几何辅助数据保留）---------------
    # spine_points 与 bilateral_pairs 一一对应（4 点：neck/scapular/axilla/waist），
    # 展开为前 4 个语义 spine 键；thoracic/waist_lower 由 morphology 推导。
    z_mean: float = float(vertices[:, 2].mean())
    shoulder_y: float = y_min + 0.85 * y_range
    pelvic_y: float = y_min + 0.15 * y_range
    shoulder_line: np.ndarray = np.array([[x_min, shoulder_y, z_mean], [x_max, shoulder_y, z_mean]])
    pelvic_line: np.ndarray = np.array([[x_min, pelvic_y, z_mean], [x_max, pelvic_y, z_mean]])

    flat: dict = {
        "neck_root_L": neck_root[0],
        "neck_root_R": neck_root[1],
        "shoulder_transition_L": shoulder_transition[0],
        "shoulder_transition_R": shoulder_transition[1],
        "scapular_peaks_L": scapular_peaks[0],
        "scapular_peaks_R": scapular_peaks[1],
        "axilla_L": axilla[0],
        "axilla_R": axilla[1],
        "waist_L": waist[0],
        "waist_R": waist[1],
    }
    # spine 语义键（前 4 个基础点，顺序 = bilateral_pairs 顺序，单源 FLAT_SPINE_KEYS）
    spine_semantic = FLAT_SPINE_KEYS[:4]
    for idx, key in enumerate(spine_semantic):
        if idx < len(spine_points):
            flat[key] = spine_points[idx]

    flat.update(
        {
            "lateral_profiles": lateral_profiles,
            "shoulder_line": shoulder_line,
            "pelvic_line": pelvic_line,
        }
    )
    # 统一为完整 18 键契约：算法检测缺 waist_lower / thoracic / 部分 spine 点，
    # 用训练集平均模板拟合相似变换补全（复用 complete_landmarks_flat，mesh 最近顶点映射）
    return complete_landmarks_flat(flat, mesh)
