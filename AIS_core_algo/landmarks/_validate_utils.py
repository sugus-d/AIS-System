"""Landmark GT 校验共用工具。

从 ``gt_validate.py`` 提取以避免循环导入：
各 landmark 的 validate 模块导入本模块的工具函数，
而 ``gt_validate.py`` 同时导入本工具和各 landmark 的 validate 模块。
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d

CACHE_DIR: str = "results/cache"
GT_DIR: str = "results/ground-truth"

_MIN_VERTICAL_COMPONENT = 0.5  # dy 小于该值视为垂直轮廓（转角取 89.9）
_NR_X_HALF_WIDTH_MM = 35.0     # 颈根 X 附近候选区半宽（mm）


def load_mesh(subject: str) -> np.ndarray:
    """加载指定 subject 的网格顶点。

    优先从 extract_roi 缓存读取，回退到 align 缓存。

    Raises:
        FileNotFoundError: 两个缓存目录均找不到对应 mesh。
    """
    for subdir in ("extract_roi", "align"):
        path = str(Path(CACHE_DIR) / subject / subdir / "output.ply")
        if Path(path).exists():
            return np.asarray(o3d.io.read_triangle_mesh(path).vertices)
    raise FileNotFoundError(f"No mesh for {subject}")


def load_curvature_at_point(subject: str, x: float, y: float, vertices: np.ndarray) -> float:
    """加载曲率缓存，取最靠 (x,y) 的顶点曲率值。"""
    curv_path = Path(CACHE_DIR) / subject / "curvature" / "mean_curvature.npy"
    if not curv_path.exists():
        return 0.0
    curv: np.ndarray = np.load(curv_path)
    dists: np.ndarray = np.sum((vertices[:, :2] - np.array([x, y])) ** 2, axis=1)
    return float(curv[np.argmin(dists)])


def load_gt(subject: str) -> dict:
    """加载指定 subject 的 Ground Truth JSON 文件。"""
    path = Path(GT_DIR) / subject / "ground_truth.json"
    return dict(json.loads(path.read_text())) if path.exists() else {}


def bilateral(gt: dict, name: str) -> dict:
    """从扁平 GT 组装双边 dict（``{"L": [x,y,z], "R": [x,y,z]}``）。

    磁盘 GT 为扁平 18 键（``neck_root_L``/``neck_root_R``），validate 模块沿用
    ``{"L","R"}`` 访问；两侧均缺失时返回空 dict（兼容旧 ``gt.get(name, {})`` 语义）。
    """
    left = gt.get(f"{name}_L")
    right = gt.get(f"{name}_R")
    if left is None and right is None:
        return {}
    return {"L": left, "R": right}


def contour_distance(contour: np.ndarray, x: float, y: float) -> float:
    """计算 (x, y) 到轮廓的最短欧氏距离（mm）。"""
    return float(np.sqrt(np.min(np.sum((contour[:, :2] - np.array([x, y])) ** 2, axis=1))))


def long_axis_angle(contour: np.ndarray, x: float, y: float, arc_len: float = 15.0) -> float:
    """长轴转角：沿轮廓向前后各 arc_len mm，计算该段与垂直方向的夹角(°)。

    WHY: 在轮廓上取局部弧段的方向来判断该段是否接近垂直。
    腰部最窄处轮廓近乎垂直（< 10°），颈根在过渡区（20°~70°）。
    """
    dists: np.ndarray = np.sum((contour[:, :2] - np.array([x, y])) ** 2, axis=1)
    idx = int(np.argmin(dists))
    n = len(contour)

    # 沿轮廓向前累计 arc_len 弧长
    cum: float = 0.0
    fwd: int = idx
    while cum < arc_len and fwd < n - 1:
        cum += float(np.sqrt(np.sum((contour[fwd + 1, :2] - contour[fwd, :2]) ** 2)))
        fwd += 1

    # 沿轮廓向后累计 arc_len 弧长
    cum = 0.0
    bwd: int = idx
    while cum < arc_len and bwd > 0:
        cum += float(np.sqrt(np.sum((contour[bwd, :2] - contour[bwd - 1, :2]) ** 2)))
        bwd -= 1

    fwd = min(fwd, n - 1)
    bwd = max(bwd, 0)
    dy: float = contour[fwd, 1] - contour[bwd, 1]
    dx: float = contour[fwd, 0] - contour[bwd, 0]
    if abs(dy) < _MIN_VERTICAL_COMPONENT:
        return 89.9
    return float(np.degrees(np.arctan(abs(dx / dy))))


def z_top_percent(
    vertices: np.ndarray,
    nr_x: float,
    ax_y: float,
    nr_y: float,
    sp_z: float,
) -> float:
    """计算 sp_Z 在颈根-腋窝候选区内的 top 百分比。

    WHY: 肩胛峰在后凸区域中表现为 Z 值最高的前百分之几的点，
    用此指标判断 scapular_peak 是否在合理的后凸区域内。
    值越小说明该点越接近区域 Z 峰值。
    """
    # 在颈根 X 附近 ±35mm 区域、腋窝-颈根 Y 范围内筛选候选顶点
    mask: np.ndarray = np.abs(vertices[:, 0] - nr_x) < _NR_X_HALF_WIDTH_MM
    mask &= (vertices[:, 1] > min(ax_y, nr_y) - 10) & (vertices[:, 1] < max(ax_y, nr_y) + 10)
    region_z: np.ndarray = vertices[mask][:, 2]
    if len(region_z) == 0:
        return 100.0
    return float((region_z >= sp_z).sum() / len(region_z) * 100)


def check(condition: bool, tag: str, detail: str = "") -> dict | None:
    """条件判断辅助：条件为 False 时返回问题字典，否则返回 None。"""
    if not condition:
        return {"tag": tag, "detail": detail}
    return None
