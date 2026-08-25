"""Geodesic boundary computation and mesh cutting for back-surface parameterization."""

from collections import Counter

import numpy as np
from matplotlib.path import Path as MPath
from scipy.ndimage import gaussian_filter1d

from parameterization.template import TEMPLATE_LANDMARKS
from utils.logger import logger

_MIN_PATH_POINTS = 2    # 测地路径至少 2 点（不足则退化为两端点直连）
_MIN_SMOOTH_POINTS = 5  # 路径点数超过该值才做高斯平滑


def geodesic_boundary(
    V: np.ndarray,
    F: np.ndarray,
    k: np.ndarray,
    y: np.ndarray,
    outer_names: list[str],
    smoothing: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """计算连续外部地标顶点之间的测地路径。

    使用 pygeodesic 计算精确测地距离，连接 10 个外部地标形成闭合边界。

    Args:
        V:           (N, 3) 简化网格顶点。
        F:           (M, 3) 三角面。
        k:           (14,) 地标在 V 中的顶点索引。
        y:           (14, 2) 各地标的目标 UV 坐标。
        outer_names:  10 个外部地标名称（顺时针顺序）。
        smoothing:   高斯平滑 sigma，默认 6.0。

    Returns:
        (bverts, Va):
            bverts: (B,) 边界顶点索引（闭合环路）。
            Va:     (N, 2) Procrustes 对齐后的 XY 坐标。
    """
    import pygeodesic.geodesic as geo

    all_names = list(TEMPLATE_LANDMARKS.keys())
    outer_idx = [all_names.index(n) for n in outer_names]
    ov = [int(k[i]) for i in outer_idx]

    geoalg = geo.PyGeodesicAlgorithmExact(V, F)
    bverts: list[int] = []
    for i in range(len(ov)):
        s, d = ov[i], ov[(i + 1) % len(ov)]
        _, path_raw = geoalg.geodesicDistance(s, d)
        path = [V[s].tolist(), V[d].tolist()] if len(path_raw) < _MIN_PATH_POINTS else list(reversed(path_raw))
        pv: list[int] = []
        for pt in path:
            vi = int(np.argmin(np.linalg.norm(V - np.array(pt), axis=1)))
            if not pv or vi != pv[-1]:
                pv.append(vi)
        pv[0] = s
        pv[-1] = d
        pts = V[pv].copy()
        if len(pts) > _MIN_SMOOTH_POINTS:
            pts = gaussian_filter1d(pts, sigma=smoothing, axis=0)
            pts[0] = V[s]
            pts[-1] = V[d]
        seg: list[int] = []
        for pt in pts:
            vi = int(np.argmin(np.linalg.norm(V - pt, axis=1)))
            if not seg or vi != seg[-1]:
                seg.append(vi)
        if i > 0:
            seg = seg[1:]
        bverts.extend(seg)

    # Procrustes alignment for polygon classification
    src, tgt = V[k, :2], y
    sc, tc = src - src.mean(0), tgt - tgt.mean(0)
    ss = np.linalg.norm(tc) / max(np.linalg.norm(sc), 1e-10)
    U, _, Vt = np.linalg.svd(sc.T @ tc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t_off = tgt.mean(0) - ss * src.mean(0) @ R.T
    Va = V.copy()
    Va[:, :2] = ss * V[:, :2] @ R.T + t_off

    return np.array(bverts, dtype=np.int64), Va


def classify_and_cut(
    V: np.ndarray,
    F: np.ndarray,
    k: np.ndarray,
    y: np.ndarray,
    Va: np.ndarray,
    bverts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """2D 多边形内外判断 → 膨胀 → 切割网格。

    先用 matplolib Path 判断顶点是否在边界多边形内，
    再通过一次邻接膨胀包含边界相邻顶点，
    最后提取子网格并重映射地标索引。

    Args:
        V:      (N, 3) 原始网格顶点。
        F:      (M, 3) 三角面。
        k:      (14,) 地标顶点索引。
        y:      (14, 2) 地标 UV 坐标。
        Va:     (N, 2) Procrustes 对齐后的 XY 坐标。
        bverts: (B,) 边界顶点索引。

    Returns:
        (Vc, Fc, kc, yc, o2n, order_map):
            Vc: (K, 3) 切割后子网格顶点。
            Fc: (P, 3) 子网格三角面。
            kc: (L,) 子网格中的地标索引。
            yc: (L, 2) 子网格中的地标 UV。
            o2n: (N,) 原始顶点到子网格顶点的映射（-1 表示被切除）。
            order_map: 旧索引到新索引的映射字典。
    """
    NV = len(V)
    bpath = MPath(Va[bverts, :2], closed=True)
    inside_2d = bpath.contains_points(Va[:, :2])

    al = [[] for _ in range(NV)]
    for a, b, c in F:
        al[a].extend([b, c])
        al[b].extend([a, c])
        al[c].extend([a, b])
    al = [list(set(nb)) for nb in al]

    # 1-iteration dilation to include boundary-adjacent vertices
    new = inside_2d.copy()
    for v in np.where(inside_2d)[0]:
        for nb in al[v]:
            new[nb] = True
    inside_2d = new

    # Keep faces where all 3 vertices are inside
    Fi = F[np.all(inside_2d[F.ravel()].reshape(-1, 3), axis=1)]
    vi = np.unique(Fi)
    o2n = np.full(NV, -1, dtype=np.int64)
    o2n[vi] = np.arange(len(vi))
    Fc = o2n[Fi.ravel()].reshape(-1, 3)
    Vc = V[vi]

    order_map = {old: new for new, old in enumerate(vi)}
    survived = np.array([int(ki) in order_map for ki in k], dtype=bool)
    kc = np.array([order_map[int(ki)] for ki in k[survived]])
    yc = y[survived]

    logger.info(f"Cut mesh: {len(Vc)}v/{NV}v, {len(Fc)}f, {len(kc)} landmarks")
    return Vc, Fc, kc, yc, o2n, order_map


def mesh_rim(Fc: np.ndarray) -> list[int]:
    """提取三角网格的有序边界环（拓扑边界）。

    通过统计每条边出现的次数（边界边只出现一次），
    然后沿边界边追踪形成顺时针排列的顶点序列。

    Args:
        Fc: (P, 3) 切割后网格的三角面。

    Returns:
        顺时针排列的边界顶点索引列表。
    """
    ec = Counter()
    for a, b, c in Fc:
        ec[tuple(sorted((int(a), int(b))))] += 1
        ec[tuple(sorted((int(b), int(c))))] += 1
        ec[tuple(sorted((int(a), int(c))))] += 1
    bedges = [e for e, c in ec.items() if c == 1]
    if not bedges:
        return []
    badj: dict[int, list[int]] = {}
    for a, b in bedges:
        badj.setdefault(a, []).append(b)
        badj.setdefault(b, []).append(a)
    a0, b0 = bedges[0]
    rim = [a0, b0]
    for _ in range(len(bedges) * 2):
        cur = rim[-1]
        if cur not in badj:
            break
        nbs = [n for n in badj[cur] if n != rim[-2]]
        if not nbs:
            break
        rim.append(nbs[0])
        if rim[-1] == a0:
            break
    return rim[:-1]
