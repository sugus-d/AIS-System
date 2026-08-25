"""Signed-distance-field extraction with morphological cleanup."""

import numpy as np
import open3d as o3d

from utils.logger import logger

# ── 裤子振荡带检测阈值 ──
_MIN_EXTRACT_VERTS = 500      # 网格顶点数下限（太少不检测）
_MIN_TRIANGLE_COUNT = 100    # 三角面数下限
_MIN_Y_SPAN_MM = 400         # 网格纵向跨度下限（mm）
_MIN_BAND_VERTICES = 20      # 每 y 分桶最少顶点数
_MIN_BAND_WIDTH_MM = 20      # 分桶 x 跨度下限（mm）
_MIN_CELL_VERTICES = 2       # 每 x 单元格最少顶点数
_MIN_VALID_CELLS = 4         # 有效单元格数下限
_MIN_VALID_SCORE_BINS = 10   # 有效分数桶数下限
_MIN_WINDOW_SCORES = 3       # 窗口内有效分数下限
_MIN_WINDOW_SCORE = 0.0004   # 窗口分数下限（过低视为无振荡带）
_MAX_CUT_FRACTION = 0.50     # 切割位置不得超过网格下半部
# ── 提取器阈值 ──
_MIN_EXTRACT_VERTICES = 50   # BFS 结果顶点数下限（太少直接返回）
_MAX_SDF_DIST_MM = 2         # SDF 距离阈值：质心在此距离内的三角面保留


def _detect_pants_band(mesh: o3d.geometry.TriangleMesh, n_grid_rows: int, y_min: float, y_max: float) -> int | None:
    """Detect pants by curvature oscillation band near mesh bottom.

    Fabric folds create a band where mean|κ| and |dκ/dx| spike together.
    Returns grid row index to cut at, or None.
    """
    import pyvista as pv

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.triangles, dtype=np.int64)
    if len(v) < _MIN_EXTRACT_VERTS or len(f) < _MIN_TRIANGLE_COUNT:
        return None

    y_span = y_max - y_min
    if y_span < _MIN_Y_SPAN_MM:
        return None

    pvm = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]))
    pvm = pvm.smooth_taubin(n_iter=5, pass_band=0.1)
    curv = np.abs(pvm.curvature(curv_type="mean"))

    n_bins = 60
    bins = np.linspace(y_min, y_max, n_bins + 1)
    scores = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (v[:, 1] >= bins[i]) & (v[:, 1] < bins[i + 1])
        if mask.sum() < _MIN_BAND_VERTICES:
            continue
        band_v = v[mask]
        band_c = curv[mask]

        x_min_b, x_max_b = band_v[:, 0].min(), band_v[:, 0].max()
        if x_max_b - x_min_b < _MIN_BAND_WIDTH_MM:
            continue
        nx = min(30, max(10, int((x_max_b - x_min_b) / 5)))
        xi = np.clip(
            ((band_v[:, 0] - x_min_b) / (x_max_b - x_min_b) * (nx - 1)).astype(int),
            0,
            nx - 1,
        )

        cell_k = np.full(nx, np.nan)
        for j in range(nx):
            cm = xi == j
            if cm.sum() >= _MIN_CELL_VERTICES:
                cell_k[j] = np.mean(band_c[cm])

        valid = np.isfinite(cell_k)
        if valid.sum() < _MIN_VALID_CELLS:
            continue

        fill = valid.sum() / nx
        mean_k = float(np.nanmean(cell_k))
        dkdx = float(np.nanmean(np.abs(np.diff(cell_k[valid]))))

        if mean_k > 0 and dkdx > 0:
            scores[i] = mean_k * dkdx * fill

    valid_scores = scores[np.isfinite(scores)]
    if len(valid_scores) < _MIN_VALID_SCORE_BINS:
        return None

    bottom40 = int(n_bins * 0.4)
    skip_bottom = 3
    window = 5
    max_window_score = 0.0
    best_row = -1
    for end_row in range(skip_bottom + window, bottom40 + 1):
        win = scores[end_row - window : end_row]
        win_valid = win[np.isfinite(win)]
        if len(win_valid) < _MIN_WINDOW_SCORES:
            continue
        win_score = float(np.nanmean(win_valid))
        win_max = float(np.nanmax(win_valid))
        if win_max > win_score * 4.0:
            continue
        if win_score > max_window_score:
            max_window_score = win_score
            best_row = end_row

    if max_window_score < _MIN_WINDOW_SCORE or best_row < 0:
        return None

    cut_frac = (bins[best_row] - y_min) / y_span
    if cut_frac > _MAX_CUT_FRACTION:
        return None
    cut_row = max(0, int(cut_frac * n_grid_rows))
    logger.info(f"Pants osc band: row={best_row} score={max_window_score:.6f}")
    return cut_row


def extract_by_xy_hull(
    original: o3d.geometry.TriangleMesh,
    bfs_result: o3d.geometry.TriangleMesh,
    grid_mm: float = 2.0,
    erode_iters: int = 3,
) -> o3d.geometry.TriangleMesh:
    """Extract using signed distance field — triangles near contour included.

    Morphological opening (erode → keep largest → dilate) is applied to the
    XY mask to sever thin protrusions (hair, fabric strips, etc.) before
    the SDF-based triangle selection. *erode_iters* controls the strength
    (0 = disabled).
    """
    from scipy.ndimage import (
        binary_closing,
        binary_dilation,
        binary_erosion,
        binary_fill_holes,
        distance_transform_edt,
        generate_binary_structure,
        label,
    )

    bv = np.asarray(bfs_result.vertices, dtype=np.float64)
    if len(bv) < _MIN_EXTRACT_VERTICES:
        return bfs_result

    x_min, x_max = bv[:, 0].min(), bv[:, 0].max()
    y_min, y_max = bv[:, 1].min(), bv[:, 1].max()
    nx = max(10, int((x_max - x_min) / grid_mm) + 1)
    ny = max(10, int((y_max - y_min) / grid_mm) + 1)

    xi = np.clip(((bv[:, 0] - x_min) / grid_mm).astype(int), 0, nx - 1)
    yi = np.clip(((bv[:, 1] - y_min) / grid_mm).astype(int), 0, ny - 1)

    grid = np.zeros((ny, nx), dtype=bool)
    grid[yi, xi] = True

    pants_row = _detect_pants_band(bfs_result, ny, y_min, y_max)
    if pants_row is not None:
        grid[:pants_row, :] = False

    struct = generate_binary_structure(2, 2)
    grid = binary_fill_holes(grid)
    grid = binary_closing(grid, structure=struct, iterations=3)

    # ── Morphological cleanup: sever thin protrusions ──────────────────
    if erode_iters > 0:
        eroded = binary_erosion(grid, structure=struct, iterations=erode_iters)
        labeled, n_features = label(eroded, structure=struct)
        if n_features > 0:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            grid = labeled == sizes.argmax()
            grid = binary_dilation(grid, structure=struct, iterations=erode_iters)
            logger.info(
                "Morph cleanup: erode=%d, components=%d, kept=%d cells",
                erode_iters,
                n_features,
                int(grid.sum()),
            )

    d_in = distance_transform_edt(grid)
    d_out = distance_transform_edt(~grid)
    sdf = d_out - d_in
    ov = np.asarray(original.vertices, dtype=np.float64)
    ot = np.asarray(original.triangles)
    centroids = ov[ot].mean(axis=1)
    cx = np.clip(((centroids[:, 0] - x_min) / grid_mm).astype(int), 0, nx - 1)
    cy = np.clip(((centroids[:, 1] - y_min) / grid_mm).astype(int), 0, ny - 1)
    keep = sdf[cy, cx] <= _MAX_SDF_DIST_MM

    result = o3d.geometry.TriangleMesh()
    result.vertices = original.vertices
    result.triangles = o3d.utility.Vector3iVector(ot[keep])
    result.remove_unreferenced_vertices()
    result.compute_vertex_normals()

    # ── Post-processing: keep only the largest component ───────────────
    # SDF extraction may re-attach small fragments (hair, fabric) that
    # were within 2 mm of the cleaned mask — discard them here.
    from .bfs import largest_component

    result = largest_component(result)

    n_in = len(np.asarray(bfs_result.vertices))
    n_out = len(np.asarray(result.vertices))
    logger.info(f"Distance-field extract: {n_in}v -> {n_out}v")
    return result
