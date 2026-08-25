# 可复用/简单函数候选 (自动扫描)
生成时间: 2026-05-11T15:11:04.600872

> **⚠️ 部分过期** — 本文由静态分析自动生成，引用的单文件路径（`landmarks/spine.py`、
> `landmarks/neck_histogram.py` 等）已于 2026-07 拆分为子包（`landmarks/*/core.py`）。
> 仍有效的条目：`landmarks/extract.py`、`landmarks/lateral_profile.py`。
> 审阅时请以当前目录结构为准。
>
> **2026-08-16 更新**：landmark 调试数据已全面清理（neck/waist/axilla 等 `*_debug`
> 数据与 `build_side_debug`/`build_neck_root_debug`/`build_scapular_peak_debug`/
> `build_spine_debug`/`_empty_debug` 等构建函数已删除），本文中相关条目均已失效。

说明: 本文档由静态分析生成，采用启发式规则筛选“较为简单/仅依赖 numpy/标准库”的函数作为候选。请人工审核。

审阅指引：
- 已知业务相关（请排除）：[landmarks/spine.py](landmarks/spine.py) 中的 `derive_spine_points` 属于与脊柱/中线拟合等业务逻辑密切相关，不建议提取为通用工具。
- 延后复核（暂不处理）：[landmarks/angle_utils.py](landmarks/angle_utils.py) 中的若干私有几何/角度函数（如 `_seg_intersect`、`_interpolate_point_at_distance` 等），这些函数实现虽简洁但紧耦合于角度/侧向计算，先不纳入自动提取范围。

## 业务相关 - 排除列表（自动识别建议）

生成时间: 2026-05-11T15:38:01.170100

说明：下面函数被自动识别为与医学/landmark/网格处理流程强耦合，建议在提取为通用工具时排除。最终以人工审核为准。

- `landmarks/neck_histogram.py` :: `compute_width_profile` (lines 12-68)
    - 代码预览: def compute_width_profile(
- `landmarks/neck_histogram.py` :: `compute_histogram_mode_width` (lines 71-147)
    - 代码预览: def compute_histogram_mode_width(
- `landmarks/neck_histogram.py` :: `collect_candidates` (lines 150-188)
    - 代码预览: def collect_candidates(
- `landmarks/neck_histogram.py` :: `gradient_filter_candidates` (lines 191-219)
    - 代码预览: def gradient_filter_candidates(
- `landmarks/scapular_peak.py` :: `detect_scapular_peak` (lines 6-106)
    - 代码预览: def detect_scapular_peak(
- `landmarks/scapular_peak.py` :: `_detect_one_side` (lines 109-186)
    - 代码预览: def _detect_one_side(
- `landmarks/spine.py` :: `fit_spine_midline` (lines 6-128)
    - 代码预览: def fit_spine_midline(
- `landmarks/spine.py` :: `derive_spine_points` (lines 131-171)
    - 代码预览: def derive_spine_points(
- `landmarks/spine.py` :: `_build_debug_dict` (lines 174-200)
    - 代码预览: def _build_debug_dict(
- `landmarks/waist.py` :: `detect_waist` (lines 6-55)
    - 代码预览: def detect_waist(
- `landmarks/lateral_profile.py` :: `extract_split_contours` (lines 16-28)
    - 代码预览: def extract_split_contours(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
- `landmarks/lateral_profile.py` :: `compute_width_profile` (lines 31-66)
    - 代码预览: def compute_width_profile(
- `landmarks/lateral_profile.py` :: `_extract_body_contour` (lines 69-151)
    - 代码预览: def _extract_body_contour(
- `landmarks/lateral_profile.py` :: `_split_contours` (lines 154-213)
    - 代码预览: def _split_contours(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
- `landmarks/angle_utils.py` :: `compute_lateral_angle_at_point` (lines 13-53)
    - 代码预览: def compute_lateral_angle_at_point(
- `landmarks/angle_utils.py` :: `compute_lateral_angle_profile` (lines 56-95)
    - 代码预览: def compute_lateral_angle_profile(
- `landmarks/angle_utils.py` :: `_interpolate_point_at_distance` (lines 121-131)
    - 代码预览: def _interpolate_point_at_distance(p1, p2, neck_pt, distance):
- `landmarks/angle_utils.py` :: `_select_side_point` (lines 134-197)
    - 代码预览: def _select_side_point(
- `landmarks/angle_utils.py` :: `_find_contour_neighbors` (lines 213-238)
    - 代码预览: def _find_contour_neighbors(
- `landmarks/extract.py` :: `extract_landmarks` (lines 15-148)
    - 代码预览: def extract_landmarks(mesh: o3d.geometry.TriangleMesh) -> dict:
- `landmarks/axilla.py` :: `detect_axilla_strips` (lines 21-420)
    - 代码预览: def detect_axilla_strips(
- `landmarks/axilla.py` :: `_has_arms` (lines 423-482)
    - 代码预览: def _has_arms(
- `landmarks/axilla.py` :: `_find_arm_boundary_x` (lines 485-521)
    - 代码预览: def _find_arm_boundary_x(sorted_pts, dydx, d2ydx2, side_name):
- `landmarks/shoulder_transition.py` :: `detect_shoulder_transition` (lines 22-162)
    - 代码预览: def detect_shoulder_transition(
- `landmarks/shoulder_transition.py` :: `_empty_debug` (lines 165-180)
    - 代码预览: def _empty_debug() -> dict:
- `landmarks/shoulder_transition.py` :: `_to_2d` (lines 40-46)
    - 代码预览: def _to_2d(contour: np.ndarray) -> np.ndarray:
- `landmarks/neck_root.py` :: `_filter_contour` (lines 28-64)
    - 代码预览: def _filter_contour(
- `landmarks/neck_root.py` :: `detect_neck_root_strips` (lines 66-347)
    - 代码预览: def detect_neck_root_strips(
- `landmarks/neck_root.py` :: `_build_search_segment` (lines 353-370)
    - 代码预览: def _build_search_segment(
- `landmarks/neck_root.py` :: `_extract_longest_contiguous` (lines 373-388)
    - 代码预览: def _extract_longest_contiguous(idxs: list[int]) -> list[int]:
- `landmarks/neck_root.py` :: `_smoothed_segment_xy` (lines 391-400)
    - 代码预览: def _smoothed_segment_xy(contour: np.ndarray, idxs: list[int]) -> np.ndarray:
- `landmarks/neck_root.py` :: `_compute_segment_derivatives` (lines 403-407)
    - 代码预览: def _compute_segment_derivatives(smoothed_xy: np.ndarray) -> np.ndarray:
- `landmarks/neck_root.py` :: `_filter_by_derivative` (lines 410-430)
    - 代码预览: def _filter_by_derivative(
- `landmarks/neck_root.py` :: `_compute_angle_candidates_from_idxs` (lines 433-444)
    - 代码预览: def _compute_angle_candidates_from_idxs(
- `landmarks/neck_root.py` :: `_store_simple_entry` (lines 447-490)
    - 代码预览: def _store_simple_entry(
- `landmarks/neck_root.py` :: `_process_candidates` (lines 496-547)
    - 代码预览: def _process_candidates(
- `landmarks/neck_root.py` :: `_store_entry` (lines 558-587)
    - 代码预览: def _store_entry(
- `landmarks/neck_root.py` :: `_fallback_candidate` (lines 590-611)
    - 代码预览: def _fallback_candidate(candidate_indices, ys, contour, vertices):
- `landmarks/neck_root.py` :: `_append_width_check` (lines 614-623)
    - 代码预览: def _append_width_check(neck_root, W_mode, verification_log, angle_debug):
- `landmarks/neck_root.py` :: `pick_best` (lines 206-207)
    - 代码预览: def pick_best(cands):
- `mesh/alignment.py` :: `align_mesh` (lines 25-98)
    - 代码预览: def align_mesh(
- `mesh/alignment.py` :: `apply_rotation` (lines 101-118)
    - 代码预览: def apply_rotation(mesh: o3d.geometry.TriangleMesh, x=0, y=0, z=0, in_degrees=True):
- `mesh/alignment.py` :: `calculate_distance_from_plane` (lines 121-133)
    - 代码预览: def calculate_distance_from_plane(

## 总结
- 扫描目录: ['landmarks', 'mesh']
- 候选函数数量 (高概率可提取): 23
- 可能候选 (需要人工判断): 5
- 解析错误/文件数: 0

## 候选函数（高概率）

### landmarks/spine.py :: derive_spine_points (lines 131-171, len=41)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def derive_spine_points(
    vertices: np.ndarray,
    bilateral_pairs: list[np.ndarray],
    search_radius_frac: float = 0.08,
) -> np.ndarray:
    """对每个双侧中点，在改进后的中线上按 Y 采样得到脊柱点。

    WHY：在平滑中线上直接采样 Y 对应位置，避免独立 Z-min 搜索的不连续性。

    Args:
        vertices: 网格顶点数组 (N, 3)。
        bilateral_pairs: 若干 (2,3) 的左右成对点列表。
        search_radius_frac: 已弃用，保留参数兼容调用方。

    Returns:
        (K, 3) 的脊柱点数组，与 bilateral_pairs 一一对应。
    """
    _ = search_radius_frac  # deprecated, kept for signature compatibility

    # 计算 mid_x（与 extract.py 逻辑一致）
    y = vertices[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    y_range = y_max - y_min
    lower_half = vertices[y < y_min + 0.50 * y_range]
    if len(lower_half) > 10:
        mid_x_val = float((lower_half[:, 0].min() + lower_half[:, 0].max()) / 2.0)
    else:
        mid_x_val = float((vertices[:, 0].min() + vertices[:, 0].max()) / 2.0)

    midline, _ = fit_spine_midline(vertices, mid_x_val)

    if len(midline) < 2:
        return np.array([(pair[0] + pair[1]) / 2.0 for pair in bilateral_pairs])

    spine_pts = []
    for pair in bilateral_pairs:
        y_target = float((pair[0, 1] + pair[1, 1]) / 2.0)
        y_target = float(np.clip(y_target, midline[:, 1].min(), midline[:, 1].max()))
        idx = int(np.argmin(np.abs(midline[:, 1] - y_target)))
        spine_pts.append(midline[idx])
    return np.array(spine_pts)
```

### landmarks/spine.py :: _build_debug_dict (lines 174-200, len=27)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _build_debug_dict(
    pts: np.ndarray,
    mid_x: float,
    n_bins_total: int,
    rejected_mad: np.ndarray | None = None,
    rejected_residual: np.ndarray | None = None,
    pts_clean: np.ndarray | None = None,
) -> dict:
    """构建调试字典，兼容候选点不足和完整过滤两条路径。"""
    n_raw = len(pts)
    if rejected_mad is None:
        rejected_mad = np.zeros(n_raw, dtype=bool) if n_raw else np.zeros(0, dtype=bool)
    if rejected_residual is None:
        rejected_residual = (
            np.zeros(n_raw, dtype=bool) if n_raw else np.zeros(0, dtype=bool)
        )
    if pts_clean is None:
        pts_clean = np.zeros((0, 3))
    clean_mask = ~(rejected_mad | rejected_residual)
    n_clean = int(clean_mask.sum())
    x_range_raw = float(pts[:, 0].max() - pts[:, 0].min()) if n_raw > 0 else 0.0
    x_range_clean = (
        float(pts_clean[:, 0].max() - pts_clean[:, 0].min()) if n_clean > 0 else 0.0
    )
    return {
        "bin_candidates": pts,
        "rejected_mad": rejected_mad,
        "rejected_residual": rejected_residual,
        "candidates_clean": pts_clean,
        "mid_x": mid_x,
        "n_bins_total": n_bins_total,
        "n_candidates_raw": n_raw,
        "n_candidates_clean": n_clean,
        "x_range_raw": x_range_raw,
        "x_range_clean": x_range_clean,
    }
```

### landmarks/lateral_profile.py :: extract_split_contours (lines 16-28, len=13)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def extract_split_contours(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从网格顶点提取实际边界轮廓并分割为左右两侧。

    Args:
        vertices: 网格顶点数组 (N, 3)。

    Returns:
        tuple: (left_contour, right_contour)，均为 (M, 2) 数组，保持原始 CW 顺序。
    """
    contour = _extract_body_contour(vertices)
    left_contour, right_contour = _split_contours(contour)

    return left_contour, right_contour
```

### landmarks/lateral_profile.py :: _split_contours (lines 154-213, len=60)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _split_contours(contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从完整轮廓中分割为左右两侧轮廓线。

    Args:
        contour: 完整轮廓点数组，形状 (N, 2)，包含 XY 坐标，CW 排列且从最高点开始。

    Returns:
        tuple: (left_contour, right_contour)，均为 (M, 2) 数组，保持原始 CW 顺序。
    """
    if contour is None or contour.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    pts = np.asarray(contour, dtype=np.float64)
    n = pts.shape[0]
    if n < 3:
        split_x = float(np.median(pts[:, 0]))
        mask = pts[:, 0] < split_x
        return pts[mask], pts[~mask]

    # PCA 对齐主轴，在旋转坐标系中按 X 中位数切分，适应歪斜 body
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    principal = vt[0]
    if principal[1] < 0 or (principal[1] == 0 and principal[0] < 0):
        principal = -principal
    phi = float(np.arctan2(principal[1], principal[0]))
    rot = np.pi / 2.0 - phi
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    rotated = centered.dot(R.T)

    split_x = float(np.median(rotated[:, 0]))
    is_left = rotated[:, 0] < split_x

    trans = np.where(is_left[1:] != is_left[:-1])[0] + 1
    if trans.size < 2:
        fx = float(np.median(pts[:, 0]))
        mask = pts[:, 0] < fx
        return pts[mask], pts[~mask]

    runs = []
    prev = 0
    for t in trans:
        runs.append((prev, t - prev, bool(is_left[prev])))
        prev = t
    runs.append((prev, n - prev, bool(is_left[prev])))

    left_run = max((r for r in runs if r[2]), key=lambda r: r[1], default=None)
    right_run = max((r for r in runs if not r[2]), key=lambda r: r[1], default=None)

    if left_run is None or right_run is None:
        fx = float(np.median(pts[:, 0]))
        mask = pts[:, 0] < fx
        return pts[mask], pts[~mask]

    # 把左侧段滚到 index 0，然后在右侧段起点处切开
    n_pts = pts.shape[0]
    pts = np.roll(pts, -left_run[0], axis=0)
    cut = int((right_run[0] - left_run[0]) % n_pts)
    return pts[:cut], pts[cut:]
```

### landmarks/angle_utils.py :: _seg_intersect (lines 98-118, len=21)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _seg_intersect(A, B, C, r):
    """求解线段 AB 与以 C 为中心、r 为半径的圆的参数 t（0<=t<=1）。

    返回值为 t（若存在交点）或 None。此为私有函数，仅用于在线段上精确定位与圆的交点。
    """
    # 线段参数化：A + t*(B-A)，解一元二次方程
    D = B - A
    OC = A - C
    a = float(np.dot(D, D))
    if a < 1e-12:
        return None
    b = 2.0 * float(np.dot(D, OC))
    c = float(np.dot(OC, OC)) - r * r
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    for t in [(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)]:
        if 0.0 <= t <= 1.0:
            return t
    return None
```

### landmarks/angle_utils.py :: _interpolate_point_at_distance (lines 121-131, len=11)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _interpolate_point_at_distance(p1, p2, neck_pt, distance):
    """在线段 p1-p2 上内插出距离 neck_pt 恰为 distance 的点（若存在）。"""
    # 在一段轮廓边上找到距离参考点正好等于 distance 的位置，用于精确获取侧点坐标
    p1_xy = np.asarray(p1, dtype=np.float64)[:2]
    p2_xy = np.asarray(p2, dtype=np.float64)[:2]
    t = _seg_intersect(
        p1_xy, p2_xy, np.asarray(neck_pt, dtype=np.float64)[:2], distance
    )
    if t is None:
        return None
    return p1_xy + t * (p2_xy - p1_xy)
```

### landmarks/angle_utils.py :: _is_contour_ccw (lines 200-210, len=11)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _is_contour_ccw(contour_xy: np.ndarray) -> bool:
    """判断轮廓点列是否为逆时针方向（Y-up 坐标系）。

    用鞋带公式计算 signed area。area > 0 为逆时针（CCW），area < 0 为顺时针（CW）。
    对于 open curve 不闭合最后一段，相邻叉积累积已能反映弧线走向。
    """
    xy = contour_xy[:, :2]
    area = 0.0
    for i in range(len(xy) - 1):
        area += xy[i, 0] * xy[i + 1, 1] - xy[i + 1, 0] * xy[i, 1]
    return area > 0.0
```

### landmarks/angle_utils.py :: _find_contour_neighbors (lines 213-238, len=26)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _find_contour_neighbors(
    contour: np.ndarray,
    pt: np.ndarray,
    distance: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """沿轮廓取候选点的前后邻点（利用 CW 顺序：index-1 = before, index+1 = after）。

    轮廓已保证为顺时针顺序，前后邻点直接在数组上取相邻 index 即可。
    `distance` 参数保留仅用于兼容调用侧，不再参与搜索。

    Returns:
        (pt_before, pt_after)：轮廓遍历方向的前后点。
    """
    contour_xy = np.asarray(contour, dtype=np.float64)
    n = len(contour_xy)
    if n < 3:
        return contour_xy[0, :2].copy() if n else np.zeros(2), contour_xy[
            -1, :2
        ].copy() if n else np.zeros(2)

    neck_xy = np.asarray(pt, dtype=np.float64)
    dists = np.linalg.norm(contour_xy[:, :2] - neck_xy[:2], axis=1)
    idx = int(np.argmin(dists))

    pt_before = contour_xy[(idx - 1) % n, :2].copy()
    pt_after = contour_xy[(idx + 1) % n, :2].copy()
    return pt_before, pt_after
```

### landmarks/extract.py :: _lift_2d_to_vertex (lines 59-78, len=20)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
    def _lift_2d_to_vertex(vertices: np.ndarray, pts2d: np.ndarray) -> np.ndarray:
        """将 (N,2) 的 xy 点映射到最接近的 vertices 顶点（按 xy 最近邻），返回 (N,3)。

        如果传入 pts2d 已经是 3D（N,3），则原样返回。
        """
        if pts2d is None:
            return None
        pts = np.asarray(pts2d)
        if pts.ndim != 2:
            raise ValueError("pts2d must be (N,2) or (N,3)")
        if pts.shape[1] == 3:
            return pts.copy()
        # KD-tree for nearest neighbor on xy
        verts_xy = vertices[:, :2]
        out = np.zeros((pts.shape[0], 3), dtype=np.float64)
        for i, p in enumerate(pts):
            dists = np.sum((verts_xy - p[:2]) ** 2, axis=1)
            idx = int(np.argmin(dists))
            out[i] = vertices[idx]
        return out
```

### landmarks/axilla.py :: _has_arms (lines 423-482, len=60)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _has_arms(
    left_c: np.ndarray,
    right_c: np.ndarray,
    widths: np.ndarray,
    y_cen: np.ndarray,
    y_min: float,
    y_range: float,
) -> tuple[bool, bool]:
    """Detect if arms are present on each side.

    Returns (has_left_arm, has_right_arm).
    """
    # WHY: 手臂存在时肩部轮廓比腰部更向外侧延伸。
    # 通过肩部最大 X 偏移与腰部最大 X 偏移的比值（>1.2）判断手臂是否存在。
    # 阈值 1.2 是经验值：太小会把无臂 case 误判为有臂，太大会漏掉有臂 case。
    frac = (y_cen - y_min) / y_range
    waist_mask = (frac >= 0.40) & (frac <= 0.60)
    shoulder_mask = (frac >= 0.60) & (frac <= 0.90)

    waist_w = float(np.median(widths[waist_mask])) if waist_mask.sum() > 0 else 1.0

    has_left = has_right = False
    if shoulder_mask.sum() > 3 and waist_w > 1:
        # NOTE: y_cen 长度固定（例如 150），但 left_c/right_c 可能为不同长度的轮廓。
        # 不可直接用 mask 索引轮廓，会导致 IndexError。按 y_cen 的 mask 取出对应的 Y 范围，
        # 然后用该 Y 范围过滤实际轮廓点。
        waist_y_lo = y_cen[waist_mask].min() if waist_mask.sum() > 0 else y_min
        waist_y_hi = y_cen[waist_mask].max() if waist_mask.sum() > 0 else y_min
        shoulder_y_lo = y_cen[shoulder_mask].min() if shoulder_mask.sum() > 0 else y_min
        shoulder_y_hi = y_cen[shoulder_mask].max() if shoulder_mask.sum() > 0 else y_min

        sh_left = left_c[
            (left_c[:, 1] >= shoulder_y_lo) & (left_c[:, 1] <= shoulder_y_hi)
        ]
        sh_right = right_c[
            (right_c[:, 1] >= shoulder_y_lo) & (right_c[:, 1] <= shoulder_y_hi)
        ]
        left_extent = float(np.abs(sh_left[:, 0].min())) if len(sh_left) > 0 else 0.0
        right_extent = float(np.abs(sh_right[:, 0].max())) if len(sh_right) > 0 else 0.0

        waist_left_pts = left_c[
            (left_c[:, 1] >= waist_y_lo) & (left_c[:, 1] <= waist_y_hi)
        ]
        waist_right_pts = right_c[
            (right_c[:, 1] >= waist_y_lo) & (right_c[:, 1] <= waist_y_hi)
        ]
        waist_left = (
            float(np.abs(waist_left_pts[:, 0].min()))
            if len(waist_left_pts) > 0
            else 1.0
        )
        waist_right = (
            float(np.abs(waist_right_pts[:, 0].max()))
            if len(waist_right_pts) > 0
            else 1.0
        )

        has_left = (left_extent / max(waist_left, 1.0)) > 1.2
        has_right = (right_extent / max(waist_right, 1.0)) > 1.2
    return has_left, has_right
```

### landmarks/axilla.py :: _find_arm_boundary_x (lines 485-521, len=37)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _find_arm_boundary_x(sorted_pts, dydx, d2ydx2, side_name):
    """Walk inner to outer, find 20 mm stretch where |dy/dx|<0.3 and |d²|<0.01.

    Returns the X position at the start of the stretch, or None.
    """
    # WHY: 手臂区域的轮廓特征是平坦的（|dy/dx| 小、|d²| 小），
    # 从躯干内侧向外侧搜索，找到第一个连续 20mm 满足平坦条件的区段，
    # 其起点即为手臂边界，用于把搜索范围限制在躯干上而不搜到手臂区域。
    if side_name == "left":
        dydx = dydx[::-1]
        d2ydx2 = d2ydx2[::-1]
        xs = sorted_pts[::-1, 0]
    else:
        xs = sorted_pts[:, 0]

    window_mm = 20
    n = len(xs)

    for i in range(n):
        if side_name == "left":
            j = i
            while j < n and xs[i] - xs[j] < window_mm:
                j += 1
        else:
            j = i
            while j < n and xs[j] - xs[i] < window_mm:
                j += 1

        if j >= n:
            break

        # WHY: |dy/dx| < 0.3 表示该段斜率平缓，|d²| < 0.01 表示曲率很小。
        # 连续 20mm 同时满足这两个条件说明已经进入手臂区域。
        if np.all(np.abs(dydx[i:j]) < 0.3) and np.all(np.abs(d2ydx2[i:j]) < 0.01):
            return float(xs[i])

    return None
```

### landmarks/shoulder_transition.py :: _empty_debug (lines 165-180, len=16)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _empty_debug() -> dict:
    return {
        "contour": np.empty((0, 2)),
        "long_axis_angles": np.empty(0),
        "candidate_mask": np.empty(0, dtype=bool),
        "box_mask": np.empty(0, dtype=bool),
        "peak_index": 0,
        "peak_point": np.zeros(2),
        "peak_angle_deg": 180.0,
        "has_arm": True,
        "fallback": True,
        "outer_x": 0.0,
        "inner_x": 0.0,
        "axY": 0.0,
        "n_candidates": 0,
    }
```

### landmarks/shoulder_transition.py :: _to_2d (lines 40-46, len=7)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
    def _to_2d(contour: np.ndarray) -> np.ndarray:
        arr = np.asarray(contour)
        if arr.ndim != 2:
            raise ValueError("contour must be 2D array")
        if arr.shape[1] == 2:
            return arr.astype(np.float64)
        return arr[:, :2].astype(np.float64)
```

### landmarks/neck_root.py :: _filter_contour (lines 28-64, len=37)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _filter_contour(
    left_c: np.ndarray,
    right_c: np.ndarray,
    waist_points: np.ndarray,
    sigma: float = CONTOUR_SIGMA,
):
    """对左右轮廓的 XY 坐标应用高斯平滑，减少后续导数计算的噪声。

    Args:
        left_c: 左侧轮廓点数组，形状 (N, 3)。
        right_c: 右侧轮廓点数组，形状 (N, 3)。
        waist_points: 腰部左右点数组，形状 (2, 3)，用于日志记录但不直接影响平滑。
        sigma: 高斯核的标准差，控制平滑程度。默认值为 0.5。
    Returns:
        tuple: (left_c_smoothed, right_c_smoothed)，与输入形状相同的平滑后轮廓数组。
    """

    y_min = min(left_c[:, 1].min(), right_c[:, 1].min())
    y_max = max(left_c[:, 1].max(), right_c[:, 1].max())
    y_range = y_max - y_min
    waist_x_l, waist_x_r = waist_points[:, 0]

    lo = y_max - TOP_RATIO * y_range
    left_mask = (left_c[:, 1] >= lo) & (left_c[:, 0] >= waist_x_l)
    right_mask = (right_c[:, 1] >= lo) & (right_c[:, 0] <= waist_x_r)
    if left_mask.sum() <= 0 or right_mask.sum() <= 0:
        raise ValueError(
            "No contour points in top region after masking: check waist points and contour data"
        )

    left_c = np.stack(
        [
            gaussian_filter1d(left_c[left_mask, 0], sigma=sigma, mode="nearest"),
            gaussian_filter1d(left_c[left_mask, 1], sigma=sigma, mode="nearest"),
        ],
        axis=1,
    )

    right_c = np.stack(
        [
            gaussian_filter1d(right_c[right_mask, 0], sigma=sigma, mode="nearest"),
            gaussian_filter1d(right_c[right_mask, 1], sigma=sigma, mode="nearest"),
        ],
        axis=1,
    )

    return left_c, right_c
```

### landmarks/neck_root.py :: _build_search_segment (lines 353-370, len=18)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _build_search_segment(
    contour: np.ndarray,
    before_pt: np.ndarray,
    after_pt: np.ndarray,
    is_left: bool,  # 当前未使用，左右两侧搜索逻辑相同
) -> np.ndarray:
    if len(contour) == 0:
        return np.array([], dtype=int)

    before_dists = np.linalg.norm(contour[:, :2] - before_pt[:2], axis=1)
    before_idx = int(np.argmin(before_dists))
    print(before_idx, before_pt, contour[before_idx])

    after_dists = np.linalg.norm(contour[:, :2] - after_pt[:2], axis=1)
    after_idx = int(np.argmin(after_dists))
    print(after_idx, after_pt, contour[after_idx])

    return contour[before_idx : after_idx + 1]
```

### landmarks/neck_root.py :: _extract_longest_contiguous (lines 373-388, len=16)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _extract_longest_contiguous(idxs: list[int]) -> list[int]:
    """Given a list of contour indices (in order discovered), return the longest contiguous
    sub-run (by index adjacency)."""
    if not idxs:
        return []
    runs = []
    cur = [idxs[0]]
    for a, b in zip(idxs, idxs[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    longest = max(runs, key=lambda r: len(r))
    return longest
```

### landmarks/neck_root.py :: _smoothed_segment_xy (lines 391-400, len=10)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _smoothed_segment_xy(contour: np.ndarray, idxs: list[int]) -> np.ndarray:
    if not idxs:
        return np.empty((0, 2), dtype=float)
    pts = np.array([contour[i].astype(float) for i in idxs])
    if len(pts) < 3:
        return pts
    sigma = max(1.0, len(pts) // 15)
    xs = gaussian_filter1d(pts[:, 0], sigma=sigma, mode="nearest")
    ys = gaussian_filter1d(pts[:, 1], sigma=sigma, mode="nearest")
    return np.column_stack([xs, ys])
```

### landmarks/neck_root.py :: _compute_segment_derivatives (lines 403-407, len=5)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _compute_segment_derivatives(smoothed_xy: np.ndarray) -> np.ndarray:
    """Compute derivative of X w.r.t. contour index. Returns zeros for tiny segments."""
    if len(smoothed_xy) < 3:
        return np.zeros(len(smoothed_xy))
    return np.gradient(smoothed_xy[:, 0])
```

### landmarks/neck_root.py :: _filter_by_derivative (lines 410-430, len=21)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _filter_by_derivative(
    idxs: list[int], deriv: np.ndarray, is_left: bool
) -> list[int]:
    """Filter segment indices by derivative thresholds: left d > +1, right d < -1.
    deriv corresponds to smoothed segment order; map back to idxs accordingly."""
    if not idxs or len(deriv) == 0:
        return []
    kept = []
    for i, d in enumerate(deriv):
        if is_left and d > 1.0:
            kept.append(idxs[i])
        if (not is_left) and d < -1.0:
            kept.append(idxs[i])
    # Return unique preserved in original order
    seen = set()
    out = []
    for v in kept:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out
```

### landmarks/neck_root.py :: _store_entry (lines 558-587, len=30)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _store_entry(
    entry,
    side_name,
    idx_out,
    neck_root,
    angle_debug,
    verification_log,
    valid_left,
    valid_right,
):
    best_pt, best_angle_deg, left_pt, right_pt, left_dist, right_dist = entry
    angle_ok = best_angle_deg > 90
    angle_debug[side_name] = {
        "left": [float(left_pt[0]), float(left_pt[1])],
        "right": [float(right_pt[0]), float(right_pt[1])],
        "curr": [float(best_pt[0]), float(best_pt[1])],
        "left_dist": left_dist,
        "right_dist": right_dist,
        "angle_deg": best_angle_deg,
        "valid": angle_ok,
    }
    verification_log.append({
        "side": side_name,
        "left_dist": round(left_dist, 1),
        "right_dist": round(right_dist, 1),
        "cwa": round(best_angle_deg, 1),
        "valid_count": len(valid_left if side_name == "left" else valid_right),
        "angle_ok": angle_ok,
    })
    neck_root[idx_out] = best_pt
```

### landmarks/neck_root.py :: _append_width_check (lines 614-623, len=10)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _append_width_check(neck_root, W_mode, verification_log, angle_debug):
    neck_w = float(np.linalg.norm(neck_root[1] - neck_root[0]))
    W_mode_check = W_mode if W_mode and W_mode > 0 else 1
    ratio_check = neck_w / W_mode_check
    for v in verification_log:
        v["neck_width_ratio"] = round(ratio_check, 2)
        v["neck_width_ok"] = ratio_check < 1.4
    angle_debug["neck_width"] = neck_w
    angle_debug["neck_width_ratio"] = round(ratio_check, 2)
    angle_debug["neck_width_ok"] = ratio_check < 1.4
```

### landmarks/neck_root.py :: pick_best (lines 206-207, len=2)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
    def pick_best(cands):
        return min(cands, key=lambda x: x[1]) if cands else None
```

### mesh/clean.py :: _estimate_radius (lines 71-76, len=6)

理由: 小函数，未发现项目级调用，主要使用 numpy/基础运算。

```python
def _estimate_radius(vertices: np.ndarray, nb_neighbors: int) -> float:
    from scipy.spatial import KDTree

    tree = KDTree(vertices)
    dists, _ = tree.query(vertices, k=min(nb_neighbors + 1, len(vertices)))
    return float(np.median(dists[:, -1]) * 2.0)
```

## 可能候选（需人工复核）

### landmarks/neck_histogram.py :: gradient_filter_candidates (lines 191-219, len=29)

理由: 中等长度，未发现明显业务调用，但仍需人工确认。

```python
def gradient_filter_candidates(
    candidate_indices: list[int], ws_smooth: np.ndarray, ys: np.ndarray
) -> list[int]:
    """基于宽度梯度过滤肩部边缘处的候选行。

    说明：肩部边缘通常伴随宽度突变（梯度高），本函数用候选行处的宽度梯度的低位百分位
    作为阈值，去除梯度过高的候选，从而避免将肩部边缘误判为颈根候选。

    Args:
        candidate_indices: 初始候选行索引列表。
        ws_smooth: 平滑后的宽度曲线数组。
        ys: 每行对应的 Y 值数组（用于梯度计算的自变量）。

    Returns:
        list[int]: 过滤后的候选索引列表（若过滤结果为空则返回原列表以防止空集合）。
    """
    if len(candidate_indices) <= 3:
        return candidate_indices

    ws_grad = np.gradient(ws_smooth, ys)
    cand_grads = np.array([ws_grad[ci] for ci in candidate_indices])
    threshold = float(np.percentile(cand_grads, 10))
    kept = [ci for ci, g in zip(candidate_indices, cand_grads) if g >= threshold]
    if kept and len(kept) < len(candidate_indices):
        logger.info(
            f"[GRAD] removed {len(candidate_indices) - len(kept)}/{len(candidate_indices)} "
            f"high-gradient candidates (threshold={threshold:.3f})"
        )
    return kept if kept else candidate_indices
```

### landmarks/scapular_peak.py :: detect_scapular_peak (lines 6-106, len=101)

理由: 中等长度，未发现明显业务调用，但仍需人工确认。

```python
def detect_scapular_peak(
    vertices: np.ndarray,
    y_min: float,
    y_range: float,
    spine_midline: np.ndarray,
    neck_root: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """检测每侧肩胛峰（scapular peak）：在肩胛带选择最向后且较高的点。

    WHY：肩胛峰位于肩胛骨上角，在背部点云中表现为后凸且相对靠上的区域。
    用脊柱中线（而非下半身中点）分侧，结合 Z 值分位过滤和对称性校验，
    在含噪声的点云中稳健定位左右肩胛峰。

    Args:
        vertices: 网格顶点数组 (N, 3)。
        y_min: 顶点 Y 最小值。
        y_range: 顶点 Y 范围。
        spine_midline: 脊柱中线 (M, 3)，来自 fit_spine_midline。
        neck_root: 颈根点 (2, 3) [left, right]，用于解剖合理性校验。

    Returns:
        Tuple (scapular_peaks, debug):
            scapular_peaks: (2, 3) [left, right] 左右肩胛峰坐标。
            debug: 包含搜索参数、左右中间过程、对称性信息的字典。
    """
    # 主 Y 搜索区间 60%~80%：肩胛骨上角位于 T2-T3 水平
    y_lo = y_min + 0.60 * y_range
    y_hi = y_min + 0.80 * y_range
    mask = (vertices[:, 1] >= y_lo) & (vertices[:, 1] <= y_hi)
    band_size = int(mask.sum())
    logger.info(
        f"Searching scapular peak: y_range={y_range:.1f}, band=[{y_lo:.1f}, {y_hi:.1f}], size={band_size}"
    )
    if band_size < 10:
        logger.warning(f"Band size {band_size} < 10, fallback to 50%-85% Y range")
        # 回退：放宽到 50%~85%
        y_lo = y_min + 0.50 * y_range
        y_hi = y_min + 0.85 * y_range
        mask = (vertices[:, 1] >= y_lo) & (vertices[:, 1] <= y_hi)
        band_size = int(mask.sum())

    band_indices: np.ndarray = np.where(mask)[0]
    band = vertices[mask]

    # 从 spine_midline 取搜索区间内的点，中值作为分界中心
    spine_in_band = spine_midline[
        (spine_midline[:, 1] >= y_lo) & (spine_midline[:, 1] <= y_hi)
    ]
    if len(spine_in_band) >= 3:
        mid_x = float(np.median(spine_in_band[:, 0]))
    elif len(spine_midline) > 0:
        mid_x = float(np.median(spine_midline[:, 0]))
    else:
        mid_x = float(np.median(band[:, 0]))

    # 验证 mid_x 在 band X 范围内（spine_midline 可能不可靠）
    band_x_min, band_x_max = float(band[:, 0].min()), float(band[:, 0].max())
    if not (band_x_min < mid_x < band_x_max):
        logger.warning(
            f"mid_x {mid_x:.1f} outside band X range [{band_x_min:.1f}, {band_x_max:.1f}], "
            f"fallback to band median"
        )
        mid_x = float(np.median(band[:, 0]))

    x_span = float(band[:, 0].max() - band[:, 0].min())
    margin = 0.015 * x_span  # 中线缓冲区，防止分界上的点被两侧都取到

    # 左右分侧检测
    left_peak, left_debug = _detect_one_side(band, band_indices, True, mid_x, margin)
    right_peak, right_debug = _detect_one_side(band, band_indices, False, mid_x, margin)
    results = np.stack([left_peak, right_peak])

    # 对称性校验：两侧 Y 差超过 3% 身高时触发修正
    dy_mm = abs(float(results[0][1] - results[1][1]))
    corrected = dy_mm > 0.03 * y_range
    if corrected:
        logger.warning(
            f"Symmetry correction triggered: dY={dy_mm:.1f}mm > {0.03 * y_range:.1f}mm"
        )
        # 保留 Y 较高（偏上，更靠近脖子）的一侧
        if results[0][1] >= results[1][1]:
            ref_peak, bad_side_name, bad_idx = results[0], "right", 1
        else:
            ref_peak, bad_side_name, bad_idx = results[1], "left", 0
        _, corrected_debug = _detect_one_side(
            band,
            band_indices,
            bad_side_name == "left",
            mid_x,
            margin,
            target_y=ref_peak[1],
        )
        results[bad_idx] = corrected_debug["peak"]
        if bad_side_name == "left":
            left_debug = corrected_debug
        else:
            right_debug = corrected_debug

    debug = {
        "y_lo": y_lo,
        "y_hi": y_hi,
        "mid_x": mid_x,
        "band_size": band_size,
        "left": left_debug,
        "right": right_debug,
        "symmetry": {
            "dy_mm": dy_mm,
            "corrected": corrected,
        },
    }
    return results, debug
```

### landmarks/scapular_peak.py :: _detect_one_side (lines 109-186, len=78)

理由: 中等长度，未发现明显业务调用，但仍需人工确认。

```python
def _detect_one_side(
    band: np.ndarray,
    band_indices: np.ndarray,
    is_left: bool,
    mid_x: float,
    margin: float,
    target_y: float | None = None,
) -> tuple[np.ndarray, dict]:
    """在 band 中检测一侧肩胛峰。

    Args:
        band: Y band 内的顶点 (B, 3)。
        band_indices: band 顶点在原 vertices 中的索引 (B,)。
        is_left: True=左侧, False=右侧。
        mid_x: 分界中线 X。
        margin: 中线缓冲区宽度。
        target_y: 对称修正时指定的目标 Y，在该 Y 附近选取。

    Returns:
        Tuple (peak, debug):
            peak: 该侧肩胛峰坐标 (3,)。
            debug: 该侧的中间过程字典。
    """
    side_mask = band[:, 0] < mid_x - margin if is_left else band[:, 0] >= mid_x + margin

    side = band[side_mask]
    side_indices = band_indices[side_mask]
    side_size = len(side)
    z_threshold = 0.0
    high_z_size = 0
    candidate_indices: np.ndarray = np.array([], dtype=np.int64)
    fallback = False
    peak: np.ndarray = np.zeros(3)

    if side_size < 3:
        # 点数太少，取该侧 Z 最大点
        logger.warning(
            f"{'Left' if is_left else 'Right'} side_size {side_size} < 3, fallback to Z max"
        )
        fallback = True
        if side_size > 0:
            peak = side[np.argmax(side[:, 2])]
        elif len(band) > 0:
            peak = band[0]
    else:
        # Z 值分位过滤（top 85%），降低阈值以保留更多候选
        z_threshold = float(np.percentile(side[:, 2], 85))
        high_z_mask = side[:, 2] >= z_threshold
        high_z = side[high_z_mask]
        high_z_indices = side_indices[high_z_mask]
        high_z_size = len(high_z)

        if high_z_size < 2:
            # 高 Z 点太少，取该侧 Y 最高点
            logger.warning(
                f"{'Left' if is_left else 'Right'} high_z_size {high_z_size} < 2, "
                f"fallback to Y max"
            )
            fallback = True
            peak = side[np.argmax(side[:, 1])]
        else:
            if target_y is not None:
                # 对称修正：在所有 high_z 候选点中找 Y 最接近目标值的点
                candidate_indices = high_z_indices
                peak = high_z[np.argmin(np.abs(high_z[:, 1] - target_y))]
            else:
                # 取前 K 个 Z 最高候选，从中选 Y 最高
                K = max(5, int(len(side) * 0.10))
                k_idx = np.argsort(high_z[:, 2])[::-1][: min(K, high_z_size)]
                candidates = high_z[k_idx]
                candidate_indices = high_z_indices[k_idx]
                peak = candidates[np.argmax(candidates[:, 1])]

    debug = {
        "side_size": side_size,
        "z_threshold": z_threshold,
        "high_z_size": high_z_size,
        "peak": peak,
        "candidate_indices": candidate_indices,
        "fallback": fallback,
    }
    return peak, debug
```

### landmarks/angle_utils.py :: _select_side_point (lines 134-197, len=64)

理由: 中等长度，未发现明显业务调用，但仍需人工确认。

```python
def _select_side_point(
    contour_xy: np.ndarray,
    dists: np.ndarray,
    neck_pt: np.ndarray,
    distance: float,
    nearest_idx: int,
    side: str,
    snap_rel_tol: float = 0.01,
) -> np.ndarray:
    """沿轮廓在指定一侧选择与 neck_pt 距离约为 distance 的点（优先精确内插）。

    规则：优先选择与目标距离误差最小的邻点；若误差在 snap_rel_tol 内则直接取该点；
    否则尝试在相邻两点上内插；若内插失败则退回到较近的邻点。
    """
    # 在候选点的一侧沿轮廓找到"弧步距离"位置的点，优先精确内插，退回到最近邻
    if side == "prev":
        candidate_indices = np.arange(0, nearest_idx, dtype=int)
        fallback_idx = 0
    else:
        candidate_indices = np.arange(nearest_idx + 1, len(contour_xy), dtype=int)
        fallback_idx = len(contour_xy) - 1

    if len(candidate_indices) == 0:
        fallback_pt = contour_xy[fallback_idx, :2].copy()
        return fallback_pt

    # 选择误差最小的点作为候选
    target_errors = np.abs(dists[candidate_indices] - distance)
    p1_idx = int(candidate_indices[int(np.argmin(target_errors))])
    p1 = contour_xy[p1_idx, :2].copy()
    p1_dist = float(dists[p1_idx])
    p1_rel_diff = abs(p1_dist - distance) / max(abs(distance), 1e-9)

    # 若相对误差足够小则直接使用该邻点
    if p1_rel_diff <= snap_rel_tol:
        return p1

    # 尝试使用相邻点进行内插以获得更精确的位置
    neighbor_indices = [
        neighbor_idx
        for neighbor_idx in (p1_idx - 1, p1_idx + 1)
        if 0 <= neighbor_idx < len(contour_xy)
        and neighbor_idx != p1_idx
        and (
            (side == "prev" and neighbor_idx < nearest_idx)
            or (side == "next" and neighbor_idx > nearest_idx)
        )
    ]
    if not neighbor_indices:
        return p1

    p2_idx = min(
        neighbor_indices, key=lambda neighbor_idx: abs(dists[neighbor_idx] - distance)
    )
    p2 = contour_xy[p2_idx, :2].copy()
    p2_dist = float(dists[p2_idx])

    interp_pt = _interpolate_point_at_distance(p1, p2, neck_pt, distance)
    if interp_pt is not None:
        return interp_pt

    # 内插失败则返回距离更接近的邻点
    fallback_pt = p1 if abs(p1_dist - distance) <= abs(p2_dist - distance) else p2
    return fallback_pt
```

### landmarks/angle_utils.py :: _compute_angle_and_cosine (lines 241-288, len=48)

理由: 中等长度，未发现明显业务调用，但仍需人工确认。

```python
def _compute_angle_and_cosine(
    pt_prev: np.ndarray,
    pt_next: np.ndarray,
    pt: np.ndarray,
) -> tuple[float, float, float, float]:
    """计算顺时针长轴转角与余弦值。

    给定轮廓遍历顺序的前后两点与候选点，构造向量并计算：
    1. 余弦值（向量夹角的余弦，值域 [-1, 1]）
    2. 顺时针转角（从 before→cand_pt 到 after→cand_pt，值域 [0°, 360°)，< 180°=顺时针转，> 180°=逆时针转）

    Args:
        pt_prev: 轮廓遍历方向的前一点坐标（至少包含 x,y）。
        pt_next: 轮廓遍历方向的后一点坐标（至少包含 x,y）。
        pt: 候选点坐标（至少包含 x,y）。

    Returns:
        Tuple 下列顺序：
        - cosine (float): 向量夹角余弦值。
        - clockwise_deg (float): 顺时针方向的角度，单位度，值域 [0, 360)。
        - dist_prev (float): pt_prev 到 pt 的欧氏距离。
        - dist_next (float): pt_next 到 pt 的欧氏距离。

    Notes:
        若任一向量长度小于 1e-9，返回退化值 (cosine=1.0, clockwise=0.0)。
    """
    v_before = pt_prev[:2] - pt[:2]
    v_after = pt_next[:2] - pt[:2]
    dist_before = float(np.linalg.norm(v_before))
    dist_after = float(np.linalg.norm(v_after))

    if dist_before < 1e-9 or dist_after < 1e-9:
        logger.info(
            f"_compute_angle_and_cosine: degenerate distances "
            f"before={dist_before:.3e}, after={dist_after:.3e} at cand_pt={pt[:2]}"
        )
        return 1.0, 0.0, dist_before, dist_after

    # 余弦值（arccos 得到 acute angle 0-180°）
    dot = np.dot(v_before, v_after)
    cosine = float(dot / (dist_before * dist_after))

    # 顺时针角 0-360°：atan2(cross, dot) 给出逆时针角 → 转成顺时针
    cross = float(v_before[0] * v_after[1] - v_before[1] * v_after[0])
    signed_deg = float(np.degrees(np.arctan2(cross, dot)))  # [-180, 180]
    clockwise_deg = (360.0 - signed_deg) % 360.0  # [0, 360)

    return cosine, clockwise_deg, dist_before, dist_after
```

## 解析失败文件列表
