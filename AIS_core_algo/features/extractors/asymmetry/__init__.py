"""非对称性特征提取包 — 225 区域 x 8 种测量 x DM/PW 差异 + Composite Index。"""

from .asymmetric_index import compute_asymmetric_index
from .differences import compute_asymmetry_dm, compute_asymmetry_pw
from .extract import extract_asymmetry
from .landmark_regions import (
    build_candidate_polygons,
    classify_by_region,
    compute_candidate_asymmetry,
    compute_candidate_asymmetry_pairwise,
    compute_curvature_asymmetry,
    compute_region_asymmetry,
    compute_region_features,
    SEG_LUMBAR,
    SEG_PELVIC,
    SEG_SHOULDER,
    SEG_THORACIC,
)
from .measures import (
    compute_gauss_curvature,
    compute_height,
    compute_mean_curvature,
    compute_normal_angle,
    compute_normal_cos,
    compute_roughness,
)
from .regions import build_region_polygons, mask_vertices, points_in_polygon
from .surface_height import compute_surface_height
from .z_index import compute_z_index

__all__ = [
    "build_candidate_polygons",
    "build_region_polygons",
    "classify_by_region",
    "compute_asymmetric_index",
    "compute_asymmetry_dm",
    "compute_asymmetry_pw",
    "compute_candidate_asymmetry",
    "compute_candidate_asymmetry_pairwise",
    "compute_curvature_asymmetry",
    "compute_gauss_curvature",
    "compute_height",
    "compute_mean_curvature",
    "compute_normal_angle",
    "compute_normal_cos",

    "compute_region_asymmetry",
    "compute_region_features",
    "compute_roughness",
    "compute_surface_height",
    "compute_z_index",
    "extract_asymmetry",
    "mask_vertices",
    "points_in_polygon",
    "SEG_LUMBAR",
    "SEG_PELVIC",
    "SEG_SHOULDER",
    "SEG_THORACIC",
]
