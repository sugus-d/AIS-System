"""Landmark-based anatomical regions for left-right asymmetry analysis.

Instead of Y-height-threshold based segmentation (:mod:`mesh.roi.segmentation`),
this module defines anatomical regions as UV-space polygons using the template
landmark coordinates (:mod:`parameterization.template`), then splits each region
into left / right sub-regions at the spine midline (U=0).

Regions (IDs compatible with :mod:`mesh.roi.segmentation`):
    ========  ==
    Shoulder   2  (upper back, neck → scapular line)
    Thoracic   0  (mid-upper back, scapular → axilla line)
    Lumbar     1  (lower back, axilla → waist line)
    Pelvic     3  (below waist line)
    ========  ==

Typical usage::

    from features.extractors.asymmetry.landmark_regions import classify_by_region, compute_region_asymmetry
    labels, sides = classify_by_region(uv_coords)
    ai_global, ai_per_region = compute_region_asymmetry(curvature, labels, sides)

Implementation split: 基础多边形/点包含测试在 :mod:`._regions`，
候选区域生成在 :mod:`._regions_gen`，特征计算在 :mod:`._features`，
候选区域特征矩阵在 :mod:`._features_candidates`。
"""

from ._features import (
    classify_by_region,
    compute_curvature_asymmetry,
    compute_region_asymmetry,
)
from ._features_candidates import (
    compute_candidate_asymmetry,
    compute_candidate_asymmetry_pairwise,
    compute_region_features,
)
from ._regions import (
    build_candidate_polygons,
    CANDIDATE_VERSIONS,
    SEG_LUMBAR,
    SEG_PELVIC,
    SEG_SHOULDER,
    SEG_THORACIC,
)

__all__ = [
    "SEG_SHOULDER",
    "SEG_THORACIC",
    "SEG_LUMBAR",
    "SEG_PELVIC",
    "CANDIDATE_VERSIONS",
    "build_candidate_polygons",
    "classify_by_region",
    "compute_region_asymmetry",
    "compute_curvature_asymmetry",
    "compute_candidate_asymmetry",
    "compute_candidate_asymmetry_pairwise",
    "compute_region_features",
]
