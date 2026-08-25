"""Parameterization package.

Pipeline modules:
  template.py       — Template landmark UV coordinates.
  landmark_io.py    — JSON parsing + vertex matching.
  harmonic.py       — Cotangent-Laplacian harmonic parameterization.
  geodesic_cut.py   — Geodesic boundary computation + mesh cutting.
  pipeline.py       — End-to-end orchestration.
  arap.py           — (Experimental) ARAP parameterization.
"""

from .harmonic import harmonic_parameterize
from .landmark_io import find_landmark_vertices, parse_landmarks_json
from .pipeline import run_pipeline
from .template import TEMPLATE_LANDMARKS

__all__ = [
    "TEMPLATE_LANDMARKS",
    "find_landmark_vertices",
    "harmonic_parameterize",
    "parse_landmarks_json",
    "run_pipeline",
]
