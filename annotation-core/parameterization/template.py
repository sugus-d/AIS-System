"""Template landmark positions in UV space for harmonic parameterization.

The 18 anatomical landmarks are placed on a 5×5 grid:
  U ∈ [-2.5, 2.5], V ∈ [-4, 2]

Widths derived from data analysis: neck=base (U=±1), scapular peaks≈1.37→±1.5,
shoulder transition≈2.06→±2, axilla≈2.68→±2.5, waist≈2.2→±2, waist lower≈2.3.
Vertical positions: P0/neck=+2, scapular/P1=+1, axilla/P2=0, P5=-1.5, P3/waist=-3, P4/waist lower=-4.
"""

TEMPLATE_LANDMARKS: dict[str, tuple[float, float]] = {
    # Spine midline (U=0)
    "spine_P0": (0.0, 2.0),
    "spine_P1": (0.0, 1.0),
    "spine_P2": (0.0, 0.0),
    "spine_P3": (0.0, -3.0),
    # Bottom spine
    "spine_P4": (0.0, -4.0),
    # Mid-spine between P2 and P3
    "spine_P5": (0.0, -1.5),
    # Top row (V=+2): neck and shoulder transition
    "neck_root_L": (-0.75, 2.0),
    "neck_root_R": (0.75, 2.0),
    "shoulder_transition_L": (-1.75, 1.75),
    "shoulder_transition_R": (1.75, 1.75),
    # Scapular row (V=+1)
    "scapular_peaks_L": (-1.25, 1.0),
    "scapular_peaks_R": (1.25, 1.0),
    # Axilla row (V=0)
    "axilla_L": (-2.5, 0.0),
    "axilla_R": (2.5, 0.0),
    # Waist row (V=-3)
    "waist_L": (-2.0, -3.0),
    "waist_R": (2.0, -3.0),
    # Lower waist row (V=-4): waist lower edge
    "waist_lower_L": (-2.3, -4.0),
    "waist_lower_R": (2.3, -4.0),
}
