"""Facial landmark definitions and constants."""

from enum import Enum


class FaceLandmarks(Enum):
    """5-point face landmark indices (right_eye, left_eye, nose, mouth_right, mouth_left)."""

    RIGHT_EYE = 0
    """Subject's right eye (viewer's left)."""
    LEFT_EYE = 1
    """Subject's left eye (viewer's right)."""
    NOSE = 2
    """Nose tip."""
    MOUTH_RIGHT = 3
    """Right mouth corner (subject's right)."""
    MOUTH_LEFT = 4
    """Left mouth corner (subject's left)."""


class IrisLandmarks(Enum):
    """Defines MediaPipe Iris indices.

    Layout::

          2
      3   0   1
          4

    """

    CENTER = 0
    """Iris center point."""
    RIGHT = 1
    """Rightmost iris point."""
    TOP = 2
    """Topmost iris point."""
    LEFT = 3
    """Leftmost iris point."""
    BOTTOM = 4
    """Bottommost iris point."""


class FaceMeshLandmarks:
    """Defines MediaPipe FaceMesh indices.

    Layout::

          10 11 12 13 14
        9                15
      0                     8
        1                 7
           2  3  4  5  6

    """

    EYE = list(range(16))
    """All 16 iris landmark indices."""
    BOTTOM_ALL = list(range(1, 8))
    """Indices for the lower half of the iris contour (positions 1–7)."""
    TOP_ALL = list(range(9, 16))
    """Indices for the upper half of the iris contour (positions 9–15)."""
    BOTTOM = [4]
    """Index of the bottommost iris point."""
    TOP = [12]
    """Index of the topmost iris point."""
    LEFT = [0]
    """Index of the leftmost iris point."""
    RIGHT = [8]
    """Index of the rightmost iris point."""
    TOP_LEFT = [11]
    """Index of the upper-left iris point."""
    TOP_RIGHT = [13]
    """Index of the upper-right iris point."""
    BOTTOM_LEFT = [3]
    """Index of the lower-left iris point."""
    BOTTOM_RIGHT = [5]
    """Index of the lower-right iris point."""


class FaceMesh478Regions:
    """MediaPipe FaceMesh 478-point landmark index groups by facial region.

    Naming follows MediaPipe's **subject-centric** convention — RIGHT/LEFT refers
    to the subject's anatomical right/left.  When looking at a frontal face image:

    * Subject's **RIGHT** side → **left** side of the image (viewer's perspective).
    * Subject's **LEFT** side → **right** side of the image (viewer's perspective).

    Landmark index sources:

    * Eyes / eyebrows: extracted from ``FACEMESH_RIGHT_EYE``,
      ``FACEMESH_LEFT_EYE``, ``FACEMESH_RIGHT_EYEBROW``,
      ``FACEMESH_LEFT_EYEBROW`` connection sets.
    * Mouth: extracted from ``FACEMESH_LIPS`` connection set, split at the
      vertical midline.  ``MOUTH_LEFT`` contains the viewer's-left half
      (subject's right commissure, index 61/78) plus the four midline points.
      ``MOUTH_RIGHT`` contains the viewer's-right half (subject's left
      commissure, index 291/308).
    * Nose: tip, alar wings, columella, and bridge.
    * Face oval: outer silhouette from ``FACEMESH_FACE_OVAL`` connection set.
    """

    # ------------------------------------------------------------------
    # Eyes  (16 landmarks each)
    # ------------------------------------------------------------------

    # Subject's right eye — appears on the viewer's LEFT of a frontal image.
    RIGHT_EYE: list[int] = [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        246,
        161,
        160,
        159,
        158,
        157,
        173,
    ]
    """Subject's right-eye landmark indices (viewer's left side, 16 points)."""

    # Subject's left eye — appears on the viewer's RIGHT of a frontal image.
    LEFT_EYE: list[int] = [
        263,
        249,
        390,
        373,
        374,
        380,
        381,
        382,
        362,
        466,
        388,
        387,
        386,
        385,
        384,
        398,
    ]
    """Subject's left-eye landmark indices (viewer's right side, 16 points)."""

    # ------------------------------------------------------------------
    # Eyebrows  (10 landmarks each)
    # ------------------------------------------------------------------

    # Subject's right eyebrow — viewer's left.
    RIGHT_EYEBROW: list[int] = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
    """Subject's right-eyebrow landmark indices (viewer's left side, 10 points)."""

    # Subject's left eyebrow — viewer's right.
    LEFT_EYEBROW: list[int] = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
    """Subject's left-eyebrow landmark indices (viewer's right side, 10 points)."""

    # ------------------------------------------------------------------
    # Nose  (tip + bridge + alar wings + nostrils, ~40 landmarks)
    # ------------------------------------------------------------------

    NOSE: list[int] = [
        1,
        2,
        4,
        5,
        6,
        19,
        44,
        45,
        48,
        51,
        64,
        94,
        98,
        102,
        115,
        125,
        129,
        131,
        134,
        168,
        195,
        197,
        240,
        241,
        242,
        274,
        275,
        278,
        281,
        294,
        327,
        331,
        344,
        354,
        358,
        360,
        363,
        460,
        461,
        462,
    ]
    """Nose landmark indices — tip, bridge, alar wings, nostrils (~40 points)."""

    # ------------------------------------------------------------------
    # Mouth  (viewer-perspective left/right split at midline)
    # ------------------------------------------------------------------

    # Viewer's LEFT half of mouth (subject's right commissure + midline).
    # Includes outer/inner lip landmarks near indices 0, 13, 14, 17, 37–95.
    MOUTH_LEFT: list[int] = [
        0,
        13,
        14,
        17,
        37,
        39,
        40,
        61,
        78,
        80,
        81,
        82,
        84,
        87,
        88,
        91,
        95,
        146,
        178,
        181,
        185,
        191,
    ]
    """Viewer's-left mouth landmark indices (subject's right commissure + midline)."""

    # Viewer's RIGHT half of mouth (subject's left commissure).
    # Includes outer/inner lip landmarks near indices 267–415.
    MOUTH_RIGHT: list[int] = [
        267,
        269,
        270,
        291,
        308,
        310,
        311,
        312,
        314,
        317,
        318,
        321,
        324,
        375,
        402,
        405,
        409,
        415,
    ]
    """Viewer's-right mouth landmark indices (subject's left commissure)."""

    # ------------------------------------------------------------------
    # Face oval  (outer silhouette, 36 landmarks)
    # ------------------------------------------------------------------

    FACE_OVAL: list[int] = [
        10,
        21,
        54,
        58,
        67,
        93,
        103,
        109,
        127,
        132,
        136,
        148,
        149,
        150,
        152,
        162,
        172,
        176,
        234,
        251,
        284,
        288,
        297,
        323,
        332,
        338,
        356,
        361,
        365,
        377,
        378,
        379,
        389,
        397,
        400,
        454,
    ]
    """Outer face silhouette landmark indices (36 points)."""


def build_facemesh_region_colors(
    right_eye_color: tuple[int, int, int] = (0, 128, 0),
    left_eye_color: tuple[int, int, int] = (100, 238, 100),
    eyebrow_color: tuple[int, int, int] = (180, 0, 180),
    nose_color: tuple[int, int, int] = (255, 0, 0),
    mouth_left_color: tuple[int, int, int] = (0, 0, 200),
    mouth_right_color: tuple[int, int, int] = (50, 150, 200),
    face_oval_color: tuple[int, int, int] = (50, 50, 50),
    other_color: tuple[int, int, int] = (220, 220, 220),
) -> list[tuple[int, int, int]]:
    """Build a 478-element per-landmark color list for FaceMesh regional visualization.

    Colours are in **RGB** order (matching the image format used throughout
    the library).  Pass the result as the ``colors`` argument of
    :func:`~exordium.video.face.landmark.facemesh.visualize_landmarks`.

    Default color scheme (all RGB):

    * Right eye (viewer's left)    → dark green    ``(0, 128, 0)``
    * Left eye  (viewer's right)   → light green   ``(100, 238, 100)``
    * Eyebrows                     → dark magenta  ``(180, 0, 180)``
    * Nose                         → red           ``(255, 0, 0)``
    * Mouth left  (viewer's left)  → dark blue     ``(0, 0, 200)``
    * Mouth right (viewer's right) → light blue    ``(50, 150, 200)``
    * Face oval                    → dark grey     ``(50, 50, 50)``
    * All other landmarks          → light grey    ``(220, 220, 220)``

    Args:
        right_eye_color: RGB color for ``FaceMesh478Regions.RIGHT_EYE``.
        left_eye_color: RGB color for ``FaceMesh478Regions.LEFT_EYE``.
        eyebrow_color: RGB color for both eyebrow regions.
        nose_color: RGB color for nose landmarks.
        mouth_left_color: RGB color for viewer's-left mouth half.
        mouth_right_color: RGB color for viewer's-right mouth half.
        face_oval_color: RGB color for outer face silhouette.
        other_color: RGB color for all remaining landmarks.

    Returns:
        List of 478 ``(R, G, B)`` tuples, one per FaceMesh landmark.

    Example::

        from exordium.video.face.landmark.constants import FACEMESH_REGION_COLORS
        from exordium.video.face.landmark.facemesh import visualize_landmarks

        vis = visualize_landmarks(crop, landmarks, colors=FACEMESH_REGION_COLORS)

    """
    colors: list[tuple[int, int, int]] = [other_color] * 478
    for i in FaceMesh478Regions.RIGHT_EYE:
        colors[i] = right_eye_color
    for i in FaceMesh478Regions.LEFT_EYE:
        colors[i] = left_eye_color
    for i in FaceMesh478Regions.RIGHT_EYEBROW + FaceMesh478Regions.LEFT_EYEBROW:
        colors[i] = eyebrow_color
    for i in FaceMesh478Regions.NOSE:
        colors[i] = nose_color
    for i in FaceMesh478Regions.MOUTH_LEFT:
        colors[i] = mouth_left_color
    for i in FaceMesh478Regions.MOUTH_RIGHT:
        colors[i] = mouth_right_color
    for i in FaceMesh478Regions.FACE_OVAL:
        colors[i] = face_oval_color
    return colors


FACEMESH_REGION_COLORS: list[tuple[int, int, int]] = build_facemesh_region_colors()
"""Pre-built default regional color map — 478 RGB tuples, one per landmark.

See :func:`build_facemesh_region_colors` for the full color scheme.
"""
