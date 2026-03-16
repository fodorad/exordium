from enum import Enum

import numpy as np


class FaceLandmarks(Enum):
    """5-point face landmark indices (right_eye, left_eye, nose, mouth_right, mouth_left)."""

    RIGHT_EYE = 0
    LEFT_EYE = 1
    NOSE = 2
    MOUTH_RIGHT = 3
    MOUTH_LEFT = 4


class IrisLandmarks(Enum):
    """Defines MediaPipe Iris indices.

        2
    3   0   1
        4
    """

    CENTER = 0
    RIGHT = 1
    TOP = 2
    LEFT = 3
    BOTTOM = 4


class TddfaLandmarks(Enum):
    """Defines 3DDFA_V2 landmark indices.

       1   2
    0         3
       5   4
    """

    LEFT = 0
    TOP_LEFT = 1
    TOP_RIGHT = 2
    RIGHT = 3
    BOTTOM_RIGHT = 4
    BOTTOM_LEFT = 5


class FaceMeshLandmarks:
    """Defines MediaPipe FaceMesh indices.

        10 11 12 13 14
      9                15
    0                     8
      1                 7
         2  3  4  5  6
    """

    EYE = np.arange(16)
    BOTTOM_ALL = np.arange(1, 8)
    TOP_ALL = np.arange(9, 16)
    BOTTOM = np.array([4])
    TOP = np.array([12])
    LEFT = np.array([0])
    RIGHT = np.array([8])
    TOP_LEFT = np.array([11])
    TOP_RIGHT = np.array([13])
    BOTTOM_LEFT = np.array([3])
    BOTTOM_RIGHT = np.array([5])
