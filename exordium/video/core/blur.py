"""Blur detection utilities."""

import os

import cv2
import numpy as np

from exordium.video.core.io import image_to_np


def variance_of_laplacian(image: os.PathLike | np.ndarray) -> float:
    """Computes the variance of Laplacian of the image as a focus measure.

    Args:
        image (np.ndarray): image of shape (H, W, C)

    Return:
        float: focus measure

    """
    image_gray = image_to_np(image, "GRAY")
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


def is_blurry(image: os.PathLike | np.ndarray, threshold: float = 400.0) -> tuple[bool, float]:
    """Check if an image is blurry using Laplacian variance.

    Args:
        image: Image file path or numpy array.
        threshold: Blur threshold. Images with variance below this are blurry.
            Defaults to 400.0.

    Returns:
        Tuple of (is_blurry, variance) where is_blurry is True if the image
        is considered blurry.
    """
    vl = variance_of_laplacian(image)
    return vl < threshold, round(vl, ndigits=2)
