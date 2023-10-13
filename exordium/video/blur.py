from PIL import Image
import cv2
import numpy as np
from exordium import PathType
from exordium.video.io import image2np


def variance_of_laplacian(image: PathType | Image.Image | np.ndarray) -> float:
    """Computes the variance of Laplacian of the image as a focus measure

    Args:
        image (np.ndarray): image of shape (H, W, C)

    Return:
        float: focus measure
    """
    image_gray = image2np(image, 'GRAY')
    return cv2.Laplacian(image_gray, cv2.CV_64F).var()


def is_blurry(image: PathType | Image.Image | np.ndarray, threshold: float = 400.) -> tuple[bool, float]:
    vl = variance_of_laplacian(image)
    return vl < threshold, round(vl, ndigits=2)