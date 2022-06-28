from PIL import Image
from typing import Tuple
import numpy as np


def flip(data):
    temp = np.zeros(data.shape)
    for i in range(data.shape[0]):
        img = Image.fromarray(np.reshape(data[i, :], (int(np.sqrt(data.shape[1])), int(np.sqrt(data.shape[1])))))
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        temp[i, :] = np.reshape(np.array(img), (1, data.shape[1]))
    return temp


def center_crop(x: np.ndarray, center_crop_size: Tuple[int, int], **kwargs) -> np.ndarray:
    """Center crop

    Args:
        x (np.array): input image
        center_crop_size (Tuple[int, int]): crop size

    Returns:
        np.ndarray: cropped image
    """
    centerh, centerw = x.shape[0]//2, x.shape[1]//2
    halfh, halfw = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerh-halfh:centerh+halfh, centerw-halfw:centerw+halfw, :]


def apply_10_crop(img: np.ndarray, crop_size: Tuple[int, int] = (224,224)) -> np.ndarray:
    """Applies 10-crop method to image

    Args:
        img (np.ndarray): image. Expected shape is (H,W,C)
        crop_size (Tuple[int, ...], optional): crop size. Defaults to (224,224).

    Returns:
        np.ndarray: 10-crop with shape (10,crop_size[0],crop_size[1],C)
    """
    h = crop_size[0]
    w = crop_size[1]
    flipped_X = np.fliplr(img)
    crops = [
        img[:h,:w, :], # Upper Left
        img[:h, img.shape[1]-w:, :], # Upper Right
        img[img.shape[0]-h:, :w, :], # Lower Left
        img[img.shape[0]-h:, img.shape[1]-w:, :], # Lower Right
        center_crop(img, (h, w)),

        flipped_X[:h,:w, :],
        flipped_X[:h, flipped_X.shape[1]-w:, :],
        flipped_X[flipped_X.shape[0]-h:, :w, :],
        flipped_X[flipped_X.shape[0]-h:, flipped_X.shape[1]-w:, :],
        center_crop(flipped_X, (h, w))
    ]
    return np.array(crops)

