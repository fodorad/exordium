"""Video transformation and face manipulation utilities."""

import math
from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from exordium.video.core.bb import xyxy2full
from exordium.video.face.landmark.constants import FaceLandmarks
from exordium.video.face.landmark.facemesh import rotate_landmarks


def rotate_face(
    face: np.ndarray | torch.Tensor,
    rotation_degree: float,
) -> tuple[np.ndarray | torch.Tensor, np.ndarray]:
    """Rotate a face image by an explicit angle.

    Accepts either a numpy array or a torch tensor and returns the same type,
    avoiding unnecessary device transfers or dtype conversions.

    * **numpy** ``(H, W, 3)`` uint8 — uses ``cv2.warpAffine``; fast on CPU.
    * **torch** ``(C, H, W)`` or ``(B, C, H, W)`` — uses
      ``torchvision.transforms.functional.rotate``; stays on the original
      device and dtype.

    Args:
        face: Face image as ``np.ndarray (H, W, 3)`` or
            ``torch.Tensor (C, H, W)`` / ``(B, C, H, W)``.
        rotation_degree: Counter-clockwise rotation in degrees.

    Returns:
        Tuple of ``(rotated_face, rotation_matrix)`` where ``rotated_face``
        has the same type and shape as the input and ``rotation_matrix`` is
        the ``(2, 3)`` affine matrix used for numpy rotation (or ``np.eye(2)``
        for the zero-rotation fast-path and tensor inputs).

    """
    if rotation_degree == 0:
        return face, np.eye(2)

    if isinstance(face, torch.Tensor):
        rotated = TF.rotate(face, angle=rotation_degree)
        return rotated, np.eye(2)

    # numpy path
    height, width = face.shape[:2]
    R = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_degree, 1)
    face_rotated = cv2.warpAffine(face, R, (width, height))
    return face_rotated, R


def align_face(
    image: np.ndarray | torch.Tensor,
    bb_xyxy: np.ndarray | torch.Tensor,
    landmarks: np.ndarray,
) -> dict:
    """Align face image to canonical orientation using landmarks.

    Aligns the x-axis of the Head Coordinate System (HCS) to the
    x-axis of the Camera Coordinate System (CCS). The face is rotated
    based on the eye positions to align it horizontally.

    Accepts ``(H, W, C)`` numpy arrays or ``(C, H, W)`` uint8 torch tensors.
    The ``"rotated_image"`` and ``"rotated_face"`` entries in the returned
    dictionary preserve the input type.

    Args:
        image: Input image — ``np.ndarray (H, W, C)`` or
            ``torch.Tensor (C, H, W)`` uint8.
        bb_xyxy: Bounding box of shape ``(4,)`` — numpy or tensor.
        landmarks: Landmark coordinates of shape ``(5, 2)`` — numpy array.

    Raises:
        ValueError: If landmarks do not have shape ``(5, 2)``.

    Returns:
        Dictionary with keys:

        * ``"rotated_image"`` — same type as ``image``
        * ``"rotated_face"`` — same type as ``image``
        * ``"rotated_bb_xyxy"`` — ``np.ndarray (4,)`` int
        * ``"rotation_degree"`` — float
        * ``"rotation_matrix"`` — ``np.ndarray (2, 3)`` affine matrix

    """
    if landmarks.shape != (5, 2):
        raise ValueError(f"Expected landmarks with shape (5, 2) got instead {landmarks.shape}.")

    if isinstance(image, torch.Tensor):
        img_np: np.ndarray = image.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = cast("np.ndarray", image)

    # Normalise bb to numpy for rotate_landmarks
    bb_np = bb_xyxy.cpu().numpy() if isinstance(bb_xyxy, torch.Tensor) else np.asarray(bb_xyxy)

    lmks = np.rint(landmarks).astype(int)
    left_eye_x, left_eye_y = lmks[FaceLandmarks.LEFT_EYE.value, :]
    right_eye_x, right_eye_y = lmks[FaceLandmarks.RIGHT_EYE.value, :]

    dY = right_eye_y - left_eye_y
    dX = right_eye_x - left_eye_x
    rotation_degree = math.degrees(math.atan2(dY, dX)) - 180

    height, width = img_np.shape[:2]
    image_center = (width // 2, height // 2)
    R = cv2.getRotationMatrix2D(image_center, rotation_degree, 1.0)
    abs_cos = abs(R[0, 0])
    abs_sin = abs(R[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    R[0, 2] += bound_w / 2 - image_center[0]
    R[1, 2] += bound_h / 2 - image_center[1]
    rotated_np = cv2.warpAffine(img_np, R, (bound_w, bound_h))

    rotated_bb_full = rotate_landmarks(cast("np.ndarray", xyxy2full(bb_np)), R)
    min_x, min_y = np.min(rotated_bb_full, axis=0)
    max_x, max_y = np.max(rotated_bb_full, axis=0)
    rotated_bb_xyxy = np.array([min_x, min_y, max_x, max_y], dtype=int)
    rotated_face_np = rotated_np[min_y:max_y, min_x:max_x]

    if isinstance(image, torch.Tensor):
        rotated_image_out = torch.from_numpy(rotated_np).permute(2, 0, 1)
        rotated_face_out = torch.from_numpy(rotated_face_np).permute(2, 0, 1)
    else:
        rotated_image_out = rotated_np
        rotated_face_out = rotated_face_np

    return {
        "rotated_image": rotated_image_out,
        "rotated_face": rotated_face_out,
        "rotated_bb_xyxy": rotated_bb_xyxy,
        "rotation_degree": rotation_degree,
        "rotation_matrix": R,
    }


class ToTensor:
    """Converts np.ndarray with value range [0..255] to torch Tensor with value range [0..1].

    Video processing class, similar to torchvision but for videos.
    """

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to normalized torch tensor.

        Args:
            x: Input numpy array with value range [0..255].

        Returns:
            Torch tensor with value range [0..1].
        """
        return torch.from_numpy(x).float() / 255.0


class Resize:
    """Resizes video represented as a torch Tensor of shape (C, T, H, W).

    Video processing class, similar to torchvision but for videos.
    """

    def __init__(self, size: int | None, mode: str = "bilinear"):
        self.size = size
        self.mode = mode

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Resize video tensor.

        Args:
            video: Input tensor of shape (C, T, H, W).

        Returns:
            Resized tensor.
        """
        size = self.size
        scale = None

        if isinstance(size, int):
            scale = float(size) / min(video.shape[-2:])
            size = None

        return torch.nn.functional.interpolate(
            video,
            size=size,
            scale_factor=scale,
            mode=self.mode,
            align_corners=False,
            recompute_scale_factor=True,
        )


class CenterCrop:
    """Center crops a video represented as a torch Tensor of shape (C, T, H, W).

    Video processing class, similar to torchvision but for videos.
    """

    def __init__(self, size: int | tuple[int, int]):
        self.size = size

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Center crop video tensor.

        Args:
            video: Input tensor of shape (C, T, H, W).

        Returns:
            Center-cropped tensor.
        """
        size = self.size

        if isinstance(size, int):
            size = size, size

        th, tw = size
        h, w = video.shape[-2:]
        i = int(round((h - th) / 2.0))
        j = int(round((w - tw) / 2.0))
        return video[..., i : (i + th), j : (j + tw)]


class Normalize:
    """Standardizes a video represented as a torch Tensor of shape (C, T, H, W).

    Video processing class, similar to torchvision but for videos.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Standardize video tensor using mean and std.

        Args:
            video: Input tensor of shape (C, T, H, W).

        Returns:
            Standardized tensor.
        """
        shape = (-1,) + (1,) * (video.dim() - 1)
        mean = torch.as_tensor(self.mean, device=video.device).reshape(shape)
        std = torch.as_tensor(self.std, device=video.device).reshape(shape)
        return (video - mean) / std


class Denormalize:
    """Reverses the standardization of a video represented as a torch Tensor of shape (C, T, H, W).

    Video processing class, similar to torchvision but for videos.
    """

    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Reverse standardization on video tensor.

        Args:
            video: Input tensor of shape (C, T, H, W).

        Returns:
            Denormalized tensor.
        """
        shape = (-1,) + (1,) * (video.dim() - 1)
        mean = torch.as_tensor(self.mean, device=video.device).reshape(shape)
        std = torch.as_tensor(self.std, device=video.device).reshape(shape)
        return (video * std) + mean
