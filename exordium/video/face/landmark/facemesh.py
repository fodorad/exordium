import os
from collections.abc import Sequence
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from exordium.utils.decorator import load_or_create
from exordium.video.core.detection import Track
from exordium.video.core.io import batch_iterator


class FaceMeshWrapper:  # pragma: no cover
    """Wrapper for the MediaPipe FaceMesh landmark detector."""

    def __init__(self):
        """Initializes the MediaPipe FaceMesh model."""
        mp_face_mesh = mp.solutions.face_mesh
        self.model = mp_face_mesh.FaceMesh()

    def __call__(self, rgb_images: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Detects 468 facial landmarks for each image in the batch.

        Only the first detected face per image is used. Images with no
        detected face are silently skipped.

        Args:
            rgb_images (Sequence[np.ndarray]): Sequence of RGB images each of
                shape (H, W, 3) with uint8 values.

        Returns:
            list[np.ndarray]: List of landmark arrays each of shape (468, 2)
                containing (x, y) pixel coordinates.

        """
        batch = []
        for rgb_image in rgb_images:
            height, width, _ = rgb_image.shape
            result = self.model.process(rgb_image)
            if not result.multi_face_landmarks:
                continue
            face = result.multi_face_landmarks[0]  # first face
            landmarks = []
            for i in range(468):
                pt = face.landmark[i]
                x = int(pt.x * width)
                y = int(pt.y * height)
                landmarks.append((x, y))
            batch.append(np.array(landmarks))
        return batch

    @load_or_create("pkl")
    def track_to_feature(
        self, track: Track, batch_size: int = 30, **_kwargs
    ) -> tuple[list, np.ndarray]:
        """Extracts FaceMesh landmarks for all non-interpolated detections in a track.

        Args:
            track (Track): Track containing a sequence of Detection objects.
            batch_size (int, optional): Number of detections to process per
                batch. Defaults to 30.
            **_kwargs: Additional keyword arguments forwarded to
                ``load_or_create`` (e.g. ``output_path``, ``overwrite``).

        Returns:
            tuple[list, np.ndarray]: A tuple of (ids, features) where ids is a
                list of int frame IDs and features is an array of shape
                (N, 468, 2).

        """
        ids, features = [], []
        for subset in batch_iterator(track, batch_size):
            ids += [detection.frame_id for detection in subset if not detection.is_interpolated]
            samples = [
                detection.bb_crop() for detection in subset if not detection.is_interpolated
            ]  # (B, H, W, C)
            feature = self(samples)
            features.append(feature)
        features = np.concatenate(features, axis=0)
        return ids, features


def rotate_landmarks(landmarks: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate landmarks using a transformation matrix.

    Args:
        landmarks: Landmark coordinates of shape (N, 2).
        R: Rotation/transformation matrix of shape (2, 3).

    Returns:
        Rotated landmarks of shape (N, 2) as integers.
    """
    return np.rint(
        np.dot(R, np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1).T).T
    ).astype(int)


def visualize_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_path: str | os.PathLike | None = None,
    show_indices: bool = True,
    radius: int = 1,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    fontScale: float = 0.3,
    font_thickness: int = 1,
) -> np.ndarray:
    """Draw 2D landmarks onto an image.

    Args:
        image: Input image of shape ``(H, W, C)``.
        landmarks: Landmark coordinates of shape ``(N, 2)``.
        output_path: Path to save the output image. ``None`` skips saving.
        show_indices: Draw landmark indices next to each point.
        radius: Radius of each landmark circle. Defaults to 1.
        color: BGR color for landmarks. Defaults to ``(0, 255, 0)``.
        thickness: Circle border thickness. Defaults to 2.
        font: OpenCV font type.
        fontScale: Font scale for index labels. Defaults to 0.3.
        font_thickness: Thickness of index label text. Defaults to 1.

    Returns:
        Copy of the image with landmarks drawn on it.

    Raises:
        Exception: If ``landmarks`` does not have shape ``(N, 2)``.

    """
    if not (landmarks.ndim == 2 and landmarks.shape[1] == 2):
        raise Exception(f"Expected landmarks with shape (N, 2) got instead {landmarks.shape}.")

    image_out = np.copy(image)
    landmarks = np.rint(landmarks).astype(int)

    for index in range(len(landmarks)):
        cv2.circle(image_out, landmarks[index, :], radius, color, thickness)
        if show_indices:
            cv2.putText(
                image_out,
                str(index),
                landmarks[index, :] + 5,
                font,
                fontScale,
                color,
                font_thickness,
                cv2.LINE_AA,
            )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image_out)

    return image_out
