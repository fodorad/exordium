"""MediaPipe FaceMesh landmark detector wrapper using latest tasks API."""

import logging
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from exordium.utils.decorator import load_or_create
from exordium.video.core.detection import Track
from exordium.video.core.io import batch_iterator


class FaceMeshWrapper:
    """Wrapper for the MediaPipe FaceMesh landmark detector.

    Detects 478 dense facial landmarks using MediaPipe's latest tasks API.
    Returns pixel-space coordinates for all face keypoints.

    Args:
        min_detection_confidence: Minimum confidence threshold for face
            detection. Default: 0.5.
        min_tracking_confidence: Minimum confidence threshold for tracking.
            Default: 0.5.

    Example:
        >>> import numpy as np
        >>> from exordium.video.face.landmark.facemesh import FaceMeshWrapper
        >>> wrapper = FaceMeshWrapper()
        >>> face_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        >>> landmarks = wrapper([face_image])  # list of detected faces
        >>> if landmarks:
        ...     print(landmarks[0].shape)  # (478, 2) — 478 landmarks, x,y coords
    """

    def __init__(
        self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5
    ):
        """Initialize FaceMeshWrapper with MediaPipe face mesh model.

        Args:
            min_detection_confidence: Minimum confidence for face detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        model_path = self._download_model()

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        logging.info("MediaPipe FaceMesh loaded.")

    def _download_model(self) -> str:
        """Download MediaPipe face landmarker model if not present.

        Returns:
            Path to the downloaded model file.

        Raises:
            RuntimeError: If download fails.
        """
        model_dir = Path.home() / ".cache" / "mediapipe_models"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_name = "face_landmarker.task"
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

        model_path = model_dir / model_name

        if not model_path.exists():
            logging.info(f"Downloading MediaPipe model: {model_name}")
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download MediaPipe model: {e}") from e

        return str(model_path)

    def __call__(self, rgb_images: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Detect 478 dense facial landmarks for face images.

        **Input:** Face image (typically cropped from a larger frame)
        **Output:** 478 landmark coordinates (x, y) in pixel space

        Args:
            rgb_images: Sequence of face images, each of shape ``(H, W, 3)``
                with uint8 values in RGB format. Images should be face crops
                (e.g., from face detection bounding boxes).

        Returns:
            List of landmark arrays, one per input image. Each array has shape
            ``(478, 2)`` containing (x, y) pixel coordinates for all 478 facial
            landmarks. Images with no detected face are silently skipped (not
            included in output list).

        Example:
            >>> face_crop = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            >>> landmarks_list = wrapper([face_crop])
            >>> if landmarks_list:
            ...     lmks = landmarks_list[0]  # shape (478, 2)
            ...     print(f"Detected {len(lmks)} landmarks")
        """
        batch = []
        for rgb_image in rgb_images:
            h, w = rgb_image.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            result = self.landmarker.detect(mp_image)

            if result.face_landmarks:
                # Use first detected face
                face_landmarks = result.face_landmarks[0]
                landmarks = []
                for landmark in face_landmarks:
                    x = landmark.x * w
                    y = landmark.y * h
                    landmarks.append([x, y])
                batch.append(np.array(landmarks, dtype=np.float32))

        return batch

    @load_or_create("pkl")
    def track_to_feature(
        self, track: Track, batch_size: int = 30, **_kwargs
    ) -> tuple[list, np.ndarray]:
        """Extract FaceMesh landmarks for all non-interpolated detections in a track.

        Processes detections in batches and caches results using the
        ``load_or_create`` decorator.

        Args:
            track: Track containing a sequence of Detection objects.
            batch_size: Number of detections to process per batch. Default: 30.
            **_kwargs: Additional keyword arguments forwarded to
                ``load_or_create`` (e.g. ``output_path``, ``overwrite``).

        Returns:
            Tuple of ``(frame_ids, landmarks)`` where ``frame_ids`` is a list
            of frame indices and ``landmarks`` is a numpy array of shape
            ``(N, 478, 2)`` containing 478 landmarks per detection.
        """
        ids, features = [], []
        for subset in batch_iterator(track, batch_size):
            ids += [detection.frame_id for detection in subset if not detection.is_interpolated]
            samples = [
                detection.bb_crop() for detection in subset if not detection.is_interpolated
            ]
            feature = self(samples)
            if feature:
                features.append(np.array(feature))
        if features:
            features = np.concatenate(features, axis=0)
        else:
            features = np.empty((0, 468, 2), dtype=np.float32)
        return ids, features

    def __del__(self) -> None:
        """Clean up landmarker resources."""
        if hasattr(self, "landmarker"):
            self.landmarker.close()


def rotate_landmarks(landmarks: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate landmarks using a transformation matrix.

    Args:
        landmarks: Landmark coordinates of shape ``(N, 2)``.
        R: Rotation/transformation matrix of shape ``(2, 3)``.

    Returns:
        Rotated landmarks of shape ``(N, 2)`` as integers.
    """
    return np.rint(
        np.dot(R, np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1).T).T
    ).astype(int)


def visualize_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_path: str | Path | None = None,
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
        image: Input image of shape ``(H, W, C)`` in BGR format.
        landmarks: Landmark coordinates of shape ``(N, 2)``.
        output_path: Path to save the output image. ``None`` skips saving.
        show_indices: Draw landmark indices next to each point. Default: True.
        radius: Radius of each landmark circle. Default: 1.
        color: BGR color for landmarks. Default: ``(0, 255, 0)`` (green).
        thickness: Circle border thickness. Default: 2.
        font: OpenCV font type. Default: cv2.FONT_HERSHEY_SIMPLEX.
        fontScale: Font scale for index labels. Default: 0.3.
        font_thickness: Thickness of index label text. Default: 1.

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
