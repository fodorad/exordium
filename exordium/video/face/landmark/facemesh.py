"""MediaPipe FaceMesh landmark detector wrapper using latest tasks API."""

import logging
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from exordium.utils.decorator import load_or_create
from exordium.video.core.detection import Track
from exordium.video.core.io import batch_iterator, image_to_np

logger = logging.getLogger(__name__)
"""Module-level logger."""


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

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
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

        logger.info("MediaPipe FaceMesh loaded.")

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
            logger.info(f"Downloading MediaPipe model: {model_name}")
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download MediaPipe model: {e}") from e

        return str(model_path)

    @staticmethod
    def _to_numpy_rgb(image) -> np.ndarray:
        """Convert any supported image type to a ``(H, W, 3)`` uint8 RGB numpy array.

        This conversion is applied at the MediaPipe boundary inside
        :meth:`__call__`; callers may pass tensors or paths directly.

        Args:
            image: One of:

                * ``np.ndarray (H, W, 3)`` uint8 RGB
                * ``torch.Tensor (3, H, W)`` or ``(H, W, 3)`` uint8 RGB
                * ``str | Path`` — loaded via :func:`~exordium.video.core.io.image_to_np`

        Returns:
            numpy array of shape ``(H, W, 3)`` uint8 RGB.

        """
        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] == 3:
                return image.permute(1, 2, 0).contiguous().cpu().numpy()
            return image.contiguous().cpu().numpy()
        if isinstance(image, (str, Path)):
            return image_to_np(image, "RGB")
        return image

    def __call__(
        self,
        images: Sequence[np.ndarray | torch.Tensor | str | Path],
    ) -> list[torch.Tensor]:
        """Detect 478 dense facial landmarks for a sequence of face images.

        Accepts numpy arrays, torch tensors, or file paths.  MediaPipe
        processes each image individually; results for images with no
        detected face are silently excluded.

        Args:
            images: Sequence of face images in any of the following formats:

                * ``np.ndarray (H, W, 3)`` uint8 RGB
                * ``torch.Tensor (3, H, W)`` or ``(H, W, 3)`` uint8 RGB
                * ``str | Path`` — image file path

        Returns:
            List of ``(478, 2)`` float32 tensors, one per image with a
            detected face, containing ``(x, y)`` pixel coordinates.

        Example:
            >>> face_crop = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            >>> landmarks_list = wrapper([face_crop])
            >>> if landmarks_list:
            ...     lmks = landmarks_list[0]  # shape (478, 2) torch.Tensor
            ...     print(f"Detected {len(lmks)} landmarks")
        """
        batch = []
        for image in images:
            rgb_image = self._to_numpy_rgb(image)
            h, w = rgb_image.shape[:2]
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_image)
            )
            result = self.landmarker.detect(mp_image)

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                landmarks = [[lm.x * w, lm.y * h] for lm in face_landmarks]
                batch.append(torch.from_numpy(np.array(landmarks, dtype=np.float32)))

        return batch

    @load_or_create("st")
    def track_to_feature(
        self, track: Track, batch_size: int = 30, **_kwargs
    ) -> dict[str, torch.Tensor]:
        """Extract FaceMesh landmarks for all non-interpolated detections in a track.

        Processes detections in batches and caches results using the
        ``load_or_create`` decorator.

        Args:
            track: Track containing a sequence of Detection objects.
            batch_size: Number of detections to process per batch. Default: 30.
            **_kwargs: Additional keyword arguments forwarded to
                ``load_or_create`` (e.g. ``output_path``, ``overwrite``).

        Returns:
            Dict with keys ``"frame_ids"`` (``(N,)`` long tensor) and
            ``"features"`` (``(N, 478, 2)`` float tensor), both on CPU.
        """
        ids: list[int] = []
        features: list[torch.Tensor] = []
        for subset in batch_iterator(track, batch_size):
            valid = [(d.frame_id, d.crop(square=True)) for d in subset]
            if not valid:
                continue
            batch_ids, crops = zip(*valid)
            landmarks = self(list(crops))  # may be shorter than crops if MediaPipe misses a face
            if not landmarks:
                continue
            # keep only the ids that produced a result (MediaPipe silently skips no-face images)
            n_missing = len(batch_ids) - len(landmarks)
            if n_missing:
                logger.debug(
                    "MediaPipe skipped %d image(s) in this batch (no face detected).", n_missing
                )
            ids += list(batch_ids[: len(landmarks)])
            features.append(torch.stack(landmarks))
        return {
            "frame_ids": torch.tensor(ids, dtype=torch.long),
            "features": torch.cat(features, dim=0) if features else torch.zeros((0, 478, 2)),
        }

    def __del__(self) -> None:
        """Clean up landmarker resources."""
        if hasattr(self, "landmarker"):
            self.landmarker.close()


def crop_eye_regions(
    image: np.ndarray | torch.Tensor,
    landmarks: np.ndarray | torch.Tensor,
    scale: float = 1.5,
) -> dict[str, np.ndarray | torch.Tensor]:
    """Crop right and left eye regions from a face image.

    Supports both coarse **YOLO11 5-point** keypoints and dense
    **FaceMesh 478-point** landmarks.  The square crop side length is derived
    from the Euclidean inter-eye distance multiplied by *scale*, so the result
    adapts automatically to any face size.

    Args:
        image: Face image — ``(H, W, 3)`` uint8 numpy **or**
            ``(3, H, W)`` uint8 torch tensor.
        landmarks: Facial landmark coordinates in pixel space:

            * ``(5, 2)``   — YOLO11 coarse keypoints.
              Index ``0`` = right eye (subject's right, viewer's left),
              index ``1`` = left eye.
            * ``(478, 2)`` — FaceMesh dense landmarks.
              Eye centroids are computed as the mean of
              :data:`~exordium.video.face.landmark.constants.FaceMesh478Regions.RIGHT_EYE`
              and
              :data:`~exordium.video.face.landmark.constants.FaceMesh478Regions.LEFT_EYE`
              indices.

            Either ``np.ndarray`` or ``torch.Tensor``.

        scale: Multiplier applied to the inter-eye distance to set the square
            crop side length.  Default ``1.5``.

    Returns:
        Dict with two keys, both in the same format as *image*:

        * ``"eye_region_right"`` — right-eye crop (subject's right, viewer's left).
        * ``"eye_region_left"``  — left-eye crop  (subject's left, viewer's right).

    Raises:
        ValueError: If *landmarks* does not have shape ``(5, 2)`` or ``(478, 2)``.

    Example::

        # --- coarse (YOLO11) ---
        from exordium.video.face.detector.yolo11 import YoloFace11Detector
        from exordium.video.face.landmark.facemesh import crop_eye_regions

        det   = detector.detect_image_path(path)[0]
        crop  = det.crop(square=True, extra_space=1.3)   # (3, H, W) tensor
        eyes  = crop_eye_regions(crop, det.landmarks)    # 5-pt keypoints
        right = eyes["eye_region_right"]                 # (3, bb, bb) tensor

        # --- dense (FaceMesh) ---
        lmks  = facemesh([crop])[0]                      # (478, 2) tensor
        eyes  = crop_eye_regions(crop, lmks)
        right = eyes["eye_region_right"]

    """
    from exordium.video.core.bb import crop_mid
    from exordium.video.face.landmark.constants import FaceMesh478Regions

    lm: np.ndarray = (
        landmarks.numpy() if isinstance(landmarks, torch.Tensor) else np.asarray(landmarks)
    )

    if lm.shape == (5, 2):
        # Coarse YOLO11: index 0 = right eye, index 1 = left eye
        right_center = lm[0]
        left_center = lm[1]
    elif lm.shape == (478, 2):
        # Dense FaceMesh: compute centroid of each eye's landmark cluster
        right_center = lm[FaceMesh478Regions.RIGHT_EYE].mean(axis=0)
        left_center = lm[FaceMesh478Regions.LEFT_EYE].mean(axis=0)
    else:
        raise ValueError(f"landmarks must have shape (5, 2) or (478, 2), got {lm.shape}")

    inter_eye_dist = float(np.linalg.norm(right_center - left_center))
    bb_size = max(1, int(inter_eye_dist * scale))

    return {
        "eye_region_right": crop_mid(image, right_center, bb_size),
        "eye_region_left": crop_mid(image, left_center, bb_size),
    }


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
    image: np.ndarray | torch.Tensor,
    landmarks: np.ndarray | torch.Tensor,
    output_path: str | Path | None = None,
    show_indices: bool = True,
    radius: int = 1,
    color: tuple[int, int, int] = (255, 0, 0),
    colors: list[tuple[int, int, int]] | None = None,
    thickness: int = -1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    fontScale: float = 0.5,
    font_thickness: int = 1,
) -> np.ndarray | torch.Tensor:
    """Draw 2D landmarks onto an image.

    Accepts ``(H, W, C)`` numpy arrays or ``(C, H, W)`` uint8 torch tensors;
    returns the same type as the input.  Colors are interpreted in the
    channel order of the input (RGB for tensors, BGR for numpy).

    Args:
        image: Input image — ``np.ndarray (H, W, C)`` or
            ``torch.Tensor (C, H, W)`` uint8.
        landmarks: Landmark coordinates of shape ``(N, 2)``.
        output_path: Path to save the output image. ``None`` skips saving.
        show_indices: Draw landmark indices next to each point. Default: True.
        radius: Radius of each landmark circle. Default: 1.
        color: Fallback color for landmarks when ``colors`` is not provided.
            Default: ``(255, 0, 0)`` (red in RGB).
        colors: Per-landmark color list of length ``N``.  When provided and
            ``len(colors) == N``, each landmark is drawn in its own color.
            ``None`` falls back to ``color``.
        thickness: Circle border thickness. Default: 2.
        font: OpenCV font type. Default: cv2.FONT_HERSHEY_SIMPLEX.
        fontScale: Font scale for index labels. Default: 0.3.
        font_thickness: Thickness of index label text. Default: 1.

    Returns:
        Copy of the image with landmarks drawn, same type as input.

    Raises:
        Exception: If ``landmarks`` does not have shape ``(N, 2)``.
    """
    if not (landmarks.ndim == 2 and landmarks.shape[1] == 2):
        raise Exception(f"Expected landmarks with shape (N, 2) got instead {landmarks.shape}.")

    tensor_input = isinstance(image, torch.Tensor)
    if tensor_input:
        assert isinstance(image, torch.Tensor)
        img_np: np.ndarray = image.permute(1, 2, 0).contiguous().cpu().numpy()
    else:
        img_np = np.ascontiguousarray(image)

    image_out = img_np.copy()
    lmks_arr = landmarks.numpy() if isinstance(landmarks, torch.Tensor) else landmarks
    lmks = np.rint(lmks_arr).astype(int)

    use_per_landmark_colors = colors is not None and len(colors) == len(lmks)
    for index in range(len(lmks)):
        pt_color = colors[index] if (use_per_landmark_colors and colors is not None) else color
        cv2.circle(image_out, tuple(lmks[index, :]), radius, pt_color, thickness)
        if show_indices:
            cv2.putText(
                image_out,
                str(index),
                tuple(lmks[index, :] + 5),
                font,
                fontScale,
                pt_color,
                font_thickness,
                cv2.LINE_AA,
            )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image_out)

    if tensor_input:
        return torch.from_numpy(image_out).permute(2, 0, 1)
    return image_out
