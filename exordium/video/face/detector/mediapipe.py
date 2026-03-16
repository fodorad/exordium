"""MediaPipe face detector wrapper."""

import logging
import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np

from exordium.video.face.detector.base import FaceDetector


class MediaPipeFaceDetector(FaceDetector):
    """Face detector wrapper using MediaPipe BlazeFace.

    Fast CPU-friendly face detection using MediaPipe's optimised BlazeFace
    model.  Provides bounding boxes, five keypoints (right eye, left eye,
    nose, mouth right, mouth left), and confidence scores, matching the
    ``Detection`` dataclass landmark convention.

    Args:
        batch_size: Number of images to process per progress-bar batch.
            MediaPipe processes images one at a time; this only controls
            the tqdm grouping. Default: 16.
        min_detection_confidence: Minimum confidence threshold. Default: 0.5.
        model_selection: 0 for short-range model (≤2 m), 1 for full-range
            model (≤5 m). Default: 0.
        verbose: Show progress bars. Default: False.

    """

    def __init__(
        self,
        batch_size: int = 16,
        min_detection_confidence: float = 0.5,
        model_selection: int = 0,
        verbose: bool = False,
    ):
        super().__init__(batch_size=batch_size, verbose=verbose)
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection

        model_path = self._download_model()

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence,
        )
        self.detector = vision.FaceDetector.create_from_options(options)

        logging.info(f"MediaPipe FaceDetector loaded (model_selection={model_selection}).")

    def _download_model(self) -> str:
        """Download MediaPipe face detection model if not present."""
        model_dir = Path.home() / ".cache" / "mediapipe_models"
        model_dir.mkdir(parents=True, exist_ok=True)

        if self.model_selection == 0:
            model_name = "blaze_face_short_range.tflite"
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        else:
            model_name = "blaze_face_full_range.tflite"
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite"

        model_path = model_dir / model_name

        if not model_path.exists():
            logging.info(f"Downloading MediaPipe model: {model_name}")
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download MediaPipe model: {e}") from e

        return str(model_path)

    def run_detector(
        self, images_rgb: list[np.ndarray]
    ) -> list[list[tuple[np.ndarray, np.ndarray, float]]]:
        """Run MediaPipe face detector on a list of RGB images.

        Args:
            images_rgb: List of input images of shape (H, W, 3) in RGB.

        Returns:
            List of detections per image.  Each detection is a tuple of:
            ``(bb_xyxy, landmarks, score)`` where ``bb_xyxy`` is shape
            ``(4,)``, ``landmarks`` is shape ``(5, 2)`` in the order
            ``(right_eye, left_eye, nose, mouth_right, mouth_left)``,
            and ``score`` is the detection confidence.

        """
        all_detections = []

        for image_rgb in images_rgb:
            h, w = image_rgb.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = self.detector.detect(mp_image)

            frame_detections = []
            if result.detections:
                for detection in result.detections:
                    bbox = detection.bounding_box
                    x_min = bbox.origin_x
                    y_min = bbox.origin_y
                    x_max = x_min + bbox.width
                    y_max = y_min + bbox.height
                    bb_xyxy = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

                    # MediaPipe provides 6 keypoints:
                    # 0: right eye, 1: left eye, 2: nose tip, 3: mouth center,
                    # 4: right ear tragion, 5: left ear tragion
                    # Map to Detection format (5 pts): right_eye, left_eye, nose,
                    # mouth_right, mouth_left — use mouth center for both corners.
                    kpts = []
                    if detection.keypoints:
                        for kp in detection.keypoints:
                            kpts.append([kp.x * w, kp.y * h])
                    kpts = np.array(kpts, dtype=np.float32)

                    if len(kpts) >= 4:
                        landmarks = np.array(
                            [kpts[0], kpts[1], kpts[2], kpts[3], kpts[3]],
                            dtype=np.float32,
                        )
                    else:
                        landmarks = np.zeros((5, 2), dtype=np.float32)

                    score = detection.categories[0].score if detection.categories else 0.0
                    frame_detections.append((bb_xyxy, landmarks, score))

            all_detections.append(frame_detections)

        return all_detections

    def __del__(self) -> None:
        """Clean up detector resources.

        Closes the MediaPipe detector when object is deleted.
        """
        if hasattr(self, "detector"):
            self.detector.close()
