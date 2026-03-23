"""YOLO11 face detector wrapper — bounding-box + 5-point facial keypoints."""

import logging
import urllib.request

import torch

from exordium.video.face.detector.base import FaceDetector

logger = logging.getLogger(__name__)
"""Module-level logger."""

# GitHub release base URL for zjykzj/YOLO11Face weights.
_RELEASE_BASE = "https://github.com/zjykzj/YOLO11Face/releases/download/v1.0.0"

# Available pre-trained weights.  All are Ultralytics YOLO11-pose models
# trained on WIDERFace with RetinaFace 5-point keypoint annotations.
YOLO11_FACE_MODELS: dict[str, str] = {
    "yolo11n-pose_widerface": f"{_RELEASE_BASE}/yolo11n-pose_widerface.pt",
    "yolo11s-pose_widerface": f"{_RELEASE_BASE}/yolo11s-pose_widerface.pt",
    "yolov8n-pose_widerface": f"{_RELEASE_BASE}/yolov8n-pose_widerface.pt",
    "yolov8s-pose_widerface": f"{_RELEASE_BASE}/yolov8s-pose_widerface.pt",
}
"""Mapping of model names to download URLs for available YOLO11-face weights."""

_DEFAULT_MODEL = "yolo11n-pose_widerface"


class YoloFace11Detector(FaceDetector):
    """Face detector using YOLO11-pose — **bounding-box + 5 facial keypoints**.

    Uses models from `zjykzj/YOLO11Face
    <https://github.com/zjykzj/YOLO11Face>`_ trained on WIDERFace with
    RetinaFace keypoint annotations.  Weights are downloaded automatically
    on first use and cached under :data:`~exordium.WEIGHT_DIR`.

    Available models (``model_name`` values):

    * ``"yolo11n-pose_widerface"`` — nano, fastest (default)
    * ``"yolo11s-pose_widerface"`` — small, best accuracy
    * ``"yolov8n-pose_widerface"`` — YOLOv8-nano backbone
    * ``"yolov8s-pose_widerface"`` — YOLOv8-small backbone

    Keypoint order (5 points, pixel ``(x, y)``):
    right eye, left eye, nose tip, mouth right, mouth left.

    Accepts ``(3, H, W)`` uint8 RGB ``torch.Tensor`` objects throughout.
    Numpy conversion to BGR is performed internally inside
    :meth:`run_detector` where Ultralytics requires it.

    Args:
        model_name: One of the keys in :data:`YOLO11_FACE_MODELS`, or a local
            path to a ``.pt`` file.
        device_id: Device index.  ``-1`` / ``None`` → CPU; ``0+`` →
            MPS / CUDA.
        conf: Minimum detection confidence threshold.
        batch_size: Frames per inference batch.
        verbose: Show tqdm progress bars.

    Example::

        detector = YoloFace11Detector(device_id=None)

        # single image (tensor input)
        dets = detector.detect_image(rgb_tensor)  # (3, H, W) uint8 RGB
        for det in dets:
            print(det.bb_xyxy, det.landmarks)  # landmarks (5, 2) non-zero

        # full video
        vdets = detector.detect_video("input.mp4", output_path="dets.vdet")

    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device_id: int = 0,
        conf: float = 0.25,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        super().__init__(batch_size=batch_size, verbose=verbose)
        from ultralytics import YOLO

        from exordium.utils.device import get_torch_device

        self.conf = conf
        self.device = get_torch_device(device_id)

        model_path = self._resolve_model(model_name)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        logger.info(f"YoloFace11Detector loaded '{model_name}' on {self.device} (conf={conf}).")

    @staticmethod
    def _resolve_model(model_name: str) -> str:
        """Return a local path to the ``.pt`` file, downloading if needed.

        If ``model_name`` is a key in :data:`YOLO11_FACE_MODELS` the
        corresponding weight file is downloaded from GitHub releases and
        cached under :data:`~exordium.WEIGHT_DIR`.  Otherwise ``model_name``
        is treated as a local path and returned unchanged.

        Args:
            model_name: Key in :data:`YOLO11_FACE_MODELS` or a local ``.pt``
                path.

        Returns:
            Absolute path string to a local ``.pt`` file.

        """
        if model_name in YOLO11_FACE_MODELS:
            from exordium import WEIGHT_DIR

            WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
            local_path = WEIGHT_DIR / f"{model_name}.pt"
            if not local_path.exists():
                url = YOLO11_FACE_MODELS[model_name]
                logger.info(f"Downloading {model_name} from {url} ...")
                urllib.request.urlretrieve(url, local_path)
                logger.info(f"Saved to {local_path}")
            return str(local_path)
        return model_name

    @staticmethod
    def _parse_results(
        results,
    ) -> list[list[tuple[torch.Tensor, torch.Tensor, float]]]:
        """Parse Ultralytics ``Results`` keeping outputs as torch tensors.

        Args:
            results: Sequence of ``ultralytics.engine.results.Results`` objects.

        Returns:
            One inner list per image.  Each detection is a tuple
            ``(bb_xyxy, landmarks, score)`` where:

            * ``bb_xyxy``   — ``torch.Tensor float32 (4,)``
            * ``landmarks`` — ``torch.Tensor float32 (5, 2)`` xy pixel coords
            * ``score``     — ``float`` confidence

        """
        all_detections = []
        for r in results:
            frame_detections = []
            if r.boxes is not None and len(r.boxes):
                boxes = r.boxes.xyxy.cpu().float()
                scores = r.boxes.conf.cpu()
                kpts = (
                    r.keypoints.xy.cpu().float()
                    if r.keypoints is not None
                    else torch.zeros(len(boxes), 5, 2, dtype=torch.float32)
                )
                for bb, score, lm in zip(boxes, scores, kpts):
                    frame_detections.append((bb, lm, float(score)))
            all_detections.append(frame_detections)
        return all_detections

    def run_detector(
        self, images: list[torch.Tensor]
    ) -> list[list[tuple[torch.Tensor, torch.Tensor, float]]]:
        """Run YOLO11-pose on a list of ``(3, H, W)`` uint8 RGB tensors.

        Converts each tensor to a ``(H, W, 3)`` uint8 **BGR** numpy array
        internally (Ultralytics numpy convention), runs batched inference,
        and returns detections and 5-point face keypoints as torch tensors.

        Args:
            images: List of ``(3, H, W)`` uint8 RGB ``torch.Tensor`` objects.

        Returns:
            One inner list per image.  Each detection is a tuple
            ``(bb_xyxy, landmarks, score)`` — see :meth:`_parse_results`.

        """
        # .contiguous() ensures C-contiguous memory before numpy(); without it,
        # a CHW-contiguous tensor permuted to HWC is non-contiguous and Ultralytics
        # may read incorrect data.  [:, :, ::-1] flips RGB → BGR as Ultralytics expects.
        images_bgr = [img.permute(1, 2, 0).contiguous().cpu().numpy()[:, :, ::-1] for img in images]
        results = self.model.predict(images_bgr, conf=self.conf, verbose=False)
        return self._parse_results(results)
