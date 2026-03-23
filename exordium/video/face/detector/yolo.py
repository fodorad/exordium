"""YOLOv8 face detector wrapper (bounding-box detection only)."""

import logging
from pathlib import Path

import torch
from tqdm import tqdm

from exordium.utils.decorator import load_or_create
from exordium.utils.device import get_torch_device
from exordium.video.core.detection import VideoDetections
from exordium.video.core.io import Video
from exordium.video.face.detector.base import FaceDetector

logger = logging.getLogger(__name__)
"""Module-level logger."""

_DEFAULT_MODEL = "arnabdhar/YOLOv8-Face-Detection"


class YoloFaceV8Detector(FaceDetector):
    """Face detector using YOLOv8 — **bounding-box detection only**.

    Uses ``arnabdhar/YOLOv8-Face-Detection`` (``task=detect``) by default.
    This model outputs only bounding boxes and confidence scores; it has no
    keypoint head.  Facial landmark fields in the returned
    :class:`~exordium.video.core.detection.Detection` objects will always be
    zero tensors.  Use :class:`YoloFace11Detector` for 5-point facial landmarks.

    Accepts ``(3, H, W)`` uint8 RGB ``torch.Tensor`` objects throughout.
    Numpy conversion to BGR is performed internally inside
    :meth:`run_detector` where Ultralytics requires it.

    Args:
        model_name: HuggingFace model ID (``"owner/repo"``) or local path to
            a YOLOv8 ``.pt`` file.  Hub models are downloaded once and cached
            under :data:`~exordium.WEIGHT_DIR`.
        device_id: Device index.  ``-1`` / ``None`` → CPU; ``0+`` →
            MPS / CUDA.
        conf: Minimum detection confidence threshold.
        batch_size: Frames per inference batch.
        verbose: Show tqdm progress bars.

    Example::

        detector = YoloFaceV8Detector(device_id=None)

        # single image (tensor input)
        dets = detector.detect_image(rgb_tensor)  # (3, H, W) uint8 RGB

        # full video
        vdets = detector.detect_video("input.mp4", output_path="dets.vdet")

    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device_id: int = 0,
        conf: float = 0.5,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> None:
        super().__init__(batch_size=batch_size, verbose=verbose)
        from ultralytics import YOLO

        self.conf = conf
        self.device = get_torch_device(device_id)

        model_path = self._resolve_model(model_name)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        logger.info(f"YoloFaceV8Detector loaded on {self.device} (conf={conf}).")

    @staticmethod
    def _resolve_model(model_name: str) -> str:
        """Return a local path to the YOLO ``.pt`` file.

        If ``model_name`` looks like a HuggingFace Hub repo ID
        (``"owner/repo"``), the ``model.pt`` file is downloaded via
        :func:`huggingface_hub.hf_hub_download` and the cached path is
        returned.  Otherwise ``model_name`` is treated as a local path and
        returned unchanged.

        Args:
            model_name: HuggingFace Hub repo ID (``"owner/repo"``) or a
                local path to a ``.pt`` file.

        Returns:
            Absolute path string to a local ``.pt`` file.

        """
        if "/" in model_name and not Path(model_name).exists():
            import shutil

            from huggingface_hub import hf_hub_download

            from exordium import WEIGHT_DIR

            WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
            repo_slug = model_name.replace("/", "--")
            local_path = WEIGHT_DIR / f"{repo_slug}.pt"
            if not local_path.exists():
                cached = hf_hub_download(repo_id=model_name, filename="model.pt")
                shutil.copy2(cached, local_path)
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

            * ``bb_xyxy``   — ``torch.Tensor float32 (4,)`` ``[x_min, y_min, x_max, y_max]``
            * ``landmarks`` — ``torch.Tensor float32 (5, 2)`` xy pixel coords
            * ``score``     — ``float`` confidence

        """
        all_detections = []
        for r in results:
            frame_detections = []
            if r.boxes is not None and len(r.boxes):
                boxes = r.boxes.xyxy.cpu().float()  # (N, 4) tensor
                scores = r.boxes.conf.cpu()  # (N,) tensor
                kpts = (
                    r.keypoints.xy.cpu().float()  # (N, 5, 2) tensor
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
        """Run YOLO on a list of ``(3, H, W)`` uint8 RGB tensors.

        Converts each tensor to a ``(H, W, 3)`` uint8 **BGR** numpy array
        internally (Ultralytics numpy convention), runs batched inference,
        and returns detections as torch tensors.

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

    @load_or_create("vdet")
    def detect_video(self, video_path: str | Path, **_kwargs) -> VideoDetections:
        """Run detector on a video — fully tensor-native, no numpy conversion on hot path.

        Frames are decoded by :class:`~exordium.video.core.io.Video` as
        ``(T, 3, H, W)`` uint8 **RGB** tensors.  Each batch is normalised to
        ``float32 [0, 1]`` and fed directly to YOLO.  Bounding boxes and
        keypoints are kept as tensors throughout.

        Input format to YOLO: ``(B, 3, H, W)`` float32 **RGB** ``[0, 1]``.

        Args:
            video_path: Path to the video file.
            **_kwargs: Forwarded to ``load_or_create`` (e.g. ``output_path``,
                ``overwrite``).

        Returns:
            :class:`~exordium.video.core.detection.VideoDetections` with all
            detections.

        Raises:
            FileNotFoundError: If ``video_path`` does not exist.

        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video does not exist at {video_path}")

        video_detections = VideoDetections()
        frame_id = 0

        with Video(video_path) as video:
            for batch in tqdm(
                video.iter_batches(batch_size=self.batch_size),
                total=(video.num_frames + self.batch_size - 1) // self.batch_size,
                desc="Face detection",
                disable=not self.verbose,
            ):
                # batch: (T, 3, H, W) uint8 RGB tensor from torchcodec
                # YOLO tensor API: (B, 3, H, W) float32 RGB [0, 1]
                batch_float = batch.float().div(255)
                results = self.model.predict(batch_float, conf=self.conf, verbose=False)
                frame_dets = self._parse_results(results)

                for frame_det in frame_dets:
                    video_detections.add(
                        self._build_frame_detections(
                            frame_det,
                            frame_id=frame_id,
                            source=str(video_path),
                        )
                    )
                    frame_id += 1

        return video_detections
