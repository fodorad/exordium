"""Base class for face detectors."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from exordium.utils.decorator import load_or_create
from exordium.video.core.bb import xyxy2xywh
from exordium.video.core.detection import FrameDetections, VideoDetections
from exordium.video.core.io import Video, image_to_tensor

logger = logging.getLogger(__name__)
"""Module-level logger."""


class FaceDetector(ABC):
    """Abstract base class for face detectors.

    Coordinate convention:
        y is row (height), x is column (width)::

            (0,0)---(0,w)
              |       |
            (h,0)---(h,w)

        Bounding box: ``(x_min, y_min, x_max, y_max)``

    All public methods accept images as:

    * ``np.ndarray`` ``(H, W, 3)`` uint8 RGB
    * ``torch.Tensor`` ``(3, H, W)`` uint8 RGB
    * ``str | Path`` — loaded via :func:`~exordium.video.core.io.image_to_tensor`

    Subclasses must implement :meth:`run_detector`, which receives
    ``list[torch.Tensor (3, H, W) uint8 RGB]`` and returns detections in the
    standard format with torch tensor bounding boxes and landmarks.

    """

    def __init__(self, batch_size: int = 32, verbose: bool = False):
        self.batch_size = batch_size
        self.verbose = verbose

    @staticmethod
    def _build_frame_detections(
        frame_det: list[tuple[torch.Tensor, torch.Tensor, float]],
        frame_id: int,
        source: str | torch.Tensor | np.ndarray,
    ) -> FrameDetections:
        """Build a :class:`~exordium.video.core.detection.FrameDetections` from raw detections.

        Args:
            frame_det: List of ``(bb_xyxy, landmarks, score)`` tuples.  Both
                ``bb_xyxy`` and ``landmarks`` are ``torch.Tensor``.
            frame_id: Frame index.
            source: Source identifier — a path string or an in-memory
                ``torch.Tensor (3, H, W)`` uint8 RGB frame.

        Returns:
            Populated :class:`~exordium.video.core.detection.FrameDetections`.

        """
        fd = FrameDetections()
        for bb_xyxy, landmarks, score in frame_det:
            fd.add_dict(
                {
                    "frame_id": frame_id,
                    "source": source,
                    "score": score,
                    "bb_xywh": xyxy2xywh(bb_xyxy.long()),
                    "landmarks": landmarks.round().long(),
                }
            )
        return fd

    @abstractmethod
    def run_detector(
        self, images: list[torch.Tensor]
    ) -> list[list[tuple[torch.Tensor, torch.Tensor, float]]]:
        """Run the face detector on a list of ``(3, H, W)`` uint8 RGB tensors.

        Subclasses receive tensors directly and are responsible for any
        conversion to numpy or other formats required by the underlying model.

        Args:
            images: List of ``(3, H, W)`` uint8 RGB ``torch.Tensor`` objects.

        Returns:
            One inner list per input image.  Each detection is a tuple
            ``(bb_xyxy, landmarks, score)`` where:

            * ``bb_xyxy``   — ``torch.Tensor float32 (4,)`` ``[x_min, y_min, x_max, y_max]``
            * ``landmarks`` — ``torch.Tensor float32 (5, 2)`` ``(x, y)`` pixel coords
            * ``score``     — confidence in ``[0, 1]``

            Empty inner list when no face is detected in that image.

        """

    @load_or_create("fdet")
    def detect_image_path(self, image_path: str | Path, **kwargs) -> FrameDetections:
        """Run detector on a single image given by its path.

        Args:
            image_path: Path to the image file.
            **kwargs: Forwarded to :meth:`detect_image` (e.g. ``output_path``).

        Returns:
            :class:`~exordium.video.core.detection.FrameDetections` with all
            faces found in the image.

        """
        image_tensor = image_to_tensor(image_path, channel_order="RGB")
        return self.detect_image(image_tensor, image_path=str(image_path), **kwargs)

    @load_or_create("fdet")
    def detect_image(
        self,
        image: np.ndarray | torch.Tensor,
        **kwargs,
    ) -> FrameDetections:
        """Run detector on a single image.

        Args:
            image: Input image as ``np.ndarray (H, W, 3)`` uint8 RGB or
                ``torch.Tensor (3, H, W)`` uint8 RGB.
            **kwargs: Optional keyword arguments, including ``image_path``
                used as the cache key.

        Returns:
            :class:`~exordium.video.core.detection.FrameDetections` with all
            faces found in the image.

        """
        source: str | np.ndarray | torch.Tensor = (
            str(kwargs["image_path"]) if "image_path" in kwargs else image
        )
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        frame_dets = self.run_detector([image])
        return self._build_frame_detections(
            frame_dets[0],
            frame_id=-1,
            source=source,
        )

    @load_or_create("vdet")
    def detect_frame_dir(self, frame_dir: str | Path, **_kwargs) -> VideoDetections:
        """Run detector on ordered frames of a video in a single folder.

        Args:
            frame_dir: Path to the frame directory.

        Returns:
            :class:`~exordium.video.core.detection.VideoDetections` with all
            detections.

        """
        frame_paths = sorted(Path(frame_dir).iterdir())
        video_detections = VideoDetections()

        for batch_ind in tqdm(
            range(0, len(frame_paths), self.batch_size),
            desc="Face detection",
            disable=not self.verbose,
        ):
            batch_frame_paths = frame_paths[batch_ind : batch_ind + self.batch_size]
            images = [image_to_tensor(p, "RGB") for p in batch_frame_paths]
            frame_dets = self.run_detector(images)

            for frame_det, frame_path in zip(frame_dets, batch_frame_paths):
                video_detections.add(
                    self._build_frame_detections(
                        frame_det,
                        frame_id=int(Path(frame_path).stem),
                        source=str(frame_path),
                    )
                )

        return video_detections

    @load_or_create("vdet")
    def detect_video(self, video_path: str | Path, **_kwargs) -> VideoDetections:
        """Run detector on a video given by its path.

        Args:
            video_path: Path to the video file.

        Returns:
            :class:`~exordium.video.core.detection.VideoDetections` with all
            detections.

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
                frame_dets = self.run_detector(list(batch))

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
