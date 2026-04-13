"""Detection and tracking data structures.

All ``Detection.frame()`` methods return a ``torch.Tensor (3, H, W)`` uint8
**RGB** tensor (channel-first).  Visualization helpers accept both tensors and
numpy arrays and convert to BGR numpy internally before writing with cv2.

Bounding-box fields (``bb_xywh``, ``landmarks``) are always
``torch.Tensor``; format-conversion functions in
:mod:`exordium.video.core.bb` handle them transparently.
"""

from __future__ import annotations

import csv
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from exordium.video.core.bb import crop_mid, crop_xyxy, iou_xywh, xywh2midwh, xywh2xyxy
from exordium.video.core.io import image_to_tensor, load_frames

logger = logging.getLogger(__name__)
"""Module-level logger."""


def _to_list(arr: torch.Tensor) -> list:
    """Flatten a tensor to a Python list for CSV serialisation."""
    return arr.cpu().reshape(-1).tolist()


def _arr_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Element-wise equality check for torch tensors."""
    return bool(torch.all(a == b).item())


def _to_bgr_numpy(image: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert any supported frame representation to a BGR numpy array.

    Args:
        image: Either a ``(C, H, W)`` uint8 RGB torch tensor or a
            ``(H, W, C)`` BGR numpy array (cv2 convention).

    Returns:
        ``(H, W, C)`` uint8 BGR numpy array.

    """
    if isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0).cpu().numpy()
        else:
            image = image.cpu().numpy()
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image  # assumed BGR numpy already


# ---------------------------------------------------------------------------
# Detection base class
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Detection(ABC):
    """Abstract base class for a single face detection."""

    frame_id: int
    """Zero-based index of the frame within its source."""
    source: str | Path | torch.Tensor
    """Origin of the detection — a file path, an in-memory numpy array, or a torch tensor."""
    score: float
    """Detector confidence score in ``[0, 1]``."""
    bb_xywh: torch.Tensor  # (4,) xywh pixel coords
    """Bounding box ``[x_min, y_min, width, height]`` as a ``(4,)`` long tensor."""
    landmarks: torch.Tensor  # (5, 2) xy pixel coords
    """Five facial keypoints as a ``(5, 2)`` long tensor of ``(x, y)`` pixel coords."""

    @property
    def bb_xyxy(self) -> torch.Tensor:
        """Bounding box in xyxy format.  Type mirrors ``bb_xywh``."""
        return cast("torch.Tensor", xywh2xyxy(self.bb_xywh))

    @property
    def bb_center(self) -> torch.Tensor:
        """Bounding box center ``[x, y]``.  Type mirrors ``bb_xywh``."""
        return cast("torch.Tensor", xywh2midwh(self.bb_xywh))[:2]

    def _crop_origin(
        self,
        square: bool,
        extra_space: float,
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int]:
        """Compute the top-left corner of :meth:`crop` in full-image pixel coordinates.

        Mirrors the geometry of :meth:`crop` exactly — including boundary
        clamping and integer rounding — so that :meth:`crop_landmarks` can
        project coordinates into crop-local space without re-loading the frame.

        For the square path this mirrors :func:`~exordium.video.core.bb.crop_mid`
        (``half = size // 2``).  For the non-square path it mirrors
        :func:`~exordium.video.core.bb.crop_xyxy` (``int()`` truncation).

        Args:
            square: If ``True``, compute a square region of side
                ``max(w, h) * extra_space``.  If ``False``, use the original
                bounding-box dimensions scaled by ``extra_space``.
            extra_space: Scale multiplier applied to the crop side(s).
            frame_w: Source-frame width in pixels.
            frame_h: Source-frame height in pixels.

        Returns:
            ``(x1, y1)`` top-left corner of the crop in full-image pixel space.

        """
        if square:
            mid = xywh2midwh(self.bb_xywh)[:2]
            cx, cy = int(mid[0]), int(mid[1])
            size = int(max(float(self.bb_xywh[2]), float(self.bb_xywh[3])) * extra_space)
            half = size // 2
            x1 = max(0, min(cx - half, frame_w))
            y1 = max(0, min(cy - half, frame_h))
        else:
            # Mirror crop_xyxy: uses bb_xyxy[0], bb_xyxy[1] directly
            x1 = max(0, min(int(self.bb_xyxy[0]), frame_w))
            y1 = max(0, min(int(self.bb_xyxy[1]), frame_h))
        return x1, y1

    def crop(self, square: bool = False, extra_space: float = 1.0) -> torch.Tensor:
        """Crop a region around this detection from the source frame.

        For the square path delegates to
        :func:`~exordium.video.core.bb.crop_mid` which computes a square
        region centered on the bounding-box midpoint.  For the non-square path
        delegates to :func:`~exordium.video.core.bb.crop_xyxy` which clips a
        rectangular region to image bounds.

        Returns a ``(3, H', W')`` uint8 RGB tensor.

        Args:
            square: If ``False`` (default), crop the detected bounding box
                exactly using :attr:`bb_xyxy`.  If ``True``, compute the
                bounding-box center and use ``max(w, h) * extra_space`` as
                the side length to produce a square crop.
            extra_space: Scale multiplier for the square side length.  Only
                applies when ``square=True``; ignored for rectangular crops.
                ``1.0`` (default) = tight square, no padding.

        Returns:
            ``(3, H', W')`` uint8 RGB torch tensor.

        Examples::

            crop = det.crop()                              # exact BB
            crop = det.crop(square=True)                   # square, tight
            crop = det.crop(square=True, extra_space=1.5)  # square + 50 % pad

        """
        f = self.frame()
        if square:
            mid = xywh2midwh(self.bb_xywh)[:2]
            size = int(max(float(self.bb_xywh[2]), float(self.bb_xywh[3])) * extra_space)
            return cast("torch.Tensor", crop_mid(f, mid, size))
        else:
            return cast("torch.Tensor", crop_xyxy(f, self.bb_xyxy))

    def crop_landmarks(
        self,
        square: bool = False,
        extra_space: float = 1.0,
    ) -> torch.Tensor:
        """Landmark coordinates projected into the :meth:`crop` coordinate space.

        Returns the keypoints expressed relative to the top-left corner of the
        region that :meth:`crop` would extract with the same arguments.  Use
        these coordinates to overlay landmarks on the cropped image without
        access to the original frame.

        The parameters must match the :meth:`crop` call you intend to visualize::

            crop_img = det.crop(square=True, extra_space=1.2)
            lm       = det.crop_landmarks(square=True, extra_space=1.2)
            # lm is correctly positioned inside crop_img

        Args:
            square: Must match the ``square`` argument passed to :meth:`crop`.
                Default: ``False``.
            extra_space: Must match the ``extra_space`` argument passed to
                :meth:`crop`.  Default: ``1.0``.

        Returns:
            ``(N, 2)`` ``torch.long`` tensor of ``(x, y)`` coordinates in
            crop-local pixel space, where ``N`` is the number of landmarks
            stored in :attr:`landmarks``.

        """
        _, h, w = self.frame().shape
        x1, y1 = self._crop_origin(square, extra_space, w, h)
        result = self.landmarks.clone()
        result[:, 0] -= x1
        result[:, 1] -= y1
        return result

    @classmethod
    def save_header(cls) -> list[str]:
        """CSV column names for :meth:`save_record`."""
        return [
            "frame_id",
            "source",
            "score",
            "x",
            "y",
            "w",
            "h",
            "left_eye_x",
            "left_eye_y",
            "right_eye_x",
            "right_eye_y",
            "nose_x",
            "nose_y",
            "left_mouth_x",
            "left_mouth_y",
            "right_mouth_x",
            "right_mouth_y",
        ]

    def save_record(self) -> list[str]:
        """Serialise to a flat list of strings for CSV export.

        Only detections whose ``source`` is a file path (``str``) can be
        round-tripped through CSV; in-memory sources are stored as ``None``.
        """
        source_record = self.source if isinstance(self.source, str) else None
        return list(
            map(
                str,
                [
                    self.frame_id,
                    source_record,
                    self.score,
                    *_to_list(self.bb_xywh),
                    *_to_list(self.landmarks),
                ],
            )
        )

    @abstractmethod
    def frame(self) -> torch.Tensor:
        """Load the source frame.

        Returns:
            ``(3, H, W)`` uint8 **RGB** torch tensor (channel-first).

        """

    @abstractmethod
    def frame_center(self) -> torch.Tensor:
        """Centre of the source frame in pixel coordinates.

        Returns:
            ``(2,)`` long tensor ``[x, y]``.

        """

    def __eq__(self, other) -> bool:
        """Return True if two detections are equal."""
        if not isinstance(other, Detection):
            return False
        return bool(
            self.frame_id == other.frame_id
            and self.source == other.source
            and abs(self.score - other.score) < 1e-6
            and _arr_equal(self.bb_xyxy, other.bb_xyxy)
            and _arr_equal(self.bb_xywh, other.bb_xywh)
            and _arr_equal(self.landmarks, other.landmarks)
        )


# ---------------------------------------------------------------------------
# Concrete Detection subclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True, eq=False)
class DetectionFromImage(Detection):
    """Detection from a single image file on disk.

    ``source`` is the path to the image.  ``frame()`` loads and returns it as
    a ``(3, H, W)`` uint8 RGB tensor on the fly using
    :func:`~exordium.video.core.io.image_to_tensor`.
    """

    source: str | Path
    """Path to the image file on disk."""

    def frame(self) -> torch.Tensor:
        """Load the image file as a ``(3, H, W)`` uint8 RGB tensor."""
        return image_to_tensor(self.source, channel_order="RGB")

    def frame_center(self) -> torch.Tensor:
        """Frame centre as ``(2,)`` long tensor ``[x, y]``."""
        _, h, w = self.frame().shape
        return torch.tensor([w // 2, h // 2], dtype=torch.long)


@dataclass(frozen=True, kw_only=True, eq=False)
class DetectionFromVideo(Detection):
    """Detection from a video file on disk.

    ``source`` is the path to the video.  ``frame()`` loads the single frame
    at ``frame_id`` on the fly using
    :func:`~exordium.video.core.io.load_frames` (CPU decode, no GPU
    round-trip) and returns it as a ``(3, H, W)`` uint8 RGB tensor.
    """

    source: str | Path
    """Path to the video file on disk."""

    def frame(self) -> torch.Tensor:
        """Load ``frame_id`` from the video as a ``(3, H, W)`` uint8 RGB tensor.

        Uses CPU-side torchcodec decoding; no numpy conversion.
        """
        frames = load_frames(input_path=self.source, frame_ids=[self.frame_id], device_id=None)
        return frames[0]  # (3, H, W) uint8 RGB

    def frame_center(self) -> torch.Tensor:
        """Frame centre as ``(2,)`` long tensor ``[x, y]``."""
        _, h, w = self.frame().shape
        return torch.tensor([w // 2, h // 2], dtype=torch.long)


@dataclass(frozen=True, kw_only=True, eq=False)
class DetectionFromNp(Detection):
    """Detection from an in-memory numpy frame array.

    ``source`` is an ``(H, W, C)`` uint8 **RGB** numpy array.
    ``frame()`` converts it to a ``(3, H, W)`` uint8 RGB tensor without a
    data copy (uses :func:`torch.from_numpy` + permute).
    """

    source: np.ndarray
    """In-memory ``(H, W, C)`` uint8 RGB numpy array."""

    def frame(self) -> torch.Tensor:
        """Return the in-memory frame as a ``(3, H, W)`` uint8 RGB tensor."""
        return torch.from_numpy(np.ascontiguousarray(self.source)).permute(2, 0, 1)

    def frame_center(self) -> torch.Tensor:
        """Frame centre as ``(2,)`` long tensor ``[x, y]``."""
        h, w = self.source.shape[:2]
        return torch.tensor([w // 2, h // 2], dtype=torch.long)


@dataclass(frozen=True, kw_only=True, eq=False)
class DetectionFromTorchTensor(Detection):
    """Detection from an in-memory torch tensor frame.

    ``source`` is a ``(3, H, W)`` uint8 **RGB** torch tensor (channel-first).
    This is the preferred subclass for the fully tensor-native detection
    pipeline (e.g. YOLO on GPU).
    """

    source: torch.Tensor
    """In-memory ``(3, H, W)`` uint8 RGB torch tensor."""

    def frame(self) -> torch.Tensor:
        """Return the in-memory ``(3, H, W)`` uint8 RGB tensor directly."""
        return self.source

    def frame_center(self) -> torch.Tensor:
        """Frame centre as ``(2,)`` long tensor ``[x, y]``."""
        _, h, w = self.source.shape
        return torch.tensor([w // 2, h // 2], dtype=torch.long)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class DetectionFactory:
    """Factory for creating the appropriate :class:`Detection` subclass.

    Routes creation based on the type of ``source``:

    * ``torch.Tensor``         → :class:`DetectionFromTorchTensor`
    * ``np.ndarray``           → :class:`DetectionFromNp`
    * path with video suffix   → :class:`DetectionFromVideo`
    * path with image suffix   → :class:`DetectionFromImage`

    """

    VIDEO_EXTENSIONS = {".mp4", ".mpeg", ".mov", ".mkv", ".avi"}
    """Supported video file extensions for routing to :class:`DetectionFromVideo`."""
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
    """Supported image file extensions for routing to :class:`DetectionFromImage`."""

    @classmethod
    def create_detection(cls, **kwargs) -> Detection:
        """Create a Detection instance from keyword arguments.

        Args:
            **kwargs: Must include all fields of :class:`Detection`.

        Returns:
            Appropriate :class:`Detection` subclass instance.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If the file extension is not supported or the source
                type is unrecognised.

        """
        missing = set(Detection.__dataclass_fields__.keys()) - set(kwargs.keys())
        if missing:
            raise KeyError(f"Invalid Detection dictionary. Missing keys: {missing}")

        source = kwargs.get("source")

        if isinstance(source, torch.Tensor):
            return DetectionFromTorchTensor(**kwargs)

        if isinstance(source, np.ndarray):
            return DetectionFromNp(**kwargs)

        if isinstance(source, (str, Path)):
            ext = Path(source).suffix.lower()
            if ext in cls.VIDEO_EXTENSIONS:
                return DetectionFromVideo(**kwargs)
            if ext in cls.IMAGE_EXTENSIONS:
                return DetectionFromImage(**kwargs)
            raise ValueError(f"Unsupported file extension: '{ext}' for source '{source}'.")

        raise ValueError(f"Unsupported source type: {type(source).__name__}.")


# ---------------------------------------------------------------------------
# FrameDetections
# ---------------------------------------------------------------------------


class FrameDetections:
    """Face detections for a single video frame."""

    def __init__(self):
        self.detections: list[Detection] = []

    @property
    def frame_id(self) -> int:
        """Frame index of all contained detections."""
        return self.detections[0].frame_id

    @property
    def source(self) -> str | Path | np.ndarray | torch.Tensor:
        """Source of the contained detections."""
        return self.detections[0].source

    def add_dict(self, detection: dict) -> FrameDetections:
        """Create and append a :class:`Detection` from a keyword dict."""
        self.detections.append(DetectionFactory.create_detection(**detection))
        return self

    def add(self, detection: Detection) -> FrameDetections:
        """Append an existing :class:`Detection` instance."""
        self.detections.append(detection)
        return self

    def __len__(self) -> int:
        """Return the number of detections in this frame."""
        return len(self.detections)

    def __getitem__(self, idx: int) -> Detection:
        """Return the detection at the given index."""
        return self.detections[idx]

    def __iter__(self):
        """Iterate over detections in this frame."""
        return iter(self.detections)

    def __eq__(self, other) -> bool:
        """Return True if two FrameDetections instances are equal."""
        if not isinstance(other, FrameDetections):
            return False
        return all(d1 == d2 for d1, d2 in zip(self.detections, other.detections))

    def get_detection_with_biggest_bb(self) -> Detection:
        """Return the detection with the largest bounding-box side."""
        return sorted(
            self.detections,
            key=lambda d: max(float(d.bb_xywh[2]), float(d.bb_xywh[3])),
            reverse=True,
        )[0]

    def get_detection_with_highest_score(self) -> Detection:
        """Return the detection with the highest confidence score."""
        return sorted(self.detections, key=lambda d: d.score, reverse=True)[0]

    def save_records(self) -> list[list[str]]:
        """Serialise all detections for CSV export."""
        return [d.save_record() for d in self.detections]

    def save(self, output_file: str | Path) -> None:
        """Write detections to a CSV file.

        Args:
            output_file: Destination path.

        """
        with open(output_file, "w") as f:
            writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(Detection.save_header())
            writer.writerows(self.save_records())

    def load(self, input_file: str | Path) -> FrameDetections:
        """Load detections from a CSV file.

        Args:
            input_file: Source path.

        Returns:
            Self (mutated in-place).

        """
        with open(input_file) as f:
            for line_count, record in enumerate(csv.reader(f, delimiter=",")):
                if line_count == 0:
                    continue
                d = {
                    "frame_id": int(record[0]),
                    "source": str(record[1]),
                    "score": float(record[2]),
                    "bb_xywh": torch.tensor([int(float(v)) for v in record[3:7]], dtype=torch.long),
                    "landmarks": torch.tensor(
                        [int(float(v)) for v in record[7:17]], dtype=torch.long
                    ).reshape(5, 2),
                }
                self.detections.append(DetectionFactory.create_detection(**d))
        return self


# ---------------------------------------------------------------------------
# VideoDetections
# ---------------------------------------------------------------------------


class VideoDetections:
    """Face detections for an entire video (collection of :class:`FrameDetections`)."""

    def __init__(self):
        self.detections: list[FrameDetections] = []

    def add(self, fdet: FrameDetections) -> VideoDetections:
        """Append a non-empty :class:`FrameDetections`."""
        if len(fdet) > 0:
            self.detections.append(fdet)
        return self

    def merge(self, vdet: VideoDetections) -> VideoDetections:
        """Merge all frame detections from another :class:`VideoDetections`."""
        for fdet in vdet:
            self.add(fdet)
        return self

    def frame_ids(self) -> list[int]:
        """Return the sorted list of frame indices with detections."""
        return [fdet.frame_id for fdet in self.detections]

    def get_frame_detection_with_frame_id(self, frame_id: int) -> FrameDetections:
        """Return the :class:`FrameDetections` for a given frame index."""
        return next(fd for fd in self.detections if fd.frame_id == frame_id)

    def __getitem__(self, index: int) -> FrameDetections:
        """Return the FrameDetections at the given index."""
        return self.detections[index]

    def __len__(self) -> int:
        """Return the number of frames with detections."""
        return len(self.detections)

    def __iter__(self):
        """Iterate over FrameDetections objects."""
        return iter(self.detections)

    def __eq__(self, other) -> bool:
        """Return True if two VideoDetections instances are equal."""
        if not isinstance(other, VideoDetections):
            return False
        return all(fd1 == fd2 for fd1, fd2 in zip(self.detections, other.detections))

    def save(self, output_file: str | Path) -> None:
        """Write all detections to a CSV file.

        Args:
            output_file: Destination path.

        """
        with open(output_file, "w") as f:
            writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(Detection.save_header())
            for fd in self.detections:
                writer.writerows(fd.save_records())

    def load(self, input_file: str | Path) -> VideoDetections:
        """Load detections from a CSV file.

        Args:
            input_file: Source path.

        Returns:
            Self (mutated in-place).

        """
        with open(input_file) as f:
            for line_count, record in enumerate(csv.reader(f, delimiter=",")):
                if line_count == 0:
                    continue
                d = {
                    "frame_id": int(record[0]),
                    "source": str(record[1]),
                    "score": float(record[2]),
                    "bb_xywh": torch.tensor([int(float(v)) for v in record[3:7]], dtype=torch.long),
                    "landmarks": torch.tensor(
                        [int(float(v)) for v in record[7:17]], dtype=torch.long
                    ).reshape(5, 2),
                }
                detection = DetectionFactory.create_detection(**d)

                fd = next(
                    (fd for fd in self.detections if fd[0].frame_id == detection.frame_id),
                    None,
                )
                if fd is None:
                    fd = FrameDetections()
                    self.detections.append(fd.add(detection))
                else:
                    fd.add(detection)
        return self


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------


class Track:
    """A sequence of :class:`Detection` objects spanning multiple frames.

    Represents a single face track produced by a :class:`Tracker`.
    """

    def __init__(self, track_id: int = -1, detection: Detection | None = None):
        self.track_id = track_id
        self.detections: list[Detection] = [] if detection is None else [detection]

    def frame_ids(self) -> list[int]:
        """Sorted list of frame indices in this track."""
        return sorted(d.frame_id for d in self.detections)

    def get_detection(self, frame_id: int) -> Detection:
        """Return the :class:`Detection` at a specific frame index."""
        return next(d for d in self.detections if d.frame_id == frame_id)

    def _sort(self) -> None:
        self.detections.sort(key=lambda d: d.frame_id)

    def add(self, detection: Detection) -> Track:
        """Append a detection and keep the track sorted by frame index."""
        self.detections.append(detection)
        self._sort()
        return self

    def merge(self, other: Track) -> Track:
        """Merge all detections from another track into this one."""
        self.detections += other.detections
        self._sort()
        return self

    def is_started_earlier(self, other: Track) -> bool:
        """Return ``True`` if this track starts before ``other``."""
        return self.first_detection().frame_id < other.first_detection().frame_id

    def frame_distance(self, other: Track) -> int:
        """Gap in frames between this track and ``other``.

        Returns 0 for overlapping tracks.
        """
        no1, no2 = (self, other) if self.is_started_earlier(other) else (other, self)
        if no1.last_detection().frame_id > no2.first_detection().frame_id:
            return 0
        return abs(no1.last_detection().frame_id - no2.first_detection().frame_id)

    def last_detection(self) -> Detection:
        """The last detection in the track (by frame index)."""
        return self.detections[-1]

    def first_detection(self) -> Detection:
        """The first detection in the track (by frame index)."""
        return self.detections[0]

    def sample(self, num: int = 5) -> list[Detection]:
        """Randomly sample up to ``num`` detections from the track.

        Args:
            num: Maximum number of detections to return.

        Returns:
            Randomly sampled detections (all of them if ``len < num``).

        """
        if len(self.detections) < num:
            return self.detections
        return random.sample(self.detections, num)

    def center(self) -> np.ndarray:
        """Mean centre point of all detections in the track.

        Returns:
            ``(2,)`` float numpy array ``[x, y]``.

        """
        xs = [float(xywh2midwh(d.bb_xywh)[0]) for d in self.detections]
        ys = [float(xywh2midwh(d.bb_xywh)[1]) for d in self.detections]
        return np.array([sum(xs) / len(xs), sum(ys) / len(ys)])

    def bb_size(self, extra_percent: float = 0.2) -> int:
        """Largest bounding-box dimension across all detections, with padding.

        Args:
            extra_percent: Fractional padding to add (e.g. ``0.2`` → +20 %).

        Returns:
            Integer pixel size.

        """
        sizes = [max(float(d.bb_xywh[2]), float(d.bb_xywh[3])) for d in self.detections]
        return int(max(sizes) * (1 + extra_percent))

    def __len__(self) -> int:
        """Return the number of detections in this track."""
        return len(self.detections)

    def __getitem__(self, index: int) -> Detection:
        """Return the detection at the given index."""
        return self.detections[index]

    def __iter__(self):
        """Iterate over detections in this track."""
        return iter(self.detections)

    def __str__(self) -> str:
        """Return a human-readable string representation of this track."""
        return (
            f"Track(id={self.track_id}, len={len(self)}, "
            f"frames={self.first_detection().frame_id}–{self.last_detection().frame_id})"
        )

    def save(self, output_file: str | Path) -> None:
        """Write track detections to a CSV file.

        Args:
            output_file: Destination path.

        """
        with open(output_file, "w") as f:
            writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(Detection.save_header())
            writer.writerows(d.save_record() for d in self.detections)

    def load(self, input_file: str | Path) -> Track:
        """Load track detections from a CSV file.

        Args:
            input_file: Source path.

        Returns:
            Self (mutated in-place).

        """
        with open(input_file) as f:
            for line_count, record in enumerate(csv.reader(f, delimiter=",")):
                if line_count == 0:
                    continue
                d = {
                    "frame_id": int(record[0]),
                    "source": str(record[1]),
                    "score": float(record[2]),
                    "bb_xywh": torch.tensor([int(float(v)) for v in record[3:7]], dtype=torch.long),
                    "landmarks": torch.tensor(
                        [int(float(v)) for v in record[7:17]], dtype=torch.long
                    ).reshape(5, 2),
                }
                self.add(DetectionFactory.create_detection(**d))
        return self


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class Tracker(ABC):
    """Abstract base class for multi-object face trackers.

    Implements IoU-based frame-to-track assignment via :meth:`label` and
    optional track merging via :meth:`merge`.  Subclasses implement the
    :meth:`merge_rule`.
    """

    def __init__(self, verbose: bool = False):
        self.new_track_id: int = 0
        self.tracks: dict[int, Track] = {}
        self._selected_tracks: dict[int, Track] = {}
        self.path_to_id: Callable[[str], int] = lambda p: int(Path(p).stem)
        self.id_to_path: Callable[[int], str] = lambda i: f"{i:06d}.png"
        self.verbose = verbose

    @property
    def selected_tracks(self) -> dict[int, Track]:
        """Active track selection, or all tracks if none selected."""
        return self.tracks if not self._selected_tracks else self._selected_tracks

    def _match_detection(
        self,
        detection: Detection,
        max_lost: int,
        iou_threshold: float,
    ) -> int | None:
        """Find the best existing track for a detection.

        Default implementation selects the open track whose last detection
        has the highest IoU with ``detection``, subject to ``max_lost`` and
        ``iou_threshold`` constraints.  Subclasses may override this to
        incorporate additional signals (e.g. face-identity embeddings).

        Args:
            detection: Candidate detection.
            max_lost: Maximum frame gap to keep a track open.
            iou_threshold: Minimum IoU required to continue a track.

        Returns:
            Best matching ``track_id``, or ``None`` if no track qualifies.

        """
        candidate_tracks: list[tuple[int, float]] = []
        for _, track in self.tracks.items():
            last = track.last_detection()
            iou = iou_xywh(last.bb_xywh, detection.bb_xywh)
            frame_dist = abs(detection.frame_id - last.frame_id)
            if frame_dist < max_lost and iou > iou_threshold:
                candidate_tracks.append((track.track_id, iou))

        if candidate_tracks:
            best_id, _ = sorted(candidate_tracks, key=lambda x: x[1], reverse=True)[0]
            return best_id
        return None

    def label(
        self,
        detections: VideoDetections,
        min_score: float = 0.7,
        max_lost: int = 30,
        iou_threshold: float = 0.2,
    ) -> Tracker:
        """Assign per-frame detections to persistent tracks.

        Iterates over all :class:`FrameDetections` in ``detections`` and
        delegates per-detection track matching to :meth:`_match_detection`.
        Detections below ``min_score`` are skipped.  A new track is started
        whenever :meth:`_match_detection` returns ``None``.

        Args:
            detections: Per-frame detection results.
            min_score: Minimum detection confidence to consider.
            max_lost: Maximum frame gap to keep a track open.
            iou_threshold: Minimum IoU required to continue a track.

        Returns:
            Self (tracks stored in :attr:`tracks`).

        """
        for frame_detections in tqdm(
            detections,
            total=len(detections),
            desc="Label tracks",
            disable=not self.verbose,
        ):
            for detection in frame_detections:
                if detection.score < min_score:
                    continue

                best_id = self._match_detection(detection, max_lost, iou_threshold)
                if best_id is not None:
                    self.tracks[best_id].add(detection)
                else:
                    self.tracks[self.new_track_id] = Track(self.new_track_id, detection)
                    self.new_track_id += 1

        return self

    @abstractmethod
    def merge_rule(self, track_1: Track, track_2: Track) -> tuple[bool, Track, Track]:
        """Determine whether two tracks should be merged.

        Args:
            track_1: First track.
            track_2: Second track.

        Returns:
            Tuple ``(should_merge, keep_track, drop_track)``.

        """

    def merge(self) -> Tracker:
        """Merge tracks that satisfy :meth:`merge_rule`.

        Returns:
            Self (tracks updated in-place).

        """
        if len(self.tracks) <= 1:
            return self

        track_ids = list(self.tracks.keys())
        blacklist: set[int] = set()

        for tid1 in track_ids:
            if tid1 in blacklist:
                continue
            t1 = self.tracks[tid1]
            for tid2 in track_ids:
                if tid1 == tid2 or tid2 in blacklist:
                    continue
                t2 = self.tracks[tid2]
                should_merge, keep, drop = self.merge_rule(t1, t2)
                if should_merge:
                    keep.merge(drop)
                    self.tracks.pop(drop.track_id)
                    blacklist.add(drop.track_id)
            blacklist.add(tid1)
        return self

    def select_long_tracks(self, min_length: int = 250) -> Tracker:
        """Select tracks with at least ``min_length`` detections.

        Args:
            min_length: Minimum number of detections.

        Returns:
            Self with ``_selected_tracks`` updated.

        """
        self._selected_tracks = {
            tid: t for tid, t in self.selected_tracks.items() if len(t) > min_length
        }
        return self

    def select_topk_long_tracks(self, top_k: int = 1) -> Tracker:
        """Select the ``top_k`` longest tracks.

        Args:
            top_k: Number of tracks to keep.

        Returns:
            Self with ``_selected_tracks`` updated.

        """
        sorted_tracks = sorted(self.selected_tracks.items(), key=lambda x: len(x[1]), reverse=True)
        self._selected_tracks = dict(sorted_tracks[:top_k])
        return self

    def select_topk_biggest_bb_tracks(self, top_k: int = 1) -> Tracker:
        """Select the ``top_k`` tracks with the largest average bounding boxes.

        Args:
            top_k: Number of tracks to keep.

        Returns:
            Self with ``_selected_tracks`` updated.

        """
        sorted_tracks = sorted(
            self.selected_tracks.items(),
            key=lambda x: x[1].bb_size(extra_percent=0.0),
            reverse=True,
        )
        self._selected_tracks = dict(sorted_tracks[:top_k])
        return self

    def get_center_track(self) -> Track | None:
        """Return the track whose mean position is closest to the frame centre.

        Returns:
            Closest :class:`Track`, or ``None`` if no tracks are selected.

        """

        def _dist(track: Track) -> float:
            c = track.center()
            fc = track.first_detection().frame_center()
            return sum((float(a) - float(b)) ** 2 for a, b in zip(c, fc)) ** 0.5

        sorted_tracks = sorted(self.selected_tracks.values(), key=_dist)
        return sorted_tracks[0] if sorted_tracks else None


# ---------------------------------------------------------------------------
# IouTracker
# ---------------------------------------------------------------------------


class IouTracker(Tracker):
    """IoU-based tracker for faces without large inter-frame displacements.

    Args:
        verbose: Show tqdm progress bar during labelling.
        max_lost: Maximum frame gap to keep a track open.  ``-1`` means
            unlimited.
        iou_threshold: Minimum IoU to continue an existing track.

    """

    def __init__(
        self,
        verbose: bool = False,
        max_lost: int = -1,
        iou_threshold: float = 0.2,
    ):
        super().__init__(verbose)
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def merge_rule(self, track_1: Track, track_2: Track) -> tuple[bool, Track, Track]:
        """Merge two tracks if they satisfy the max-lost and IoU constraints.

        Args:
            track_1: First track.
            track_2: Second track.

        Returns:
            ``(should_merge, keep_track, drop_track)`` where the earlier track
            is always kept.

        """
        no1, no2 = (track_1, track_2) if track_1.is_started_earlier(track_2) else (track_2, track_1)
        ok_lost = (self.max_lost == -1) or (no1.frame_distance(no2) <= self.max_lost)
        ok_iou = (
            iou_xywh(no1.last_detection().bb_xywh, no2.first_detection().bb_xywh)
            > self.iou_threshold
        )
        return (ok_lost and ok_iou), no1, no2


# ---------------------------------------------------------------------------
# FaceIdTracker
# ---------------------------------------------------------------------------


class FaceIdTracker(Tracker):
    """Face-ID-enhanced tracker using embedding similarity.

    Uses IoU as the primary fast gate.  Face identity embeddings fire only
    when IoU is ambiguous (2+ candidates) or for track recovery (0
    candidates).  This keeps most frames on the fast IoU path while still
    disambiguating overlapping faces and reconnecting tracks after gaps
    caused by occlusion, blur, or extreme head poses.

    The ``encoder`` is any callable that takes a ``(B, 3, H, W)`` uint8 RGB
    tensor and returns a ``(B, D)`` L2-normalised embedding tensor — for
    example :class:`~exordium.video.deep.adaface.AdaFaceWrapper`.

    Args:
        encoder: Face embedding model (callable).
        verbose: Show tqdm progress bar during labelling.
        max_lost: Maximum frame gap to keep a track open.  ``-1`` means
            unlimited.
        iou_threshold: Minimum IoU for the geometric gate.
        embedding_threshold: Minimum cosine similarity accepted for
            embedding-based matches (disambiguation and recovery).
        iou_weight: Weight for IoU in the combined score used when 2+ IoU
            candidates compete.  Embedding weight is ``1 - iou_weight``.
        recovery_max_lost: Maximum frame gap for embedding-only recovery
            when no IoU candidate is found.

    """

    def __init__(
        self,
        encoder: Callable[[torch.Tensor], torch.Tensor],
        verbose: bool = False,
        max_lost: int = -1,
        iou_threshold: float = 0.2,
        embedding_threshold: float = 0.5,
        iou_weight: float = 0.5,
        recovery_max_lost: int = 90,
    ):
        super().__init__(verbose)
        self.encoder = encoder
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.embedding_threshold = embedding_threshold
        self.iou_weight = iou_weight
        self.recovery_max_lost = recovery_max_lost

        self._track_embeddings: dict[int, torch.Tensor] = {}
        self._track_embed_counts: dict[int, int] = {}

    def _compute_embedding(self, detection: Detection) -> torch.Tensor:
        """Compute a normalised face embedding for a single detection."""
        crop = detection.crop(square=True, extra_space=1.5).unsqueeze(0)
        embedding = self.encoder(crop)
        vec = embedding.squeeze(0).detach().cpu()
        return torch.nn.functional.normalize(vec, p=2, dim=0)

    def _update_track_embedding(self, track_id: int, embedding: torch.Tensor) -> None:
        """Update a track's running-mean embedding with a new sample."""
        if track_id not in self._track_embeddings:
            self._track_embeddings[track_id] = embedding
            self._track_embed_counts[track_id] = 1
            return
        n = self._track_embed_counts[track_id]
        mean = self._track_embeddings[track_id]
        updated = (mean * n + embedding) / (n + 1)
        self._track_embeddings[track_id] = torch.nn.functional.normalize(updated, p=2, dim=0)
        self._track_embed_counts[track_id] = n + 1

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity between two 1-D tensors."""
        return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    def _match_detection(
        self,
        detection: Detection,
        max_lost: int,
        iou_threshold: float,
    ) -> int | None:
        """Match a detection using IoU gating with embedding tie-breaking.

        Three cases:

        * **1 IoU candidate** — fast path, assign directly and update the
          track's running embedding.
        * **2+ IoU candidates** — compute combined score
          ``iou_weight * iou + (1 - iou_weight) * cosine_sim``.
        * **0 IoU candidates** — embedding-only recovery against tracks
          within ``recovery_max_lost`` frames; accept the best if cosine
          similarity exceeds ``embedding_threshold``.

        Args:
            detection: Candidate detection to assign.
            max_lost: Maximum frame gap (from :meth:`Tracker.label`).
            iou_threshold: Minimum IoU (from :meth:`Tracker.label`).

        Returns:
            Matched ``track_id``, or ``None`` if a new track should be
            created.

        """
        candidates: list[tuple[int, float]] = []
        for _, track in self.tracks.items():
            last = track.last_detection()
            iou = iou_xywh(last.bb_xywh, detection.bb_xywh)
            frame_dist = abs(detection.frame_id - last.frame_id)
            if frame_dist < max_lost and iou > iou_threshold:
                candidates.append((track.track_id, iou))

        if len(candidates) == 1:
            best_id = candidates[0][0]
            embedding = self._compute_embedding(detection)
            self._update_track_embedding(best_id, embedding)
            return best_id

        if len(candidates) >= 2:
            embedding = self._compute_embedding(detection)
            best_id: int | None = None
            best_score = -1.0
            for track_id, iou in candidates:
                if track_id in self._track_embeddings:
                    sim = self._cosine_similarity(embedding, self._track_embeddings[track_id])
                else:
                    sim = 0.0
                score = self.iou_weight * iou + (1.0 - self.iou_weight) * sim
                if score > best_score:
                    best_score = score
                    best_id = track_id
            if best_id is not None:
                self._update_track_embedding(best_id, embedding)
            return best_id

        # Zero IoU candidates: try embedding-only recovery.
        embedding = self._compute_embedding(detection)

        if self._track_embeddings:
            best_id = None
            best_sim = -1.0
            for _, track in self.tracks.items():
                last = track.last_detection()
                frame_dist = abs(detection.frame_id - last.frame_id)
                if self.recovery_max_lost != -1 and frame_dist > self.recovery_max_lost:
                    continue
                if track.track_id not in self._track_embeddings:
                    continue
                sim = self._cosine_similarity(embedding, self._track_embeddings[track.track_id])
                if sim > best_sim:
                    best_sim = sim
                    best_id = track.track_id

            if best_id is not None and best_sim >= self.embedding_threshold:
                self._update_track_embedding(best_id, embedding)
                return best_id

        # No suitable existing track — seed a fresh embedding for the new
        # track that :meth:`Tracker.label` is about to create.
        self._track_embeddings[self.new_track_id] = embedding
        self._track_embed_counts[self.new_track_id] = 1
        return None

    def _ensure_track_embedding(self, track: Track) -> None:
        """Populate the running embedding for a track if missing."""
        if track.track_id in self._track_embeddings:
            return
        sample = track.sample(num=5)
        embeddings = [self._compute_embedding(d) for d in sample]
        mean = torch.stack(embeddings).mean(dim=0)
        self._track_embeddings[track.track_id] = torch.nn.functional.normalize(mean, p=2, dim=0)
        self._track_embed_counts[track.track_id] = len(embeddings)

    def merge_rule(self, track_1: Track, track_2: Track) -> tuple[bool, Track, Track]:
        """Merge two tracks based on frame-gap and embedding similarity.

        Args:
            track_1: First track.
            track_2: Second track.

        Returns:
            ``(should_merge, keep_track, drop_track)`` where the earlier
            track is always kept.

        """
        no1, no2 = (track_1, track_2) if track_1.is_started_earlier(track_2) else (track_2, track_1)
        ok_lost = (self.max_lost == -1) or (no1.frame_distance(no2) <= self.max_lost)
        if not ok_lost:
            return False, no1, no2

        self._ensure_track_embedding(no1)
        self._ensure_track_embedding(no2)

        sim = self._cosine_similarity(
            self._track_embeddings[no1.track_id],
            self._track_embeddings[no2.track_id],
        )
        return sim >= self.embedding_threshold, no1, no2


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def add_detections_to_frame(
    frame_detections: FrameDetections,
    frame: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    """Overlay detection bounding boxes and scores onto a video frame.

    Args:
        frame_detections: Detections for a single frame.
        frame: Image to draw on.  Accepted formats:

            * ``None`` — loaded from the first detection via
              :meth:`~Detection.frame` (returns RGB tensor, converted here).
            * ``np.ndarray (H, W, C)`` uint8 — assumed **BGR** (cv2 convention).
            * ``torch.Tensor (3, H, W)`` uint8 **RGB** — converted to BGR numpy.

    Returns:
        ``(H, W, C)`` BGR numpy array with detection annotations.

    """
    if frame is None:
        frame = frame_detections[0].frame()  # (3, H, W) RGB tensor

    frame_bgr = _to_bgr_numpy(frame)

    cv2.putText(
        frame_bgr,
        f"frame id: {frame_detections[0].frame_id:06d}",
        (5, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    for detection in frame_detections:
        x0, y0 = int(detection.bb_xywh[0]), int(detection.bb_xywh[1])
        bb = detection.bb_xyxy
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        cv2.putText(
            frame_bgr,
            f"score: {detection.score:.2f}",
            (x0 - 5, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame_bgr


def save_detections_to_video(
    video_detections: VideoDetections,
    frame_dir: str | Path,
    output_dir: str | Path,
    verbose: bool = False,
) -> None:
    """Save annotated frame images with detection overlays to a directory.

    Source frames are read from ``frame_dir`` via ``cv2.imread`` (BGR numpy),
    annotated with :func:`add_detections_to_frame`, and written as PNG files.

    Args:
        video_detections: Per-frame detection results.
        frame_dir: Directory containing source frames named with zero-padded
            integer stems (e.g. ``000000.png``).
        output_dir: Directory to write annotated PNG images to.
        verbose: Show a tqdm progress bar.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(Path(frame_dir).iterdir())
    frame_ids = [int(Path(p).stem) for p in frame_paths]
    det_ids = video_detections.frame_ids()

    for frame_id, frame_path in tqdm(
        zip(frame_ids, frame_paths),
        total=len(frame_ids),
        desc="Save frames",
        disable=not verbose,
    ):
        frame_bgr = cv2.imread(str(frame_path))  # BGR numpy
        if frame_bgr is None:
            continue
        if frame_id in det_ids:
            fd = video_detections.get_frame_detection_with_frame_id(frame_id)
            frame_bgr = add_detections_to_frame(fd, frame_bgr)
        cv2.imwrite(str(output_dir / f"{Path(frame_path).stem}.png"), frame_bgr)


def save_track_target_to_images(
    track: Track,
    output_dir: str | Path,
    bb_size: int = 224,
    fps: int = 30,
    sample_every_n: int = 1,
    save_video: bool = False,
    verbose: bool = False,
) -> None:
    """Save cropped face images for each detection in a track.

    Crops are produced by :meth:`~Detection.crop` (returns RGB tensor),
    then resized and saved as BGR PNG files via cv2.

    Args:
        track: Track containing the face detections.
        output_dir: Directory to write cropped PNG images to.
        bb_size: Target side length to resize each crop.  ``-1`` skips
            resizing.
        fps: Frame rate for the optional output video.
        sample_every_n: Save every n-th detection.
        save_video: Also assemble saved frames into an MP4.
        verbose: Show a tqdm progress bar.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, detection in tqdm(
        enumerate(track.detections),
        desc="Save track targets",
        disable=not verbose,
    ):
        if i % sample_every_n != 0:
            continue

        crop = detection.crop(square=True)  # (3, H', W') uint8 RGB tensor
        # Convert to (H', W', C) BGR numpy for cv2
        crop_np = cv2.cvtColor(crop.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)

        if bb_size != -1:
            crop_np = cv2.resize(crop_np, (bb_size, bb_size), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(output_dir / f"{detection.frame_id:06d}.png"), crop_np)

    if save_video:
        from exordium.video.core.io import sequence_to_video as frames2video

        frames2video(output_dir, output_dir.parent / f"{output_dir.stem}.mp4", fps)


def save_track_with_context_to_video(
    track: Track,
    frame_dir: str | Path,
    output_dir: str | Path,
    fps: int = 30,
    sample_every_n: int = 1,
    save_video: bool = False,
    verbose: bool = False,
) -> None:
    """Save full-frame images annotated with the track bounding box and score.

    Source frames are loaded from ``frame_dir`` via cv2 (BGR numpy).

    Args:
        track: Track containing the face detections.
        frame_dir: Directory containing source frames named with zero-padded
            integer stems.
        output_dir: Directory to write annotated PNG images to.
        fps: Frame rate for the optional output video.
        sample_every_n: Process every n-th frame.
        save_video: Also assemble saved frames into an MP4.
        verbose: Show a tqdm progress bar.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(Path(frame_dir).iterdir())
    frame_ids = [int(Path(p).stem) for p in frame_paths]
    track_frame_ids = track.frame_ids()

    for frame_id, frame_path in tqdm(
        zip(frame_ids, frame_paths),
        total=len(frame_paths),
        desc="Save video frames",
        disable=not verbose,
    ):
        if frame_id % sample_every_n != 0 or frame_id not in track_frame_ids:
            continue

        frame_bgr = cv2.imread(str(frame_path))  # BGR numpy
        if frame_bgr is None:
            continue
        detection = track.get_detection(frame_id)
        bb = detection.bb_xyxy
        x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])

        cv2.putText(
            frame_bgr,
            f"score: {detection.score:.2f}",
            (x1 - 5, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(output_dir / f"{frame_id:06d}.png"), frame_bgr)

    if save_video:
        from exordium.video.core.io import sequence_to_video as frames2video

        frames2video(output_dir, output_dir.parent / f"{output_dir.stem}.mp4", fps)


def visualize_detection(
    detection: Detection,
    output_path: str | Path | None = None,
    show_indices: bool = False,
) -> np.ndarray:
    """Visualize the landmarks of a single detection on its source frame.

    Landmarks are drawn in **full-image** pixel coordinates, so the returned
    image shows the entire source frame with keypoints overlaid.

    Args:
        detection: Detection to visualize.
        output_path: Path to save the output image.  ``None`` skips saving.
        show_indices: Draw landmark index numbers next to each point.

    Returns:
        ``(H, W, C)`` RGB numpy array with landmarks drawn on it.

    """
    from exordium.video.face.landmark.facemesh import visualize_landmarks

    frame = detection.frame()  # (3, H, W) RGB tensor
    # Convert to (H, W, C) RGB numpy for landmark drawing
    frame_np = frame.permute(1, 2, 0).contiguous().cpu().numpy()

    landmarks = detection.landmarks
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()

    return cast("np.ndarray", visualize_landmarks(frame_np, landmarks, output_path, show_indices))


def visualize_detection_crop(
    detection: Detection,
    square: bool = True,
    extra_space: float = 1.0,
    output_path: str | Path | None = None,
    show_indices: bool = False,
    radius: int = 3,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Visualize the landmarks of a single detection on its face crop.

    Unlike :func:`visualize_detection` (which draws on the full source frame),
    this function draws the keypoints on the face crop returned by
    :meth:`~Detection.crop`.  The landmark coordinates are automatically
    projected into crop-local space via :meth:`~Detection.crop_landmarks`.

    Example::

        from exordium.video.core.detection import visualize_detection_crop
        import cv2

        crop_vis = visualize_detection_crop(det, square=True, show_indices=True)
        # crop_vis is (H', W', 3) RGB numpy array
        cv2.imshow("face", cv2.cvtColor(crop_vis, cv2.COLOR_RGB2BGR))

    Args:
        detection: Detection to visualize.
        square: If ``True`` (default), produce a square crop centered on the
            bounding-box midpoint.  If ``False``, use the original bounding-box
            aspect ratio.
        extra_space: Multiplier applied to the bounding-box side(s).
            ``1.0`` (default) = tight crop.
        output_path: Path to save the output image.  ``None`` skips saving.
        show_indices: Draw landmark index numbers next to each point.
        radius: Circle radius for each keypoint in pixels.  Default: 3.
        color: BGR/RGB color tuple for the keypoints.  Default: green
            ``(0, 255, 0)``.

    Returns:
        ``(H', W', 3)`` RGB numpy array — the face crop with landmarks drawn.

    """
    from exordium.video.face.landmark.facemesh import visualize_landmarks

    # Standard RGB colors for YOLO11 5-point face keypoints:
    # 0=right eye (light green), 1=left eye (dark green), 2=nose (red),
    # 3=mouth-right (light blue), 4=mouth-left (dark blue)
    _FACE_5PT_COLORS: list[tuple[int, int, int]] = [
        (144, 238, 144),  # right eye  — light green
        (0, 128, 0),  # left eye   — dark green
        (255, 0, 0),  # nose       — red
        (135, 206, 235),  # mouth-right — light blue
        (0, 0, 200),  # mouth-left  — dark blue
    ]

    crop = detection.crop(square=square, extra_space=extra_space)  # (3, H', W') uint8 RGB tensor
    crop_np: np.ndarray = crop.permute(1, 2, 0).numpy()  # (H', W', 3) numpy
    lm = detection.crop_landmarks(square=square, extra_space=extra_space)

    colors = _FACE_5PT_COLORS if len(lm) == 5 else None
    result = visualize_landmarks(
        crop_np,
        lm,
        output_path,
        show_indices,
        radius=radius,
        color=color,
        colors=colors,
    )
    assert isinstance(result, np.ndarray)
    return result
