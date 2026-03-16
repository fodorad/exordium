"""Detection and tracking data structures."""

import csv
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import numpy as np
from tqdm import tqdm

from exordium.video.core.bb import crop_mid, xywh2midwh, xywh2xyxy
from exordium.video.core.io import image_to_np, load_frames


@dataclass(frozen=True, kw_only=True)
class Detection(ABC):
    """Abstract base class for face detections.

    Represents a single face detection with bounding box and landmarks.
    """

    frame_id: int
    source: (
        str | Path | np.ndarray
    )  # source of the detections (path to the image or video file or np.ndarray).
    score: float  # detector confidence value between [0..1].
    bb_xywh: np.ndarray  # bounding box of shape (4,) in xywh format.
    landmarks: np.ndarray  # xy pixel coordinates of the face landmarks of shape (5,2).
    # Order is left eye, right eye, nose, left mouth, right mouth.

    @property
    def bb_xyxy(self) -> np.ndarray:
        """Bounding box in xyxy format."""
        return xywh2xyxy(self.bb_xywh)

    @property
    def bb_center(self) -> np.ndarray:
        """Bounding box center coordinates."""
        return xywh2midwh(self.bb_xywh)[:2]

    @property
    def is_interpolated(self) -> bool:
        """Check if this detection is interpolated.

        Returns:
            True if the detection was interpolated between frames.
        """
        return self.score == -1

    @classmethod
    def save_header(cls):
        """Defines the header of the output file."""
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
        """Only detections with correct filepaths can be saved and loaded."""
        source_record = self.source if isinstance(self.source, str) else None
        return list(
            map(
                str,
                [
                    self.frame_id,
                    source_record,
                    self.score,
                    *self.bb_xywh,
                    *self.landmarks.flatten(),
                ],
            )
        )

    @abstractmethod
    def frame(self, *args, **kwargs) -> np.ndarray:
        """Load the corresponding frame.

        Returns:
            Frame image as numpy array.
        """

    @abstractmethod
    def frame_center(self) -> np.ndarray:
        """Calculate the center of the frame in xy format.

        Returns:
            Center coordinates as [x, y].
        """

    @abstractmethod
    def bb_crop(self) -> np.ndarray:
        """Crop the bounding box area from the frame.

        Returns:
            Cropped face image.
        """

    @abstractmethod
    def bb_crop_wide(self, extra_space: float = 1.5) -> np.ndarray:
        """Crop a wider bounding box area from the frame.

        Args:
            extra_space: Multiplier for the bounding box size. Defaults to 1.5.

        Returns:
            Cropped face image with extra context.
        """

    def __eq__(self, other) -> bool:
        """Check equality between two Detection objects.

        Args:
            other: Other object to compare with.

        Returns:
            True if detections are equal, False otherwise.
        """
        if not isinstance(other, Detection):
            return False

        return bool(
            self.frame_id == other.frame_id
            and self.source == other.source
            and abs(self.score - other.score) < 1e-6
            and np.all(np.equal(self.bb_xyxy, other.bb_xyxy))
            and np.all(np.equal(self.bb_xywh, other.bb_xywh))
            and np.all(np.equal(self.landmarks, other.landmarks))
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class DetectionFromImage(Detection):
    """Detection from a single image file."""

    source: str | Path

    def frame(self) -> np.ndarray:
        """Load the image frame.

        Returns:
            RGB image array.
        """
        return image_to_np(self.source, "RGB")

    def frame_center(self) -> np.ndarray:
        """Calculate frame center coordinates.

        Returns:
            Center point as [x, y].
        """
        height, width = self.frame().shape[:2]
        return np.rint(np.array([width / 2, height / 2])).astype(int)

    def bb_crop(self) -> np.ndarray:
        """Crop bounding box region from frame.

        Returns:
            Cropped image.
        """
        return crop_mid(
            image=self.frame(), mid=xywh2midwh(self.bb_xywh)[:2], bb_size=max(self.bb_xywh[2:])
        )

    def bb_crop_wide(self, extra_space: float = 1.5) -> np.ndarray:
        """Crop wider bounding box region from frame.

        Args:
            extra_space: Multiplier for bounding box size. Defaults to 1.5.

        Returns:
            Cropped image with extra context.
        """
        return crop_mid(
            image=self.frame(),
            mid=xywh2midwh(self.bb_xywh)[:2],
            bb_size=np.rint(max(self.bb_xywh[2:]) * extra_space).astype(int),
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class DetectionFromVideo(Detection):
    """Detection from a video file."""

    source: str | Path

    def frame(self) -> np.ndarray:
        """Load the video frame.

        Returns:
            RGB image array.
        """
        # Load single frame at self.frame_id
        # load_frames returns torch.Tensor of shape (T, C, H, W) in RGB format
        frames = load_frames(
            input_path=self.source,
            frame_ids=[self.frame_id],
            device_id=None,  # Use CPU
        )

        # Convert from (1, C, H, W) to (H, W, C) numpy array
        frame = frames[0].permute(1, 2, 0).cpu().numpy()
        return frame

    def bb_crop(self) -> np.ndarray:
        """Crop bounding box region from frame.

        Returns:
            Cropped image.
        """
        return crop_mid(
            image=self.frame(), mid=xywh2midwh(self.bb_xywh)[:2], bb_size=max(self.bb_xywh[2:])
        )

    def bb_crop_wide(self, extra_space: float = 1.5) -> np.ndarray:
        """Crop wider bounding box region from frame.

        Args:
            extra_space: Multiplier for bounding box size. Defaults to 1.5.

        Returns:
            Cropped image with extra context.
        """
        return crop_mid(
            image=self.frame(),
            mid=xywh2midwh(self.bb_xywh)[:2],
            bb_size=np.rint(max(self.bb_xywh[2:]) * extra_space).astype(int),
        )

    def frame_center(self) -> np.ndarray:
        """Calculate frame center coordinates.

        Returns:
            Center point as [x, y].
        """
        height, width = self.frame().shape[:2]
        return np.rint(np.array([width / 2, height / 2])).astype(int)


@dataclass(frozen=True, kw_only=True, eq=False)
class DetectionFromTensor(Detection):
    """Detection from a numpy array tensor (in-memory frame)."""

    source: np.ndarray  # The source is expected to be the frame itself as a numpy array.

    def frame(self) -> np.ndarray:
        """Get the frame tensor.

        Returns:
            RGB image array.
        """
        return self.source

    def frame_center(self) -> np.ndarray:
        """Calculate frame center coordinates.

        Returns:
            Center point as [x, y].
        """
        height, width = self.frame().shape[:2]
        return np.rint(np.array([width / 2, height / 2])).astype(int)

    def bb_crop(self) -> np.ndarray:
        """Crop bounding box region from frame.

        Returns:
            Cropped image.
        """
        return crop_mid(
            image=self.frame(), mid=xywh2midwh(self.bb_xywh)[:2], bb_size=max(self.bb_xywh[2:])
        )

    def bb_crop_wide(self, extra_space: float = 1.5) -> np.ndarray:
        """Crop wider bounding box region from frame.

        Args:
            extra_space: Multiplier for bounding box size. Defaults to 1.5.

        Returns:
            Cropped image with extra context.
        """
        return crop_mid(
            image=self.frame(),
            mid=xywh2midwh(self.bb_xywh)[:2],
            bb_size=np.rint(max(self.bb_xywh[2:]) * extra_space).astype(int),
        )


class DetectionFactory:
    """Factory for creating Detection instances based on source type."""

    VIDEO_EXTENSIONS = {".mp4", ".mpeg", ".mov", ".mkv", ".avi"}
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

    @classmethod
    def create_detection(cls, **kwargs) -> Detection:
        """Create a Detection instance from keyword arguments.

        Args:
            **kwargs: Detection parameters.

        Returns:
            Appropriate Detection subclass instance.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If file extension is unsupported.
        """
        missing = set(Detection.__dataclass_fields__.keys()) - set(kwargs.keys())
        if missing:
            raise KeyError(f"Invalid Detection dictionary. Missing keys: {missing}")

        source = kwargs.get("source")

        if isinstance(source, str):
            extension = Path(source).suffix.lower()

            if extension in cls.VIDEO_EXTENSIONS:
                return DetectionFromVideo(**kwargs)
            elif extension in cls.IMAGE_EXTENSIONS:
                return DetectionFromImage(**kwargs)
            else:
                raise ValueError(f"Unsupported extension: {extension}")

        return DetectionFromTensor(**kwargs)


class FrameDetections:
    """Represents face bounding boxes within a single frame."""

    def __init__(self):
        self.detections: list[Detection] = []

    @property
    def frame_id(self):
        """Returns the id of the frame."""
        return self.detections[0].frame_id

    @property
    def source(self):
        """Returns the source of the detection."""
        return self.detections[0].source

    def add_dict(self, detection: dict) -> Self:
        """Creates and adds Detection instance."""
        self.detections.append(DetectionFactory.create_detection(**detection))
        return self

    def add(self, detection: Detection) -> Self:
        """Adds a Detection instance."""
        self.detections.append(detection)
        return self

    def __len__(self) -> int:
        """Get number of detections in frame.

        Returns:
            Number of detections.
        """
        return len(self.detections)

    def __getitem__(self, idx) -> Detection:
        """Get detection at index.

        Args:
            idx: Index of detection.

        Returns:
            Detection at index.
        """
        return self.detections[idx]

    def __iter__(self):
        """Iterate over detections.

        Returns:
            Iterator over detections.
        """
        return iter(self.detections)

    def __eq__(self, other) -> bool:
        """Check equality with another FrameDetections.

        Args:
            other: Other object to compare.

        Returns:
            True if equal, False otherwise.
        """
        if not isinstance(other, FrameDetections):
            return False

        for fdet1, fdet2 in zip(self.detections, other.detections):
            if fdet1 != fdet2:
                return False
        return True

    def get_detection_with_biggest_bb(self) -> Detection:
        """Returns a Detection with the biggest bounding box side."""
        return sorted(self.detections, key=lambda d: np.max(d.bb_xywh[2:]), reverse=True)[0]

    def get_detection_with_highest_score(self) -> Detection:
        """Returns a Detection with the highest confidence score."""
        return sorted(self.detections, key=lambda d: d.score, reverse=True)[0]

    def save_records(self) -> list[list[str]]:
        """Prepares the records for export."""
        return [detection.save_record() for detection in self.detections]

    def save(self, output_file: str | Path) -> None:
        """Saves the instance to a file.

        Args:
            output_file (str | Path): path to the output file.

        """
        with open(output_file, "w") as f:
            writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(Detection.save_header())
            writer.writerows(self.save_records())

    def load(self, input_file: str | Path) -> Self:
        """Loads the instance from a file.

        Args:
            input_file (str | Path): path to the input file.

        Returns:
            FrameDetections: loaded instance.

        """
        with open(input_file) as f:
            csv_reader = csv.reader(f, delimiter=",")

            for line_count, record in enumerate(csv_reader):
                if line_count > 0:
                    d = {
                        "frame_id": int(record[0]),
                        "source": str(record[1]),
                        "score": float(record[2]),
                        "bb_xywh": np.array(record[3:7], dtype=int),
                        "landmarks": np.array(record[7:17], dtype=int).reshape(5, 2),
                    }
                    detection: Detection = DetectionFactory.create_detection(**d)
                    self.detections.append(detection)
        return self


class VideoDetections:
    """Represents face bounding boxes within multiple frames from a single video."""

    def __init__(self):
        self.detections: list[FrameDetections] = []

    def add(self, fdet: FrameDetections) -> Self:
        """Adds a FrameDetections instance."""
        if len(fdet) > 0:
            self.detections.append(fdet)
        return self

    def merge(self, vdet: Self) -> Self:
        """Merges the given VideoDetections instance to self."""
        for fdet in vdet:
            self.add(fdet)
        return self

    def frame_ids(self) -> list[int]:
        """Returns the frame ids."""
        return [fdet.frame_id for fdet in self.detections]

    def get_frame_detection_with_frame_id(self, frame_id: int) -> FrameDetections:
        """Returns the FrameDetections instance with the given frame id."""
        return next(
            frame_detection
            for frame_detection in self.detections
            if frame_detection.frame_id == frame_id
        )

    def __getitem__(self, index: int) -> FrameDetections:
        """Get FrameDetections at index.

        Args:
            index: Index of frame detections.

        Returns:
            FrameDetections at index.
        """
        return self.detections[index]

    def __len__(self):
        """Get number of frames with detections.

        Returns:
            Number of FrameDetections.
        """
        return len(self.detections)

    def __iter__(self):
        """Iterate over FrameDetections.

        Returns:
            Iterator over FrameDetections.
        """
        return iter(self.detections)

    def __eq__(self, other) -> bool:
        """Check equality with another VideoDetections.

        Args:
            other: Other object to compare.

        Returns:
            True if equal, False otherwise.
        """
        if not isinstance(other, VideoDetections):
            return False

        for fdet1, fdet2 in zip(self.detections, other.detections):
            if fdet1 != fdet2:
                return False
        return True

    def save(self, output_file: str | Path) -> None:
        """Saves the instance to a file.

        Args:
            output_file (str | Path): path to the output file.

        """
        with open(output_file, "w") as f:
            writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(Detection.save_header())

            for frame_detections in self.detections:
                writer.writerows(frame_detections.save_records())

    def load(self, input_file: str | Path) -> Self:
        """Loads the instance from a file.

        Args:
            input_file (str | Path): path to the input file.

        Returns:
            VideoDetections: loaded instance.

        """
        with open(input_file) as f:
            csv_reader = csv.reader(f, delimiter=",")

            for line_count, record in enumerate(csv_reader):
                if line_count == 0:
                    continue
                d = {
                    "frame_id": int(record[0]),
                    "source": str(record[1]),
                    "score": float(record[2]),
                    "bb_xywh": np.array(record[3:7], dtype=int),
                    "landmarks": np.array(record[7:17], dtype=int).reshape(5, 2),
                }
                detection: Detection = DetectionFactory.create_detection(**d)

                frame_detections: FrameDetections | None = next(
                    (
                        frame_detections
                        for frame_detections in self.detections
                        if frame_detections[0].frame_id == detection.frame_id
                    ),
                    None,
                )

                if frame_detections is None:
                    frame_detections = FrameDetections()
                    self.detections.append(frame_detections.add(detection))
                else:
                    frame_detections.add(detection)

        return self


class Track:
    """Represents face bounding box tracks over multiple frames from a single video."""

    def __init__(self, track_id: int = -1, detection: Detection | None = None):
        self.track_id = track_id
        self.detections: list[Detection] = [] if detection is None else [detection]

    def frame_ids(self) -> list[int]:
        """Lists the frame ids within the track."""
        return sorted([elem.frame_id for elem in self.detections])

    def get_detection(self, frame_id: int) -> Detection:
        """Returns the Detection instance with the given frame id."""
        return next(detection for detection in self.detections if detection.frame_id == frame_id)

    def __sort_detections(self) -> None:
        """Sorts the list containing the detections based on the frame id."""
        self.detections.sort(key=lambda obj: obj.frame_id)

    def add(self, detection: Detection) -> Self:
        """Adds Detection instance to the track."""
        self.detections.append(detection)
        self.__sort_detections()
        return self

    def merge(self, detection: Self) -> Self:
        """Merges the given track to self."""
        self.detections += detection.detections
        self.__sort_detections()
        return self

    def is_started_earlier(self, track: Self) -> bool:
        """Determines if self is started earlier than the given track."""
        return self.first_detection().frame_id < track.first_detection().frame_id

    def frame_distance(self, track: Self) -> int:
        """Calculate the distance between two Track instances.

        The distance is calculated as the gap between the end of the earlier
        track and the beginning of the later track. Overlapping tracks have
        a distance of 0.

        Example:
            track1: [1, 2, 3]
            track2: [5, 6, 7]
            frame distance is 2

        Args:
            track: Another track to compare with.

        Returns:
            Frame distance between tracks.
        """
        no1, no2 = (self, track) if self.is_started_earlier(track) else (track, self)

        if no1.last_detection().frame_id > no2.first_detection().frame_id:
            return 0

        return abs(no1.last_detection().frame_id - no2.first_detection().frame_id)

    def last_detection(self) -> Detection:
        """Gives the last Detection instance from the track."""
        return self.detections[-1]

    def first_detection(self) -> Detection:
        """Gives the first Detection instance from the track."""
        return self.detections[0]

    def sample(self, num: int = 5) -> list[Detection]:
        """Gives multiple random Detection instances from the track.

        Args:
            num (int): number of detections.

        Returns:
           list[Detection]: randomly sampled detections.

        """
        dets = [det for det in self.detections if not det.is_interpolated]
        if len(dets) < num:
            return dets
        return random.sample(dets, num)

    def center(self) -> np.ndarray:
        """Calculate the center point of all detections in the track.

        Returns:
            Center coordinates [x, y].
        """
        xs, ys = [], []
        for detection in self.detections:
            if not detection.is_interpolated:
                bb_midwh = xywh2midwh(detection.bb_xywh)
                xs.append(bb_midwh[0])
                ys.append(bb_midwh[1])
        return np.array([np.array(xs).mean(), np.array(ys).mean()])

    def bb_size(self, extra_percent: float = 0.2) -> int:
        """Calculate bounding box size for the track.

        Args:
            extra_percent: Extra percentage to add. Defaults to 0.2.

        Returns:
            Bounding box size.
        """
        ws, hs = [], []
        for detection in self.detections:
            if not detection.is_interpolated:
                ws.append(detection.bb_xywh[2])
                hs.append(detection.bb_xywh[3])
        return int(max(ws + hs) * (1 + extra_percent))

    def __len__(self) -> int:
        """Get number of detections in track.

        Returns:
            Number of detections.
        """
        return len(self.detections)

    def __getitem__(self, index: int) -> Detection:
        """Get detection at index.

        Args:
            index: Index of detection.

        Returns:
            Detection at index.
        """
        return self.detections[index]

    def __str__(self) -> str:
        """Get string representation of track.

        Returns:
            String representation.
        """
        return (
            f"ID {self.track_id} track with {len(self)} dets "
            f"from {self.first_detection().frame_id} to {self.last_detection().frame_id}."
        )

    def __iter__(self):
        """Iterate over detections in track.

        Returns:
            Iterator over detections.
        """
        return iter(self.detections)

    def save(self, output_file: str | Path) -> None:
        """Saves the instance to a file.

        Args:
            output_file (str | Path): path to the output file.

        """
        with open(output_file, "w") as f:
            writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(Detection.save_header())
            writer.writerows([detection.save_record() for detection in self.detections])

    def load(self, input_file: str | Path) -> Self:
        """Loads the instance from a file.

        Args:
            input_file (str | Path): path to the input file.

        Returns:
            Track: loaded instance.

        """
        with open(input_file) as f:
            csv_reader = csv.reader(f, delimiter=",")

            for line_count, record in enumerate(csv_reader):
                if line_count == 0:
                    continue
                d = {
                    "frame_id": int(record[0]),
                    "source": str(record[1]),
                    "score": float(record[2]),
                    "bb_xywh": np.array(record[3:7], dtype=int),
                    "landmarks": np.array(record[7:17], dtype=int).reshape(5, 2),
                }
                detection: Detection = DetectionFactory.create_detection(**d)
                self.add(detection)

        return self


# ── Visualization helpers ──────────────────────────────────────────────────────


def add_detections_to_frame(
    frame_detections: "FrameDetections", frame: np.ndarray | None = None
) -> np.ndarray:
    """Overlay detection bounding boxes and scores onto a video frame.

    Args:
        frame_detections: Detections for a single frame.
        frame: BGR image to draw on. If ``None``, the image is loaded from
            the first detection. Defaults to ``None``.

    Returns:
        BGR image with detection annotations.

    """
    if frame is None:
        frame = frame_detections[0].frame()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(
        frame,
        f"frame id: {frame_detections[0].frame_id:06d}",
        (5, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    for detection in frame_detections:
        cv2.putText(
            frame,
            f"score: {detection.score:.2f}",
            (detection.bb_xywh[0] - 5, detection.bb_xywh[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        bb_xyxy = xywh2xyxy(detection.bb_xywh)
        cv2.rectangle(frame, (bb_xyxy[0], bb_xyxy[1]), (bb_xyxy[2], bb_xyxy[3]), (0, 255, 0), 2)
    return frame


def save_detections_to_video(
    video_detections: "VideoDetections",
    frame_dir: str | Path,
    output_dir: str | Path,
    verbose: bool = False,
) -> None:
    """Save annotated frame images with detection overlays to a directory.

    Args:
        video_detections: Per-frame detection results for the video.
        frame_dir: Directory containing source frame images named with
            zero-padded integer stems.
        output_dir: Directory to write annotated PNG images to.
        verbose: Show a progress bar. Defaults to ``False``.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(list(Path(frame_dir).iterdir()))
    frame_ids = [int(Path(frame_path).stem) for frame_path in frame_paths]
    frame_detection_ids = video_detections.frame_ids()
    for frame_id, frame_path in tqdm(
        zip(frame_ids, frame_paths), total=len(frame_ids), desc="Save frames", disable=not verbose
    ):
        frame = cv2.imread(str(frame_path))
        if frame_id in frame_detection_ids:
            frame_detection = video_detections.get_frame_detection_with_frame_id(frame_id)
            frame = add_detections_to_frame(frame_detection, frame)
        cv2.imwrite(str(output_dir / f"{Path(frame_path).stem}.png"), frame)


def save_track_target_to_images(
    track: "Track",
    output_dir: str | Path,
    bb_size: int = 224,
    fps: int = 30,
    sample_every_n: int = 1,
    save_video: bool = False,
    verbose: bool = False,
) -> None:
    """Save cropped face bounding box images for each detection in a track.

    Args:
        track: Track object containing a sequence of detections.
        output_dir: Directory to write the cropped PNG images to.
        bb_size: Target size to resize each crop to. ``-1`` skips resizing.
            Defaults to 224.
        fps: Frame rate for the optional output video. Defaults to 30.
        sample_every_n: Save every n-th detection. Defaults to 1.
        save_video: Also assemble the saved frames into an MP4. Defaults to
            ``False``.
        verbose: Show a progress bar. Defaults to ``False``.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, detection in tqdm(
        enumerate(track.detections), desc="Save track targets", disable=not verbose
    ):
        if i % sample_every_n != 0:
            continue
        image = detection.bb_crop()
        if bb_size != -1:
            image = cv2.resize(image, (bb_size, bb_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output_dir / f"{detection.frame_id:06d}.png"), image)

    if save_video:  # pragma: no cover
        from exordium.video.core.io import sequence_to_video as frames2video

        frames2video(output_dir, output_dir.parent / f"{output_dir.stem}.mp4", fps)


def save_track_with_context_to_video(
    track: "Track",
    frame_dir: str | Path,
    output_dir: str | Path,
    fps: int = 30,
    sample_every_n: int = 1,
    save_video: bool = False,
    verbose: bool = False,
) -> None:
    """Save full-frame images annotated with the track bounding box and score.

    Args:
        track: Track object containing a sequence of detections.
        frame_dir: Directory containing source frame images named with
            zero-padded integer stems.
        output_dir: Directory to write annotated PNG images to.
        fps: Frame rate for the optional output video. Defaults to 30.
        sample_every_n: Process every n-th frame. Defaults to 1.
        save_video: Also assemble the saved frames into an MP4. Defaults to
            ``False``.
        verbose: Show a progress bar. Defaults to ``False``.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(list(Path(frame_dir).iterdir()))
    frame_ids = [int(Path(frame_path).stem) for frame_path in frame_paths]
    track_frame_ids = track.frame_ids()

    for frame_id, frame_path in tqdm(
        zip(frame_ids, frame_paths),
        total=len(frame_paths),
        desc="Save video frames",
        disable=not verbose,
    ):
        if frame_id % sample_every_n != 0:
            continue
        if frame_id not in track_frame_ids:
            continue

        frame = cv2.imread(str(frame_path))
        detection = track.get_detection(frame_id)

        cv2.putText(
            frame,
            f"score: {detection.score:.2f}",
            (detection.bb_xyxy[0] - 5, detection.bb_xyxy[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.rectangle(
            frame,
            (detection.bb_xyxy[0], detection.bb_xyxy[1]),
            (detection.bb_xyxy[2], detection.bb_xyxy[3]),
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(Path(output_dir) / f"{frame_id:06d}.png"), frame)

    if save_video:  # pragma: no cover
        from exordium.video.core.io import sequence_to_video as frames2video

        frames2video(output_dir, output_dir.parent / f"{output_dir.stem}.mp4", fps)


def visualize_detection(
    detection: "Detection",
    output_path: os.PathLike | None = None,
    show_indices: bool = False,
) -> np.ndarray:
    """Visualize the landmarks of a single detection on its source frame.

    Args:
        detection: Detection object containing the frame and landmark coordinates.
        output_path: Path to save the output image. ``None`` skips saving.
        show_indices: Draw landmark indices next to each point.

    Returns:
        Image with landmarks drawn on it.

    """
    from exordium.video.face.landmark.facemesh import visualize_landmarks

    return visualize_landmarks(detection.frame(), detection.landmarks, output_path, show_indices)
