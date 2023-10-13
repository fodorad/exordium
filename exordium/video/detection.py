import csv
import random
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Self
from dataclasses import dataclass
import numpy as np
from decord import VideoReader, cpu
from exordium import PathType
from exordium.video.io import image2np
from exordium.video.bb import xywh2xyxy, xywh2midwh, crop_mid


@dataclass(frozen=True, kw_only=True)
class Detection(ABC):
    frame_id: int # frame id.
    source: str | np.ndarray # source of the detections (path to the image or video file) or the detection is from np.ndarray.
    score: float # detector confidence value between [0..1].
    bb_xywh: np.ndarray # bounding box of shape (4,) in xywh format.
    landmarks: np.ndarray # xy pixel coordinates of the face landmarks of shape (5,2).
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
        return self.score == -1

    @classmethod
    def save_header(cls):
        """Defines the header of the output file."""
        return [
            'frame_id', 'source', 'score', 'x', 'y', 'w', 'h',
            'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y',
            'nose_x', 'nose_y',
            'left_mouth_x', 'left_mouth_y', 'right_mouth_x', 'right_mouth_y'
        ]

    def save_record(self) -> list[str]:
        """Only detections with correct filepaths can be saved and loaded."""
        source_record = self.source if isinstance(self.source, str) else None
        return list(map(str, [
            self.frame_id, source_record, self.score,
            *self.bb_xywh, *self.landmarks.flatten()
        ]))

    @abstractmethod
    def frame(self, *args, **kwargs) -> np.ndarray:
        """Loads the corresponding frame."""

    @abstractmethod
    def frame_center(self) -> np.ndarray:
        """Calculates the center of the frame in xy format."""

    @abstractmethod
    def bb_crop(self) -> np.ndarray:
        """Crop the bounding box area from the frame."""

    def __eq__(self, other) -> bool:
        if not isinstance(other, Detection):
            return False

        return bool(self.frame_id == other.frame_id and \
                    self.source == other.source and \
                    abs(self.score - other.score) < 1e-6 and \
                    np.all(np.equal(self.bb_xyxy, other.bb_xyxy)) and \
                    np.all(np.equal(self.bb_xywh, other.bb_xywh)) and \
                    np.all(np.equal(self.landmarks, other.landmarks)))


@dataclass(frozen=True, kw_only=True, eq=False)
class DetectionFromImage(Detection):

    source: str

    def frame(self) -> np.ndarray:
        return image2np(self.source, 'RGB')

    def frame_center(self) -> np.ndarray:
        height, width = self.frame().shape[:2]
        return np.rint(np.array([width / 2, height / 2])).astype(int)

    def bb_crop(self) -> np.ndarray:
        return crop_mid(image=self.frame(),
                        mid=xywh2midwh(self.bb_xywh)[:2],
                        bb_size=max(self.bb_xywh[2:]))


@dataclass(frozen=True, kw_only=True, eq=False)
class DetectionFromVideo(Detection):

    source: str

    def _create_vr(self) -> VideoReader:
        return VideoReader(str(self.source), ctx=cpu(0))

    def frame(self, vr: VideoReader | None = None) -> np.ndarray:
        vr = vr or self._create_vr()
        return vr[self.frame_id].asnumpy() # RGB

    def bb_crop(self, vr: VideoReader | None = None) -> np.ndarray:
        vr = vr or self._create_vr()
        return crop_mid(image=self.frame(vr),
                        mid=xywh2midwh(self.bb_xywh)[:2],
                        bb_size=max(self.bb_xywh[2:]))

    def frame_center(self, vr: VideoReader | None = None) -> np.ndarray:
        vr = vr or self._create_vr()
        height, width = self.frame(vr).shape[:2]
        return np.rint(np.array([width / 2, height / 2])).astype(int)


@dataclass(frozen=True, kw_only=True, eq=False)
class DetectionFromTensor(Detection):

    source: np.ndarray

    def frame(self) -> np.ndarray:
        if isinstance(self.source, np.ndarray):
            return self.source

        raise ValueError('DetectionFromTensor subclass should keep the image itself as the source.')

    def frame_center(self) -> np.ndarray:
        height, width = self.frame().shape[:2]
        return np.rint(np.array([width / 2, height / 2])).astype(int)

    def bb_crop(self) -> np.ndarray:
        return crop_mid(image=self.frame(), mid=xywh2midwh(self.bb_xywh)[:2], bb_size=max(self.bb_xywh[2:]))


class DetectionFactory:

    @classmethod
    def create_detection(cls, **kwargs) -> Detection:
        """Handles Detection instance creation based on the source of the data."""
        for key in list(Detection.__dataclass_fields__.keys()):
            if not key in list(kwargs.keys()):
                raise KeyError(f'Invalid Detection dictionary. Missing key {key}.')

        source: str | np.ndarray | None = kwargs.get('source')

        if isinstance(source, str):
            extension = Path(source).suffix.lower()

            match extension:
                case '.mp4' | '.mpeg' | '.mov' | '.mkv' | '.avi':
                    return DetectionFromVideo(**kwargs)
                case '.png' | '.jpg' | '.jpeg' | '.bmp':
                    return DetectionFromImage(**kwargs)
                case _:
                    raise ValueError(f'Given file with {extension} is not supported.')

        return DetectionFromTensor(**kwargs)


class FrameDetections:
    """Represents face bounding boxes within a single frame."""

    def __init__(self):
        self.detections: list[Detection] = []
        self.index = 0

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
        return len(self.detections)

    def __getitem__(self, idx) -> Detection:
        return self.detections[idx]

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.detections):
            value = self.detections[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration

    def __eq__(self, other) -> bool:
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

    def save(self, output_file: PathType) -> None:
        """Saves the instance to a file.

        Args:
            output_file (PathType): path to the output file.
        """
        with open(output_file, 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(Detection.save_header())
            writer.writerows(self.save_records())

    def load(self, input_file: PathType) -> Self:
        """Loads the instance from a file.

        Args:
            input_file (PathType): path to the input file.

        Returns:
            FrameDetections: loaded instance.
        """
        with open(input_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')

            for line_count, record in enumerate(csv_reader):
                if line_count > 0:
                    d = {
                        'frame_id': int(record[0]),
                        'source': str(record[1]),
                        'score': float(record[2]),
                        'bb_xywh': np.array(record[3:7], dtype=int),
                        'landmarks': np.array(record[7:17], dtype=int).reshape(5, 2)
                    }
                    detection: Detection = DetectionFactory.create_detection(**d)
                    self.detections.append(detection)
        return self


class VideoDetections:
    """Represents face bounding boxes within multiple frames from a single video."""

    def __init__(self):
        self.detections: list[FrameDetections] = []
        self.index = 0

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
        return next((frame_detection for frame_detection in self.detections
                                     if frame_detection.frame_id == frame_id))

    def __getitem__(self, index: int) -> FrameDetections:
        return self.detections[index]

    def __len__(self):
        return len(self.detections)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.detections):
            value = self.detections[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration

    def __eq__(self, other) -> bool:
        if not isinstance(other, VideoDetections):
            return False

        for fdet1, fdet2 in zip(self.detections, other.detections):
            if fdet1 != fdet2: return False
        return True

    def save(self, output_file: PathType) -> None:
        """Saves the instance to a file.

        Args:
            output_file (PathType): path to the output file.
        """
        with open(output_file, 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(Detection.save_header())

            for frame_detections in self.detections:
                writer.writerows(frame_detections.save_records())

    def load(self, input_file: PathType) -> Self:
        """Loads the instance from a file.

        Args:
            input_file (PathType): path to the input file.

        Returns:
            VideoDetections: loaded instance.
        """
        with open(input_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')

            for line_count, record in enumerate(csv_reader):
                if line_count == 0: continue
                d = {
                    'frame_id': int(record[0]),
                    'source': str(record[1]),
                    'score': float(record[2]),
                    'bb_xywh': np.array(record[3:7], dtype=int),
                    'landmarks': np.array(record[7:17], dtype=int).reshape(5, 2)
                }
                detection: Detection = DetectionFactory.create_detection(**d)

                frame_detections: FrameDetections | None = next(
                    (frame_detections for frame_detections in self.detections
                                      if frame_detections[0].frame_id == detection.frame_id), None)

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
        self.index = 0

    def frame_ids(self) -> list[int]:
        """Lists the frame ids within the track."""
        return sorted([elem.frame_id for elem in self.detections])

    def get_detection(self, frame_id: int) -> Detection:
        """Returns the Detection instance with the given frame id."""
        return next((detection for detection in self.detections
                               if detection.frame_id == frame_id))

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
        """Returns the distance between two Track instances.
        The first frame id is the earlier track's last detection.
        The second frame id is the coming up track's first detection.
        The distance is the substitude of the two frame ids.
        Overlapping tracks have 0 frame distances.

        Example:
            track1: [1, 2, 3]
            track2: [5, 6, 7]
            frame distance is 2
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
        if len(dets) < num: return dets
        return random.sample(dets, num)

    def center(self) -> np.ndarray:
        xs, ys = [], []
        for detection in self.detections:
            if not detection.is_interpolated:
                bb_midwh = xywh2midwh(detection.bb_xywh)
                xs.append(bb_midwh[0])
                ys.append(bb_midwh[1])
        return np.array([np.array(xs).mean(), np.array(ys).mean()])

    def bb_size(self, extra_percent: float = 0.2) -> int:
        ws, hs = [], []
        for detection in self.detections:
            if not detection.is_interpolated:
                ws.append(detection.bb_xywh[2])
                hs.append(detection.bb_xywh[3])
        return int(max(ws + hs) * (1 + extra_percent))

    def __len__(self) -> int:
        return len(self.detections)

    def __getitem__(self, index: int) -> Detection:
        return self.detections[index]

    def __str__(self) -> str:
        return f'ID {self.track_id} track with {len(self)} dets ' \
               f'from {self.first_detection().frame_id} to {self.last_detection().frame_id}.'

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.detections):
            value = self.detections[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration

    def save(self, output_file: PathType) -> None:
        """Saves the instance to a file.

        Args:
            output_file (PathType): path to the output file.
        """
        with open(output_file, 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(Detection.save_header())
            writer.writerows([detection.save_record() for detection in self.detections])

    def load(self, input_file: PathType) -> Self:
        """Loads the instance from a file.

        Args:
            input_file (PathType): path to the input file.

        Returns:
            Track: loaded instance.
        """
        with open(input_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')

            for line_count, record in enumerate(csv_reader):
                if line_count == 0: continue
                d = {
                    'frame_id': int(record[0]),
                    'source': str(record[1]),
                    'score': float(record[2]),
                    'bb_xywh': np.array(record[3:7], dtype=int),
                    'landmarks': np.array(record[7:17], dtype=int).reshape(5, 2)
                }
                detection: Detection = DetectionFactory.create_detection(**d)
                self.add(detection)

        return self