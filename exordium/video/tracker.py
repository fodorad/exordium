from abc import ABC, abstractmethod
import itertools
import logging
from typing import Callable, Self, Any
from pathlib import Path
from tqdm import tqdm
import numpy as np
from deepface import DeepFace
from exordium import PathType
from exordium.video.io import interpolate_1d, image2np
from exordium.video.bb import iou_xywh
from exordium.video.detection import DetectionFactory, Detection, DetectionFromImage, VideoDetections, Track


class Tracker(ABC):

    def __init__(self, verbose: bool = False):
        self.new_track_id: int = 0
        self.tracks: dict[int, Track] = {}
        self._selected_tracks: dict[int, Track] = {}
        self.path_to_id: Callable[[str], int] = lambda p: int(Path(p).stem)
        self.id_to_path: Callable[[int], str] = lambda i: f'{i:06d}.png'
        self.verbose = verbose

    @property
    def selected_tracks(self):
        return self.tracks if len(self._selected_tracks) == 0 else self._selected_tracks

    #Currently iou is supported
    #@abstractmethod
    #def label(self, **kwargs):
    #    """Connects the bounding boxes over multiple frames to form face tracks."""

    def label(self, detections: VideoDetections,
                    min_score: float = 0.7,
                    max_lost: int = 30,
                    iou_threshold: float = 0.2) -> Self:

        # iterate over frame-wise detections
        for frame_detections in tqdm(detections, total=len(detections), desc="Label tracks", disable=not self.verbose):

            # iterate over detections within a frame
            for detection in frame_detections:

                # ignore low confidence detections
                if detection.score < min_score: continue

                candidate_tracks = []
                for _, track in self.tracks.items():
                    # get track's last timestep's detection
                    track_last_detection: Detection = track.last_detection()

                    # calculate iou of the detection and the track's last detection, and the frame distance
                    iou: float = iou_xywh(track_last_detection.bb_xywh, detection.bb_xywh)
                    frame_distance: int = abs(detection.frame_id - track_last_detection.frame_id)

                    # last frame is within lost frame tolerance and IoU threshold is met
                    if frame_distance < max_lost and iou > iou_threshold:
                        candidate_tracks.append((track.track_id, iou))

                # if condition is met, assign detection to an unfinished track
                if candidate_tracks:
                    # get the track with the highest iou
                    best_candidate_id, _ = sorted(candidate_tracks, key=lambda x: x[1], reverse=True)[0]
                    # add the detection to the end of the best track
                    self.tracks[best_candidate_id].add(detection)
                else:
                    # start new track
                    self.tracks[self.new_track_id] = Track(self.new_track_id, detection)
                    self.new_track_id += 1

        self = self._interpolate_detections()
        return self

    @abstractmethod
    def merge_rule(self, track_1: Track, track_2: Track) -> tuple[bool, Track, Track]:
        """Merge rule for two tracks.

        Args:
            track_1 (Track): 1st Track instance in comparison.
            track_2 (Track): 2nd Track instance in comparison.

        Returns:
            tuple[bool, Track, Track]: whether merge or not, keep track, drop track.
        """

    def merge(self) -> Self:
        if len(self.tracks) <= 1: return self

        track_ids = list(self.tracks.keys())
        blacklist = set()

        for track_id1 in track_ids:

            if track_id1 in blacklist:
                continue

            track_1: Track = self.tracks[track_id1]

            for track_id2 in track_ids:

                if track_id1 == track_id2 or track_id2 in blacklist:
                    continue

                track_2: Track = self.tracks[track_id2]

                is_merge, track_keep, track_drop = self.merge_rule(track_1, track_2)

                if is_merge:
                    track_keep.merge(track_drop)
                    self.tracks.pop(track_drop.track_id)
                    blacklist.add(track_drop.track_id)

            blacklist.add(track_id1)
        return self

    def _interpolate_detections(self) -> Self:

        for _, track in self.tracks.items():
            frame_ids = np.array(track.frame_ids())

            # find missing frames
            indices = np.where(np.diff(frame_ids) > 1)[0][::-1]
            for ind in indices:
                # bounding box interp from frame_id_start to frame_id_end
                frame_id_start = frame_ids[ind]
                frame_id_end = frame_ids[ind + 1]

                # get start and end detections
                detection_start = track.get_detection(frame_id_start)
                detection_end = track.get_detection(frame_id_end)

                # interpolate bb coords
                new_bb = np.rint(interpolate_1d(frame_id_start, frame_id_end, detection_start.bb_xywh, detection_end.bb_xywh)).astype(int)

                # interpolate lmks coords
                new_landmarks_x = interpolate_1d(frame_id_start, frame_id_end, detection_start.landmarks[:, 0], detection_end.landmarks[:, 0])
                new_landmarks_y = interpolate_1d(frame_id_start, frame_id_end, detection_start.landmarks[:, 1], detection_end.landmarks[:, 1])
                new_landmarks = np.rint(np.stack([new_landmarks_x, new_landmarks_y], axis=2)).astype(int) # (B, N) -> (B, N, 2)

                # add interpolated detections to the tracks
                new_frame_ids = np.arange(frame_id_start + 1, frame_id_end, 1)
                for ind, frame_id in enumerate(new_frame_ids):
                    # set the source of the new Detection
                    source: str | np.ndarray = str(Path(detection_start.source).parent / self.id_to_path(frame_id)) \
                        if isinstance(detection_start, DetectionFromImage) else detection_start.source

                    # source frame or video does not exist, then skip it entirely.
                    if isinstance(source, str) and not Path(source).exists():
                        logging.info(f'Interpolation does not exist for source file {source}')
                        continue

                    new_detection = {
                        'frame_id': int(frame_id),
                        'source': source,
                        'score': -1,
                        'bb_xywh': new_bb[ind,:], # (B, 4) -> (4,)
                        'landmarks': new_landmarks[ind,...], # (B, N, 2) -> (N, 2)
                    }
                    track.add(DetectionFactory.create_detection(**new_detection))

        return self

    def select_long_tracks(self, min_length: int = 250) -> Self:
        self._selected_tracks = {
            track_id: track
            for track_id, track in self.selected_tracks.items()
            if len(track) > min_length
        }
        return self

    def select_topk_long_tracks(self, top_k: int = 1) -> Self:
        sorted_tracks: list[tuple[int, Track]] = sorted(
            [(track_id, track) for track_id, track in self.selected_tracks.items()],
            key=lambda x: len(x[1]),
            reverse=True
        )
        self._selected_tracks = {track_id: track for track_id, track in sorted_tracks[:top_k]}
        return self

    def select_topk_biggest_bb_tracks(self, top_k: int = 1) -> Self:
        sorted_tracks: list[tuple[int, Track]] = sorted(
            [(track_id, track) for track_id, track in self.selected_tracks.items()],
            key=lambda x: x[1].bb_size(),
            reverse=True
        )
        self._selected_tracks = {track_id: track for track_id, track in sorted_tracks[:top_k]}
        return self

    def get_center_track(self) -> Track:
        return sorted([track for _, track in self.selected_tracks.items()],
                       key=lambda x: np.linalg.norm(x.center() - x.first_detection().frame_center()))[0]


class IouTracker(Tracker):
    "Intersection over Union based tracker for quick object tracking without massive movements."

    def __init__(self, verbose: bool = False, max_lost: int = -1, iou_threshold: float = 0.2):
        """IoU tracker constructor.

        Args:
            verbose (bool, optional): show information. Defaults to False.
            max_lost (int, optional): maximum acceptable frame distance between tracks.
                                      -1 means that it any number of frames between two tracks are accepted. Defaults to -1.
            iou_threshold (float, optional): intersection over union threshold of two detections from the tracks. Defaults to 0.2.
        """
        super().__init__(verbose)
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def merge_rule(self, track_1: Track, track_2: Track) -> tuple[bool, Track, Track]:
        """Merge tracks if they are within the lost frame tolerance and IoU threshold is met.

        Args:
            track_1 (Track): 1st Track instance in comparison.
            track_2 (Track): 2nd Track instance in comparison.

        Returns:
            tuple[bool, Track, Track]: whether merge or not, keep track, drop track.
        """
        no1, no2 = (track_1, track_2) if track_1.is_started_earlier(track_2) else (track_2, track_1)
        is_max_lost: bool = (self.max_lost == -1) or (no1.frame_distance(no2) <= self.max_lost)
        is_iou_threshold: bool = iou_xywh(no1.last_detection().bb_xywh, no2.first_detection().bb_xywh) > self.iou_threshold
        return (is_max_lost and is_iou_threshold), no1, no2


class DeepFaceTracker(Tracker):
    "DeepFace based tracker for face tracking without massive movements and dissimilar faces."

    def __init__(self, model_name: str = 'VGG-Face', verbose: bool = False, sample: int = 5, score_threshold: float = 0.85):
        """DeepFace tracker constructor.

        Args:
            verbose (bool, optional): show information. Defaults to False.
            sample (int, optional): number of faces selected randomly from the tracks for comparison. Defaults to 5.
            score_threshold (float, optional): mean verification score treshold to determine similarity. Defaults to 0.85.
        """
        super().__init__(verbose)
        self.sample = sample
        self.score_threshold = score_threshold

        if not model_name in {"VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"}:
            raise ValueError(f'Invalid DeepFace model name {model_name}.')

        self.model_name = model_name

    def merge_rule(self, track_1: Track, track_2: Track) -> tuple[bool, Track, Track]:
        """Merge tracks if the faces within the tracks are similar.

        Args:
            track_1 (Track): 1st Track instance in comparison.
            track_2 (Track): 2nd Track instance in comparison.

        Returns:
            tuple[bool, Track, Track]: whether merge or not, keep track, drop track.
        """
        no1, no2 = (track_1, track_2) if track_1.is_started_earlier(track_2) else (track_2, track_1)
        detections_1: list[Detection] = no1.sample(self.sample)
        detections_2: list[Detection] = no2.sample(self.sample)

        # all pairwise combination of the faces within the two tracks
        pairs = list(itertools.product(list(range(len(detections_1))), list(range(len(detections_2)))))

        # verify face pairs
        deepface_response: list[dict[str, Any]] = [
            DeepFace.verify(img1_path=image2np(detections_1[index_1].bb_crop(), 'BGR'),
                            img2_path=image2np(detections_2[index_2].bb_crop(), 'BGR'),
                            model_name=self.model_name,
                            detector_backend='skip')
            for index_1, index_2 in pairs
        ]
        mean_verification_score = np.array([d['verified'] for d in deepface_response], dtype=float).mean()
        return (mean_verification_score > self.score_threshold), no1, no2