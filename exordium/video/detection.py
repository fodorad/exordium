import csv
import random
import itertools
from pathlib import Path
from typing import Callable

from batch_face import RetinaFace  # pip install git+https://github.com/elliottzheng/batch-face.git@master
from deepface import DeepFace
import cv2
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from scipy.interpolate import interp1d  # type: ignore

from exordium.utils.shared import load_or_create
from exordium.video.bb import xywh2xyxy, xyxy2xywh, xywh2midwh, crop_mid, iou_xywh
from exordium.video.frames import frames2video


class Detection:

    def __init__(self, frame_id: int, frame_path: str, score: float, bb_xywh: np.ndarray, bb_xyxy: np.ndarray, landmarks: np.ndarray):
        self.frame_id: int = frame_id
        self.frame_path: str = frame_path
        self.score: float = score
        self.bb_xywh: np.ndarray = bb_xywh # (4,)
        self.bb_xyxy: np.ndarray = bb_xyxy # (4,)
        self.landmarks: np.ndarray = landmarks # (5,2)
    
    def frame(self) -> np.ndarray:
        assert Path(self.frame_path).suffix in {'png', 'jpg'}
        return cv2.imread(self.frame_path)

    def crop(self) -> np.ndarray:
        assert Path(self.frame_path).exists(
        ), f'Image or video does not exist at {self.frame_path}.'
        if Path(self.frame_path).suffix == '.mp4':
            vr = VideoReader(str(self.frame_path), ctx=cpu(0))
            frame = vr[self.frame_id].asnumpy() # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.imread(self.frame_path) # BGR
        midwh: np.ndarray = xywh2midwh(self.bb_xywh)
        bb_size: int = max(self.bb_xywh[2:])
        image_crop = crop_mid(frame, midwh[:2], bb_size)
        return image_crop

    def frame_center(self) -> np.ndarray:
        assert Path(self.frame_path).exists(
        ), f'Image or video does not exist at {self.frame_path}.'
        if Path(self.frame_path).suffix == '.mp4':
            vr: VideoReader = VideoReader(str(self.frame_path), ctx=cpu(0))
            frame = vr[0].asnumpy(
            )  # frame size is consistent, index do not matter
        else:
            frame = cv2.imread(self.frame_path)
        height, width, _ = frame.shape
        return np.array([width // 2, height // 2]).astype(int)

    def __eq__(self, other: 'Detection') -> bool:
        return self.frame_id == other.frame_id and \
               self.frame_path == other.frame_path and \
               abs(self.score - other.score) < 1e-6 and \
               np.all(np.equal(self.bb_xyxy, other.bb_xyxy)) and \
               np.all(np.equal(self.bb_xywh, other.bb_xywh)) and \
               np.all(np.equal(self.landmarks, other.landmarks))


class FrameDetections:

    def __init__(self):
        """FrameDetections class represent face bounding boxes within a single frame.

        The following keys per detection are available:
            frame_path (str): frame path for the detections.
            score (float): detector confidence value between [0..1].
            bb_xywh (list[int]): bounding box in a form of xywh, where xy are pixels coordinates (int), wh are width and height pixel distances (int).
            bb_xyxy (list[int]): bounding box in a form of xyxy, where the first xy are top left pixels coordinates (int) and the second xy are bottom right pixel coordinates (int).
            landmarks (np.ndarray): xy pixel coordinates of the face landmarks (left eye, right eye, nose, left mouth, right mouth). Expected shape is (5, 2).
        """
        self.detections: list[Detection] = []
        self.index = 0

    def add_dict(self, detection: dict) -> None:
        for key in ['frame_id', 'frame_path', 'score', 'bb_xywh', 'bb_xyxy', 'landmarks']:
            assert key in list(detection.keys()), \
                f'Invalid detection dict. Missing key: {key}'
        self.detections.append(Detection(**detection))

    def add_detection(self, detection: Detection) -> None:
        self.detections.append(detection)

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

    def __eq__(self, other: 'FrameDetections') -> bool:
        for fdet1, fdet2 in zip(self.detections, other.detections):
            if fdet1 != fdet2:
                return False
        return True

    def get_biggest_bb(self):
        return sorted(self.detections,
                      key=lambda d: np.max(d.bb_xywh[2:]),
                      reverse=True)[0]

    def get_highest_score(self):
        return sorted(self.detections, key=lambda d: d.score, reverse=True)[0]

    def save(self, output_file: str):

        with open(output_file, 'w') as f:
            writer = csv.writer(f,
                                delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow([
                'frame_id', 'frame_path', 'score', 'x', 'y', 'w', 'h',
                'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y',
                'nose_x', 'nose_y', 'left_mouth_x', 'left_mouth_y',
                'right_mouth_x', 'right_mouth_y'
            ])

            for detection in self.detections:
                writer.writerow([
                    detection.frame_id, detection.frame_path, detection.score,
                    *detection.bb_xywh, *detection.landmarks.flatten()
                ])

    def save_format(self) -> tuple[list[str], list[str]]:
        names = [
            'frame_id', 'frame_path', 'score', 'x', 'y', 'w', 'h',
            'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y', 'nose_x',
            'nose_y', 'left_mouth_x', 'left_mouth_y', 'right_mouth_x',
            'right_mouth_y'
        ]

        values = []
        for detection in self.detections:
            values.append([
                detection.frame_id, detection.frame_path, detection.score,
                *detection.bb_xywh, *detection.landmarks.flatten()
            ])
        return names, values

    def load(self, input_file: str):

        with open(input_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')

            for line_count, record in enumerate(csv_reader):
                if line_count > 0:
                    d = {
                        'frame_id': int(record[0]),
                        'frame_path': str(record[1]),
                        'score': float(record[2]),
                        'bb_xywh': np.array(record[3:7], dtype=int),
                        'bb_xyxy': xywh2xyxy(np.array(record[3:7], dtype=int)),
                        'landmarks': np.array(record[7:17], dtype=int).reshape(5, 2)
                    }
                    detection = Detection(**d)
                    self.detections.append(detection)

        return self

    def add_detections_to_frame(self, frame: np.ndarray | None) -> np.ndarray:
        if frame is None: frame = cv2.imread(self.detections[0].frame_path)
        cv2.putText(frame, f"frame id: {self.detections[0].frame_id:06d}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for detection in self.detections:
            cv2.putText(frame, f"score: {detection.score:.2f}", (detection.bb_xyxy[0] - 5, detection.bb_xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (detection.bb_xyxy[0], detection.bb_xyxy[1]), (detection.bb_xyxy[2], detection.bb_xyxy[3]), (0, 255, 0), 2)
        return frame


class VideoDetections():

    def __init__(self):
        """VideoDetections class represent face bounding boxes within multiple frames from a single video.
        """
        self.detections: list[FrameDetections] = []
        self.index = 0

    def add(self, fdet: FrameDetections) -> None:
        # add FrameDetection only if it is not an empty.
        if len(fdet) > 0:
            self.detections.append(fdet)
    
    def merge(self, vdet: 'VideoDetections') -> None:
        for fdet in vdet:
            self.add(fdet)
    
    def frame_ids(self) -> list[int]:
        return [fdet[0].frame_id for fdet in self.detections]
    
    def get_frame_detection(self, frame_id: int) -> FrameDetections:
        return next((frame_detection for frame_detection in self.detections
                     if frame_detection[0].frame_id == frame_id))

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

    def __eq__(self, other: 'VideoDetections') -> bool:
        for fdet1, fdet2 in zip(self.detections, other.detections):
            if fdet1 != fdet2: return False
        return True

    def save(self, output_file: str):

        names, _ = self.detections[0].save_format()
        with open(output_file, 'w') as f:
            writer = csv.writer(f,
                                delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(names)

            for frame_detections in self.detections:
                _, values = frame_detections.save_format()
                for value in values:
                    writer.writerow(value)

    def load(self, input_file: str):

        with open(input_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')

            for line_count, record in enumerate(csv_reader):
                if line_count > 0:
                    d = {
                        'frame_id': int(record[0]),
                        'frame_path': str(record[1]),
                        'score': float(record[2]),
                        'bb_xywh': np.array(record[3:7], dtype=int),
                        'bb_xyxy': xywh2xyxy(np.array(record[3:7], dtype=int)),
                        'landmarks': np.array(record[7:17], dtype=int).reshape(5, 2)
                    }
                    detection = Detection(**d)

                    frame_detections: FrameDetections = next((
                        frame_detections
                        for frame_detections in self.detections
                        if frame_detections[0].frame_id == detection.frame_id),
                                                             None)
                    if frame_detections is None:
                        frame_detections = FrameDetections()
                        frame_detections.add_detection(detection)
                        self.detections.append(frame_detections)
                    else:
                        frame_detections.add_detection(detection)

        return self

    def save_detections_to_video(self, frame_dir: str | Path, output_dir: str | Path, fps: int = 30, sample_every_n: int = 1, verbose: bool = False) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = sorted(list(Path(frame_dir).iterdir()))
        frame_ids = [int(Path(frame_path).stem) for frame_path in frame_paths]
        frame_detection_ids = self.frame_ids()
        for frame_id, frame_path in tqdm(zip(frame_ids, frame_paths), total=len(frame_ids), desc='Save frames', disable=not verbose):
            frame = cv2.imread(str(frame_path))
            if frame_id in frame_detection_ids:
                frame_detection = self.get_frame_detection(frame_id)
                frame = frame_detection.add_detections_to_frame(frame)
            cv2.imwrite(str(output_dir / f'{Path(frame_path).stem}.png'), frame)


class FaceDetector():

    def __init__(self,
                 gpu_id: int = 0,
                 batch_size: int = 30,
                 verbose: bool = False):
        """FaceDetector detects faces within frames.
        """
        self.detector = RetinaFace(gpu_id=gpu_id, network='resnet50')
        self.batch_size = batch_size
        self.verbose = verbose
        if self.verbose: print('RetinaFace is loaded.')

    @load_or_create('fdet')
    def detect_image(self, frame_path: str | Path | np.ndarray,
                     **kwargs) -> FrameDetections:
        if isinstance(frame_path, (str | Path)):
            image = cv2.imread(str(frame_path))
        else:
            assert isinstance(frame_path, np.ndarray) and frame_path.ndim == 3 and frame_path.shape[-1] == 3, \
                f'Invalid input image. Expected shape is (H,W,C), got instead {frame_path.shape}.'
            image = frame_path
        frame_dets: list[tuple[np.ndarray, np.ndarray,
                               float]] = self.detector([image], cv=True)[0]

        frame_detections = FrameDetections()
        # iterate over all the faces detected within a frame
        for face_det in frame_dets:
            bb_xyxy, landmarks, score = face_det
            bb_xyxy = np.rint(np.array(bb_xyxy)).astype(int)
            bb_xyxy = np.where(bb_xyxy < 0, np.zeros_like(bb_xyxy), bb_xyxy)
            bb_xywh = xyxy2xywh(bb_xyxy).astype(int)
            landmarks = np.rint(np.array(landmarks)).astype(int)
            frame_detections.add_dict({
                'frame_id': -1,
                'frame_path': str(frame_path),
                'score': score,
                'bb_xyxy': bb_xyxy,
                'bb_xywh': bb_xywh,
                'landmarks': landmarks
            })

        return frame_detections

    @load_or_create('vdet')
    def iterate_folder(self, frame_dir: str | Path,
                       **kwargs) -> VideoDetections:
        '''Iterate over ordered frames of a video in a single folder

        image:
            (0,0)---(0,w)
              |       |
              |       |
            (h,0)---(h,w)
                    
        bounding box:
            (x_min, y_min, x_max, y_max)
                   
        original detection format
            (y_min, x_min) -- (y_min, x_max)
                  |                 |
            (y_max, x_min) -- (y_max, x_max)
                    
        y is row (height), x is column (width)
        '''
        frame_paths = sorted(list(Path(frame_dir).iterdir()))
        video_detections = VideoDetections()

        for batch_ind in tqdm(range(0, len(frame_paths), self.batch_size),
                              desc='RetinaFace detection',
                              disable=not self.verbose):
            batch_frame_paths = frame_paths[batch_ind:batch_ind +
                                            self.batch_size]
            images = [
                cv2.imread(str(frame_path)) for frame_path in batch_frame_paths
            ]

            frame_dets: list[list[tuple[np.ndarray, np.ndarray,
                                        float]]] = self.detector(images,
                                                                 cv=True)

            # iterate over the frame detections
            for frame_det, frame_path in zip(frame_dets, batch_frame_paths):
                if len(frame_det) == 0:
                    continue  # skip frames without detection
                frame_detections = FrameDetections()
                # iterate over all the faces detected within a frame
                for face_det in frame_det:
                    bb_xyxy, landmarks, score = face_det
                    bb_xyxy = np.rint(np.array(bb_xyxy)).astype(int)
                    bb_xyxy = np.where(bb_xyxy < 0, np.zeros_like(bb_xyxy),
                                       bb_xyxy)
                    bb_xywh = xyxy2xywh(bb_xyxy).astype(int)
                    landmarks = np.rint(np.array(landmarks)).astype(int)
                    frame_detections.add_dict({
                        'frame_id':
                        int(Path(frame_path).stem),
                        'frame_path':
                        str(frame_path),
                        'score':
                        score,
                        'bb_xyxy':
                        bb_xyxy,
                        'bb_xywh':
                        bb_xywh,
                        'landmarks':
                        landmarks
                    })

                if len(frame_detections) > 0:
                    video_detections.add(frame_detections)

        return video_detections

    @load_or_create('vdet')
    def detect_video(self, video_path: str | Path,
                     **kwargs) -> VideoDetections:
        """Iterate over frames of a video
        """
        assert Path(
            video_path).exists(), f'Video does not exist at {str(video_path)}'
        vr = VideoReader(str(video_path), ctx=cpu(0))

        video_detections = VideoDetections()
        for batch_ind in tqdm(range(0, len(vr), self.batch_size),
                              desc='RetinaFace detection',
                              disable=not self.verbose):
            frame_indices = [
                ind for ind in range(batch_ind, batch_ind + self.batch_size)
                if ind < len(vr)
            ]
            images: np.ndarray = vr.get_batch(frame_indices).asnumpy()
            # if no face is detected, then empty list [] will be at that frame index
            frame_dets: list[list[tuple[np.ndarray, np.ndarray,
                                        float]]] = self.detector(images,
                                                                 cv=True)
            # iterate over the frame detections
            for frame_det, frame_id in zip(frame_dets, frame_indices):
                if len(frame_det) == 0:
                    continue  # skip frames without detection
                frame_detections = FrameDetections()
                # iterate over all the faces detected within a frame
                for face_det in frame_det:
                    bb_xyxy, landmarks, score = face_det
                    bb_xyxy = np.rint(np.array(bb_xyxy)).astype(int)
                    bb_xyxy = np.where(bb_xyxy < 0, np.zeros_like(bb_xyxy),
                                       bb_xyxy)
                    bb_xywh = xyxy2xywh(bb_xyxy).astype(int)
                    landmarks = np.rint(np.array(landmarks)).astype(int)
                    frame_detections.add_dict({
                        'frame_id': frame_id,
                        'frame_path': str(video_path),
                        'score': score,
                        'bb_xyxy': bb_xyxy,
                        'bb_xywh': bb_xywh,
                        'landmarks': landmarks
                    })
                video_detections.add(frame_detections)

        return video_detections


class Track():

    def __init__(self,
                 track_id: int | None = None,
                 detection: Detection | None = None,
                 verbose: bool = False) -> None:
        self.track_id = track_id
        if detection is None:
            self.detections: list[Detection] = []
        else:
            self.detections: list[Detection] = [detection]
        self.index = 0
        if verbose: print(f'Track {self.track_id} is started.')

    def frame_ids(self) -> list[int]:
        return sorted([elem.frame_id for elem in self.detections])

    def get_detection(self, frame_id: int) -> Detection:
        return next((detection for detection in self.detections
                     if detection.frame_id == frame_id))

    def __sort_detections(self) -> None:
        self.detections.sort(key=lambda obj: obj.frame_id)

    def add(self, detection: Detection) -> None:
        self.detections.append(detection)
        self.__sort_detections()

    def merge(self, detection: 'Track') -> None:
        self.detections += detection.detections
        self.__sort_detections()

    def track_frame_distance(self, track: 'Track') -> int:
        return abs(self.last_detection().frame_id -
                   track.first_detection().frame_id)

    def last_detection(self) -> Detection:
        return self.detections[-1]

    def first_detection(self) -> Detection:
        return self.detections[0]

    def sample(self, num: int = 5) -> list[Detection]:
        dets = [elem for elem in self.detections if elem.score != -1]
        if len(dets) < num: return dets
        return random.sample(dets, num)

    def center(self) -> np.ndarray:
        xs, ys = [], []
        for detection in self.detections:
            if detection.score != -1:  # not an interpolated detection
                bb_midwh = xywh2midwh(detection.bb_xywh)
                xs.append(bb_midwh[0])
                ys.append(bb_midwh[1])
        return np.array([np.array(xs).mean(), np.array(ys).mean()])

    def bb_size(self, extra_percent: float = 0.2) -> int:
        ws, hs = [], []
        for detection in self.detections:
            if detection.score != -1:  # not an interpolated detection
                ws.append(detection.bb_xywh[2])
                hs.append(detection.bb_xywh[3])
        return int(max(ws + hs) * (1 + extra_percent))

    def __len__(self) -> int:
        return len(self.detections)

    def __getitem__(self, index: int) -> Detection:
        return self.detections[index]

    def __str__(self) -> str:
        return f'ID {self.track_id} track with {len(self.detections)} dets ' \
               f'from {self.detections[0].frame_id} to {self.detections[-1].frame_id}.'

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.detections):
            value = self.detections[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration
    
    def __save_format(self) -> tuple[list[str], list[str]]:
        names = [
            'frame_id', 'frame_path', 'score', 'x', 'y', 'w', 'h',
            'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y', 'nose_x',
            'nose_y', 'left_mouth_x', 'left_mouth_y', 'right_mouth_x',
            'right_mouth_y'
        ]

        values = []
        for detection in self.detections:
            values.append([
                detection.frame_id, detection.frame_path, detection.score,
                *detection.bb_xywh, *detection.landmarks.flatten()
            ])
        return names, values
    
    def save(self, output_file: str):
        names, values = self.__save_format()
        with open(output_file, 'w') as f:
            writer = csv.writer(f,
                                delimiter=',',
                                quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(names)
            for value in values:
                writer.writerow(value)
    
    def load(self, input_file: str | Path):

        with open(str(input_file), 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')

            for line_count, record in enumerate(csv_reader):
                if line_count == 0: continue
                d = {
                    'frame_id': int(record[0]),
                    'frame_path': str(record[1]),
                    'score': float(record[2]),
                    'bb_xywh': np.array(record[3:7], dtype=int),
                    'bb_xyxy': xywh2xyxy(np.array(record[3:7], dtype=int)),
                    'landmarks': np.array(record[7:17], dtype=int).reshape(5, 2)
                }
                self.add(Detection(**d))

        return self

    def save_track_target_to_images(self, output_dir: str | Path, bb_size: int = 224, fps: int = 30, sample_every_n: int = 1, verbose: bool = False) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, detection in tqdm(enumerate(self.detections), desc='Save track targets', disable=not verbose):
            if i % sample_every_n != 0: continue
            image = detection.crop()
            if bb_size != -1:
                image = cv2.resize(image, (bb_size, bb_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(output_dir / f'{detection.frame_id:06d}.png'), image)

        frames2video(output_dir, output_dir.parent / f'{output_dir.stem}.mp4', fps)

    def save_track_with_context_to_video(self, frame_dir: str | Path, output_dir: str | Path, fps: int = 30, sample_every_n: int = 1, verbose: bool = False) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = sorted(list(Path(frame_dir).iterdir()))
        frame_ids = [int(Path(frame_path).stem) for frame_path in frame_paths]
        track_frame_ids = self.frame_ids()

        for frame_id, frame_path in tqdm(zip(frame_ids, frame_paths), total=len(frame_paths), desc='Save video frames', disable=not verbose):
            if frame_id % sample_every_n != 0: continue
            if frame_id not in track_frame_ids: continue

            frame = cv2.imread(str(frame_path))
            detection = self.get_detection(frame_id)

            cv2.putText(frame, f"frame id: {frame_id:06d}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"score: {detection.score:.2f}", (detection.bb_xyxy[0] - 5, detection.bb_xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (detection.bb_xyxy[0], detection.bb_xyxy[1]), (detection.bb_xyxy[2], detection.bb_xyxy[3]), (0, 255, 0), 2)
            cv2.imwrite(str(Path(output_dir) / f'{frame_id:06d}.png'), frame)

        frames2video(output_dir, output_dir.parent / f'{output_dir.stem}.mp4', fps)

class Tracker():

    def __init__(self):
        self.new_track_id: int = 0
        self.tracks: dict[int, Track] = {}
        self.selected_tracks: dict[int, Track] = {}
        self.path_to_id: Callable[..., int] = lambda p: int(Path(p).stem)
        self.id_to_path: Callable[..., str] = lambda i: f'{i:06d}.png'
        self.deepface = None

    def label_tracks_iou(self,
                         detections: VideoDetections,
                         iou_threshold: float = 0.2,
                         min_score: float = 0.7,
                         max_lost: int = 30,
                         verbose: bool = False) -> 'Tracker':

        # iterate over frame-wise detections
        for frame_detections in tqdm(detections,
                                     total=len(detections),
                                     desc="Label tracks, IoU",
                                     disable=not verbose):
            # iterate over detections within a frame
            for detection in frame_detections:
                # skip low confidence detection
                if detection.score < min_score: continue

                # initialize the very first track
                if len(self.tracks) == 0:
                    self.tracks[self.new_track_id] = Track(
                        self.new_track_id, detection)
                    self.new_track_id += 1
                    continue

                # current state of the dictionary, otherwise "RuntimeError: dictionary changed size during iteration"
                #tracks = list(self.tracks.items()).copy()
                #tracks = copy.deepcopy(self.tracks)

                tracks_to_add: list[tuple[int, float]] = []
                # if condition is met, assign detection to an unfinished track
                for _, track in self.tracks.items():
                    # get track's last timestep's detection
                    track_last_detection: Detection = track.last_detection()

                    # same frame, different detection -> skip assignment
                    #if last_detection.frame_id == detection.frame_id:
                    #    #print(f'{detection["frame"]}: same frame, different detection')
                    #    continue

                    # calculate iou of the detection and the track's last detection, and the frame distance
                    iou: float = iou_xywh(track_last_detection.bb_xywh,
                                          detection.bb_xywh)
                    frame_distance: int = abs(detection.frame_id -
                                              track_last_detection.frame_id)

                    # last frame is within lost frame tolerance and IoU threshold is met
                    if frame_distance < max_lost and iou > iou_threshold:
                        # add detection to track
                        tracks_to_add.append((track.track_id, iou))
                        #print(f'{detection["frame"]}: track ({track.id}) is extended')

                if not tracks_to_add:
                    # start new track
                    self.tracks[self.new_track_id] = Track(
                        self.new_track_id, detection)
                    self.new_track_id += 1
                else:
                    # get good iou tracks, sort by iou and get the highest one
                    best_iou_track = sorted(tracks_to_add,
                                            key=lambda x: x[1],
                                            reverse=True)[0]
                    # add the detection to the end of the best track
                    self.tracks[best_iou_track[0]].add(detection)

        self = self.__interpolate()
        return self

    def __interpolate(self) -> 'Tracker':

        for _, track in self.tracks.items():
            # get sorted list of frame ids of the track
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
                bb_interp = interp1d(
                    np.array([frame_id_start, frame_id_end]),
                    np.array([detection_start.bb_xywh, detection_end.bb_xywh]).T)
                new_bb: np.ndarray = bb_interp(np.arange(frame_id_start, frame_id_end + 1, 1))

                # interpolate lmks coords
                lmks_x_interp = interp1d(
                    np.array([frame_id_start, frame_id_end]),
                    np.array([detection_start.landmarks[:, 0], detection_end.landmarks[:, 0]]).T)
                lmks_y_interp = interp1d(
                    np.array([frame_id_start, frame_id_end]),
                    np.array([detection_start.landmarks[:, 1], detection_end.landmarks[:, 1]]).T)
                new_lmks_x: np.ndarray = lmks_x_interp(np.arange(frame_id_start, frame_id_end + 1, 1))
                new_lmks_y: np.ndarray = lmks_y_interp(np.arange(frame_id_start, frame_id_end + 1, 1))

                # round and change type
                new_frame_ids = np.arange(frame_id_start + 1, frame_id_end, 1)
                new_bb = np.round(new_bb[:, 1:-1]).astype(int)
                new_lmks_x = np.round(new_lmks_x[:, 1:-1]).astype(int)
                new_lmks_y = np.round(new_lmks_y[:, 1:-1]).astype(int)

                # add interpolated detections to the tracks
                for ind, frame_id in enumerate(new_frame_ids):
                    new_detection = {
                        'frame_id': int(frame_id),
                        'frame_path': '',
                        'score': -1,
                        'bb_xywh': new_bb[:, ind],
                        'bb_xyxy': xywh2xyxy(new_bb[:, ind]),
                        'landmarks': np.stack([new_lmks_x[:, ind], new_lmks_y[:, ind]]).T,
                    }
                    track.add(Detection(**new_detection))

        return self
    
    def merge_deepface(self, sample: int = 5, threshold: float = 0.85, verbose: bool = False):
        if verbose: print('DeepFace verification started...')
        
        model_name = 'ArcFace'
        if self.deepface is None:
            # models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
            self.model = DeepFace.build_model(model_name)
            if verbose: print(model_name, 'is loaded.')

        track_ids = list(self.tracks.keys())
        if len(track_ids) == 1: return self

        blacklist = []
        for track_id1 in track_ids:
            if track_id1 in blacklist: continue

            for track_id2 in track_ids:
                if track_id1 == track_id2 or track_id2 in blacklist: continue

                # get two tracks
                track1: Track = self.tracks[track_id1]
                track2: Track = self.tracks[track_id2]
                
                # sample tracks
                detections_1: list[Detection] = track1.sample(sample)
                detections_2: list[Detection] = track2.sample(sample)

                # all pairwise combination of the faces within the two tracks
                pairs = list(itertools.product(list(range(len(detections_1))), list(range(len(detections_2)))))
                
                # load face pairs
                DF_res: list[dict] = []
                if verbose: print('')
                for i, (track_ind1, track_ind2) in enumerate(pairs):
                    if verbose: print(len(blacklist), '/', len(track_ids), '|', track_id1, 'vs', track_id2, '|', i+1, '/', len(pairs), '\r', end='', flush=True)
                    face1: np.ndarray = detections_1[track_ind1].crop()
                    face2: np.ndarray = detections_2[track_ind2].crop()
                    DF_res.append(DeepFace.verify(img1_path=face1,
                                                  img2_path=face2,
                                                  model_name=model_name,
                                                  enforce_detection=False,
                                                  detector_backend='skip'))
                mean_verification_score = np.array([d['verified'] for d in DF_res], dtype=np.float32).mean()

                if mean_verification_score > threshold:
                    # merge tracks
                    track1.merge(track2)
                    self.tracks.pop(track_id2)
                    blacklist.append(track_id2)
                    if verbose: print(len(blacklist), '/', len(track_ids), '|', f'{track_id1} & {track_id2} merged with {mean_verification_score} score', '\r', end='', flush=True)
                else:
                    if verbose: print(len(blacklist), '/', len(track_ids), '|', f'{track_id1} & {track_id2} not merged with {mean_verification_score} scores', '\r', end='', flush=True)

            blacklist.append(track_id1)
            if verbose: print(len(blacklist), '/', len(track_ids), '\r', end='', flush=True)

        return self

    def merge_iou(self,
                  max_lost: int = -1,
                  iou_threshold: float = 0.2) -> 'Tracker':
        # skip the merge step, if there is a single track in the video, or no tracks
        if len(self.tracks) <= 1: return self

        track_ids = list(self.tracks.keys())
        blacklist = []
        for track_id1 in track_ids:
            if track_id1 in blacklist: continue

            for track_id2 in track_ids:
                if track_id1 == track_id2 or track_id2 in blacklist: continue

                # get two tracks
                track_1: Track = self.tracks[track_id1]
                track_2: Track = self.tracks[track_id2]

                # get the last and first detections
                track_1_last_detection: Detection = track_1.last_detection()
                track_2_first_detection: Detection = track_2.first_detection()

                # condition check: -1 means that any number of frames between two tracks are accepted,
                # otherwise the max_lost number of frames or less are accepted
                is_max_lost: bool = (max_lost == -1) or (
                    track_1.track_frame_distance(track_2) <= max_lost)
                is_iou_threshold: bool = iou_xywh(
                    track_1_last_detection.bb_xywh,
                    track_2_first_detection.bb_xywh) > iou_threshold

                # merge tracks if within lost frame tolerance and IoU threshold is met
                if is_max_lost and is_iou_threshold:
                    track_1.merge(track_2)
                    self.tracks.pop(track_id2)
                    blacklist.append(track_id2)

            blacklist.append(track_id1)
        return self

    def select_long_tracks(self, min_length: int = 250) -> 'Tracker':
        self.selected_tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if len(track) > min_length
        }
        return self

    def select_topk_long_tracks(self, top_k: int = 1) -> 'Tracker':
        if len(self.selected_tracks) == 0:
            tracks = self.tracks
        else:
            tracks = self.selected_tracks

        tracks: list[tuple[int, Track]] = sorted(
            [(track_id, track) for track_id, track in tracks.items()],
            key=lambda x: len(x[1]),
            reverse=True)
        self.selected_tracks = {
            track_id: track
            for track_id, track in tracks[:top_k]
        }
        return self

    def select_topk_biggest_bb_tracks(self, top_k: int = 1) -> 'Tracker':
        if len(self.selected_tracks) == 0:
            tracks = self.tracks
        else:
            tracks = self.selected_tracks

        tracks: list[tuple[int, Track]] = sorted(
            [(track_id, track) for track_id, track in tracks.items()],
            key=lambda x: x[1].bb_size(),
            reverse=True)
        self.selected_tracks = {
            track_id: track
            for track_id, track in tracks[:top_k]
        }
        return self

    def get_center_track(self) -> Track:
        if len(self.selected_tracks) == 0:
            tracks = self.tracks
        else:
            tracks = self.selected_tracks

        return sorted([track for _, track in tracks.items()],
                      key=lambda x: np.linalg.norm(x.center(
                      ) - x.first_detection().frame_center()))[0]


@load_or_create('pkl')
def center_face_track(frame_dir: str | Path,
                      min_det_score: float = 0.7,
                      bb_iou_thr: float = 0.2,
                      interp_max_lost: int = 30,
                      merge_algorithm: str = 'iou',
                      merge_max_lost: int = -1,
                      merge_iou_thr: float = 0.2,
                      deepface_sample: int = 5,
                      deepface_threshold: float = 0.85,
                      select_min_length: int = 1,
                      select_track_priority: str = 'bb_size',
                      batch_size: int = 32,
                      gpu_id: int = 0,
                      retinaface_arch: str = 'resnet50',
                      verbose: bool = False,
                      cache_vdet: str = 'test.vdet',
                      **kwargs):

    assert merge_algorithm in {'iou', 'deepface'}, \
        f'Invalid merge algorithm {merge_algorithm}. Choose one from "iou" or "deepface".'
    assert select_track_priority in {'bb_size', 'track_length'}, \
        f'Invalid priority {select_track_priority}. Choose one from "bb_size" or "track_length".'
    assert retinaface_arch in {'mobilenet', 'resnet50'}, \
        'Invalid architecture choice for RetinaFace. Choose from ["mobilenet","resnet50"].'

    print('Run face detector...')
    face_detector = FaceDetector(gpu_id=gpu_id,
                                 batch_size=batch_size,
                                 verbose=verbose)
    if Path(frame_dir).is_dir():
        video_detections = face_detector.iterate_folder(frame_dir=frame_dir,
                                                        output_path=cache_vdet)
    else:  # video
        video_detections = face_detector.detect_video(video_path=frame_dir,
                                                      output_path=cache_vdet)

    print('Run tracker...')
    # label and interpolate tracks
    tracker = Tracker().label_tracks_iou(video_detections, min_score=min_det_score, iou_threshold=bb_iou_thr, max_lost=interp_max_lost) \
    
    # merge tracks
    if merge_algorithm == 'iou':
        tracker.merge_iou(max_lost=merge_max_lost, iou_threshold=merge_iou_thr)
    else: # deepface
        tracker.merge_deepface(sample=deepface_sample, threshold=deepface_threshold, verbose=verbose)

    if select_min_length > 1:
        tracker.select_long_tracks(select_min_length)

    # select the topk longest tracks, and then a single center track
    if select_track_priority == 'track_length':
        track: Track = tracker.select_topk_long_tracks(top_k=2) \
                              .get_center_track()
    else: # biggest bounding box is preferred
        track: Track = tracker.select_topk_biggest_bb_tracks(top_k=1) \
                              .get_center_track()

    print('[Track]\n\tnumber of tracks:', len(tracker.tracks),
          '\n\tlengths of tracks:', [(id, len(track))
                                     for id, track in tracker.tracks.items()],
          '\n\tlength of selected center track:', len(track))

    return track


if __name__ == '__main__':

    video1 = 'data/videos/9KAqOrdiZ4I.001.mp4'
    video2 = 'data/videos/002003_FC1_A.mp4'
    video3 = 'data/videos/multispeaker_720p.mp4'
    video4 = 'data/videos/multispeaker_360p.mp4'

    ct = center_face_track(frame_dir=video4,
                           verbose=True,
                           cache_vdet=f'data/processed/cache/{Path(video4).stem}.vdet')

    ct.save_track_with_context_to_video(frame_dir=f'data/processed/frames/{Path(video4).stem}',
                                        output_dir=f'data/processed/tracks/{Path(video4).stem}_{str(ct.track_id)}_context/frames',
                                        fps=30, sample_every_n=30)

    ct.save_track_target_to_images(output_dir=f'data/processed/tracks/{Path(video4).stem}_{str(ct.track_id)}_target/frames',
                                   fps=30, sample_every_n=30)