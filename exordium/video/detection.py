import random
from pathlib import Path
from typing import Callable, Union
import csv
from dataclasses import dataclass

from batch_face import RetinaFace  # pip install git+https://github.com/elliottzheng/batch-face.git@master
import cv2
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from scipy.interpolate import interp1d # type: ignore
#import bbox_visualizer as bbv
#from deepface import DeepFace
#import torch
#import torchvision

from exordium.utils.shared import load_or_create
from exordium.video.bb import xywh2xyxy, xyxy2xywh, xywh2midwh, crop_mid, iou_xywh


@dataclass
class Detection:
    frame_id: int
    frame_path: str
    score: float
    bb_xywh: np.ndarray # (4,)
    bb_xyxy: np.ndarray # (4,)
    landmarks: np.ndarray # (5,2)

    def crop(self) -> np.ndarray:
        assert Path(self.frame_path).exists(), f'Image or video does not exist at {self.frame_path}.'
        if Path(self.frame_path).suffix == '.mp4':
            vr = VideoReader(str(self.frame_path), ctx=cpu(0))
            frame = vr[self.frame_id].asnumpy()
        else:
            frame = cv2.imread(self.frame_path)
        midwh: np.ndarray = xywh2midwh(self.bb_xywh)[0]
        bb_size: int = max(self.bb_xywh[2:])
        image_crop = crop_mid(frame, midwh[:2], bb_size)
        return image_crop

    def frame_center(self) -> np.ndarray:
        assert Path(self.frame_path).exists(), f'Image or video does not exist at {self.frame_path}.'
        if Path(self.frame_path).suffix == '.mp4':
            vr: VideoReader = VideoReader(str(self.frame_path), ctx=cpu(0))
            frame = vr[0].asnumpy() # frame size is consistent, index do not matter
        else:
            frame = cv2.imread(self.frame_path)
        height, width, _ = frame.shape
        return np.array([width // 2, height // 2]).astype(int)

    def __eq__(self, other: 'Detection') -> bool:
        if not isinstance(other, Detection): return False
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
        for key in Detection.__annotations__.keys():
            assert key in detection.keys(), f'Invalid detection dict. Missing key: {key}'
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
        if not isinstance(other, FrameDetections): return False
        for fdet1, fdet2 in zip(self.detections, other.detections):
            if fdet1 != fdet2:
                return False
        return True

    def get_biggest_bb(self):
        return sorted(self.detections,
                      key=lambda d: np.max(d.bb_xywh[2:]),
                      reverse=True)[0]

    def get_highest_score(self):
        return sorted(self.detections,
                      key=lambda d: d.score,
                      reverse=True)[0]

    def save(self, output_file: str):

        with open(output_file, 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['frame_id', 'frame_path', 'score',
                             'x', 'y', 'w', 'h',
                             'left_eye_x', 'left_eye_y',
                             'right_eye_x', 'right_eye_y',
                             'nose_x', 'nose_y',
                             'left_mouth_x', 'left_mouth_y',
                             'right_mouth_x', 'right_mouth_y'])

            for detection in self.detections:
                writer.writerow([detection.frame_id,
                                 detection.frame_path,
                                 detection.score,
                                 *detection.bb_xywh,
                                 *detection.landmarks.flatten()])

    def save_format(self) -> tuple[list[str], list[str]]:
        names = ['frame_id', 'frame_path', 'score',
                 'x', 'y', 'w', 'h',
                 'left_eye_x', 'left_eye_y',
                 'right_eye_x', 'right_eye_y',
                 'nose_x', 'nose_y',
                 'left_mouth_x', 'left_mouth_y',
                 'right_mouth_x', 'right_mouth_y']

        values = []
        for detection in self.detections:
            values.append([detection.frame_id,
                           detection.frame_path,
                           detection.score,
                           *detection.bb_xywh,
                           *detection.landmarks.flatten()])
        return names, values

    def load(self, input_file: str):

        with open(input_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')

            for line_count, record in enumerate(csv_reader):
                if line_count > 0:
                    d = {'frame_id': int(record[0]),
                         'frame_path': str(record[1]),
                         'score': float(record[2]),
                         'bb_xywh': np.array(record[3:7], dtype=int),
                         'bb_xyxy': xywh2xyxy(np.array(record[3:7], dtype=int)),
                         'landmarks': np.array(record[7:17], dtype=int).reshape(5,2)}
                    detection = Detection(**d)
                    self.detections.append(detection)

        return self


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
        if not isinstance(other, VideoDetections): return False
        for index, (fdet1, fdet2) in enumerate(zip(self.detections, other.detections)):
            if fdet1 != fdet2:
                return False
        return True

    def save(self, output_file: str):
    
        names, _ = self.detections[0].save_format()
        with open(output_file, 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
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
                    d = {'frame_id': int(record[0]),
                         'frame_path': str(record[1]),
                         'score': float(record[2]),
                         'bb_xywh': np.array(record[3:7], dtype=int),
                         'bb_xyxy': xywh2xyxy(np.array(record[3:7], dtype=int)),
                         'landmarks': np.array(record[7:17], dtype=int).reshape(5,2)}
                    detection = Detection(**d)
                    
                    frame_detections: FrameDetections = next((frame_detections for frame_detections in self.detections 
                                                              if frame_detections[0].frame_id == detection.frame_id), None)
                    if frame_detections is None:
                        frame_detections = FrameDetections()
                        frame_detections.add_detection(detection)
                        self.detections.append(frame_detections)
                    else:
                        frame_detections.add_detection(detection)

        return self


class FaceDetector():

    def __init__(self, gpu_id: int = 0, batch_size: int = 30, verbose: bool = True):
        """FaceDetector detects faces within frames.
        """
        self.detector = RetinaFace(gpu_id=gpu_id, network='resnet50')
        self.batch_size = batch_size
        self.verbose = verbose
        if self.verbose: print('RetinaFace is loaded.')

    @load_or_create('fdet')
    def detect_image(self, frame_path: str | Path | np.ndarray, **kwargs) -> FrameDetections:
        if isinstance(frame_path, (str | Path)):
            image = cv2.imread(str(frame_path))
        else:
            assert isinstance(frame_path, np.ndarray) and frame_path.ndim == 3 and frame_path.shape[-1] == 3, \
                f'Invalid input image. Expected shape is (H,W,C), got instead {frame_path.shape}.'
            image = frame_path
        frame_dets: list[tuple[np.ndarray, np.ndarray, float]] = self.detector([image], cv=True)[0]

        frame_detections = FrameDetections()
        # iterate over all the faces detected within a frame
        for face_det in frame_dets:
            bb_xyxy, landmarks, score = face_det
            bb_xyxy = np.rint(np.array(bb_xyxy)).astype(int)
            bb_xyxy = np.where(bb_xyxy < 0, np.zeros_like(bb_xyxy), bb_xyxy)
            bb_xywh = xyxy2xywh(bb_xyxy).astype(int)
            landmarks = np.rint(np.array(landmarks)).astype(int)
            frame_detections.add_dict({'frame_id': -1,
                                       'frame_path': str(frame_path),
                                       'score': score,
                                       'bb_xyxy': bb_xyxy,
                                       'bb_xywh': bb_xywh,
                                       'landmarks': landmarks})

        return frame_detections

    @load_or_create('vdet')
    def iterate_folder(self, frame_dir: str | Path, **kwargs) -> VideoDetections:
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
                              desc='RetinaFace detection', disable=not self.verbose):
            batch_frame_paths = frame_paths[batch_ind:batch_ind+self.batch_size]
            images = [cv2.imread(str(frame_path)) for frame_path in batch_frame_paths]

            frame_dets: list[list[tuple[np.ndarray, np.ndarray, float]]] = self.detector(images, cv=True)
            
            # iterate over the frame detections
            for frame_det, frame_path in zip(frame_dets, batch_frame_paths):
                if len(frame_det) == 0: continue # skip frames without detection
                frame_detections = FrameDetections()
                # iterate over all the faces detected within a frame
                for face_det in frame_det:
                        bb_xyxy, landmarks, score = face_det
                        bb_xyxy = np.rint(np.array(bb_xyxy)).astype(int)
                        bb_xyxy = np.where(bb_xyxy < 0, np.zeros_like(bb_xyxy), bb_xyxy)
                        bb_xywh = xyxy2xywh(bb_xyxy).astype(int)
                        landmarks = np.rint(np.array(landmarks)).astype(int)
                        frame_detections.add_dict({'frame_id': int(Path(frame_path).stem),
                                                   'frame_path': str(frame_path),
                                                   'score': score,
                                                   'bb_xyxy': bb_xyxy,
                                                   'bb_xywh': bb_xywh,
                                                   'landmarks': landmarks})
            
                if len(frame_detections) > 0:
                    video_detections.add(frame_detections)

        return video_detections


    @load_or_create('vdet')
    def detect_video(self, video_path: str | Path, **kwargs) -> VideoDetections:
        """Iterate over frames of a video
        """
        assert Path(video_path).exists(), f'Video does not exist at {str(video_path)}'
        vr = VideoReader(str(video_path), ctx=cpu(0))

        video_detections = VideoDetections()
        for batch_ind in tqdm(range(0, len(vr), self.batch_size),
                              desc='RetinaFace detection', disable=not self.verbose):
            frame_indices = [ind for ind in range(batch_ind,batch_ind+self.batch_size) if ind < len(vr)]
            images: np.ndarray = vr.get_batch(frame_indices).asnumpy()
            # if no face is detected, then empty list [] will be at that frame index
            frame_dets: list[list[tuple[np.ndarray, np.ndarray, float]]] = self.detector(images, cv=True)
            # iterate over the frame detections
            for frame_det, frame_id in zip(frame_dets, frame_indices):
                if len(frame_det) == 0: continue # skip frames without detection
                frame_detections = FrameDetections()
                # iterate over all the faces detected within a frame
                for face_det in frame_det:
                    bb_xyxy, landmarks, score = face_det
                    bb_xyxy = np.rint(np.array(bb_xyxy)).astype(int)
                    bb_xyxy = np.where(bb_xyxy < 0, np.zeros_like(bb_xyxy), bb_xyxy)
                    bb_xywh = xyxy2xywh(bb_xyxy).astype(int)
                    landmarks = np.rint(np.array(landmarks)).astype(int)
                    frame_detections.add_dict({'frame_id': frame_id,
                                               'frame_path': str(video_path),
                                               'score': score,
                                               'bb_xyxy': bb_xyxy,
                                               'bb_xywh': bb_xywh,
                                               'landmarks': landmarks})
                video_detections.add(frame_detections)

        return video_detections


class Track():

    def __init__(self, track_id: int, detection: Detection, verbose: bool = False) -> None:
        self.track_id = track_id
        self.detections: list[Detection] = []
        self.add(detection)
        self.index = 0
        if verbose: print(f'track {self.track_id} is started')

    def frame_ids(self) -> list[int]:
        return sorted([elem.frame_id for elem in self.detections])

    def get_detection(self, frame_id: int) -> Detection:
        return next((detection for detection in self.detections if detection.frame_id == frame_id))

    def add(self, detection: Union[Detection,'Track']) -> None:
        if isinstance(detection, Detection):
            self.detections.append(detection)
        elif isinstance(detection, 'Track'):
            self.detections += detection.detections
        else:
            raise NotImplementedError()
        self.detections.sort(key=lambda obj: obj.frame_id)

    def track_frame_distance(self, track: 'Track') -> int:
        return abs(self.last_detection().frame_id - track.first_detection().frame_id)

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
                bb = detection.bb_xywh
                xs.append(bb[0] + bb[2] // 2)
                ys.append(bb[1] + bb[3] // 2)
        return np.array([np.array(xs).mean(), np.array(ys).mean()])

    def bb_size(self, extra_percent: float = 0.2) -> int:
        ws, hs = [], []
        for detection in self.detections:
            if detection.score != -1:  # not an interpolated detection
                bb = detection.bb_xywh
                ws.append(bb[2])
                hs.append(bb[3])
        return int(max(ws + hs) * (1 + extra_percent))

    def __len__(self) -> int:
        return len(self.detections)

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


class Tracker():

    def __init__(self):
        self.new_track_id: int = 0
        self.tracks: dict[int,Track] = {}
        self.selected_tracks: dict[int,Track] = {}
        self.path_to_id: Callable[..., int] = lambda p: int(Path(p).stem)
        self.id_to_path: Callable[..., str] = lambda i: f'{i:06d}.png'

    def label_tracks_iou(self,
                         detections: VideoDetections,
                         iou_threshold: float = 0.2,
                         min_score: float = 0.7,
                         max_lost: int = 30,
                         verbose: bool = False) -> 'Tracker':

        # iterate over frame-wise detections
        for frame_detections in tqdm(detections, total=len(detections), desc="Label tracks, IoU", disable=not verbose):
            # iterate over detections within a frame
            for detection in frame_detections:
                # skip low confidence detection
                if detection.score < min_score: continue

                # initialize the very first track
                if len(self.tracks) == 0:
                    self.tracks[self.new_track_id] = Track(self.new_track_id, detection)
                    self.new_track_id += 1
                    continue

                # current state of the dictionary, otherwise "RuntimeError: dictionary changed size during iteration"
                # tracks = list(self.tracks.items()).copy()
                # tracks = copy.deepcopy(self.tracks)

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
                    iou: float = iou_xywh(track_last_detection.bb_xywh, detection.bb_xywh)
                    frame_distance: int = abs(detection.frame_id - track_last_detection.frame_id)

                    # last frame is within lost frame tolerance and IoU threshold is met
                    if frame_distance < max_lost and iou > iou_threshold:
                        # add detection to track
                        tracks_to_add.append((track.track_id, iou))
                        #print(f'{detection["frame"]}: track ({track.id}) is extended')

                if not tracks_to_add:
                    # start new track
                    self.tracks[self.new_track_id] = Track(self.new_track_id, detection)
                    self.new_track_id += 1
                else:
                    # get good iou tracks, sort by iou and get the highest one
                    best_iou_track = sorted(tracks_to_add, key=lambda x: x[1], reverse=True)[0]
                    # add the detection to the end of the best track
                    self.tracks[best_iou_track[0]].add(detection)

        self = self.interpolate()
        return self

    def interpolate(self) -> 'Tracker':

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

    '''
    def merge_deepface(self,
                       frames: list,
                       sample: int = 5,
                       threshold: float = 0.85):
        print('DeepFace verification started...')
        # models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
        model_name = 'ArcFace'
        model = DeepFace.build_model(model_name)
        print(model_name, 'is loaded.')

        track_ids = list(self.tracks.keys())
        if len(track_ids) == 1:
            return self

        blacklist = []
        for k in track_ids:

            if k in blacklist:
                continue

            for t in track_ids:

                if k == t or t in blacklist:
                    continue

                dets_k = self.tracks[k].sample(sample)
                dets_t = self.tracks[t].sample(sample)

                if dets_k is None or dets_t is None:
                    continue

                combs = list(
                    itertools.product(list(range(len(dets_k))),
                                      list(range(len(dets_t)))))
                frame_pairs = []
                print('')
                for i, (k_i, t_i) in enumerate(combs):
                    print(len(blacklist),
                          '/',
                          len(track_ids),
                          '|',
                          k,
                          'vs',
                          t,
                          '|',
                          i + 1,
                          '/',
                          len(combs),
                          '\r',
                          end='',
                          flush=True)
                    frame_id_k = dets_k[k_i][0]
                    frame_id_t = dets_t[t_i][0]
                    bb_k_xyxy = xywh2xyxy(dets_k[k_i][1]['bb'])
                    bb_t_xyxy = xywh2xyxy(dets_t[t_i][1]['bb'])
                    frame_k = cv2.imread(
                        frames[frame_id_k])[bb_k_xyxy[1]:bb_k_xyxy[3],
                                            bb_k_xyxy[0]:bb_k_xyxy[2], :]
                    frame_t = cv2.imread(
                        frames[frame_id_t])[bb_t_xyxy[1]:bb_t_xyxy[3],
                                            bb_t_xyxy[0]:bb_t_xyxy[2], :]
                    frame_pairs.append([frame_k, frame_t])

                DF_res = DeepFace.verify(img1_path=frame_pairs,
                                         model_name=model_name,
                                         model=model,
                                         enforce_detection=False,
                                         detector_backend='skip',
                                         prog_bar=False)

                mean_verification_score = np.array(
                    [v['verified'] for _, v in DF_res.items()],
                    dtype=np.float32).mean()

                if mean_verification_score > threshold:
                    # merge tracks
                    self.tracks[k].merge(self.tracks[t])
                    self.tracks.pop(t)
                    blacklist.append(t)
                    print(
                        len(blacklist),
                        '/',
                        len(track_ids),
                        '|',
                        f'{k} & {t} merged with {mean_verification_score} score',
                        '\r',
                        end='',
                        flush=True)
                else:
                    print(
                        len(blacklist),
                        '/',
                        len(track_ids),
                        '|',
                        f'{k} & {t} not merged with {mean_verification_score} scores',
                        '\r',
                        end='',
                        flush=True)

            blacklist.append(k)
            print(len(blacklist),
                  '/',
                  len(track_ids),
                  '\r',
                  end='',
                  flush=True)

        return self
    '''

    def merge_iou(self, max_lost: int = -1, iou_threshold: float = 0.2) -> 'Tracker':
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
                is_max_lost: bool = (max_lost == -1) or (track_1.track_frame_distance(track_2) <= max_lost)
                is_iou_threshold: bool = iou_xywh(track_1_last_detection.xywh, track_2_first_detection.xywh) > iou_threshold

                # merge tracks if within lost frame tolerance and IoU threshold is met
                if  is_max_lost and is_iou_threshold:
                    track_1.add(track_2)
                    self.tracks.pop(track_id2)
                    blacklist.append(track_id2)

            blacklist.append(track_id1)
        return self

    def select_long_tracks(self, min_length: int = 250) -> 'Tracker':
        self.selected_tracks = {track_id: track for track_id, track in self.tracks.items() 
                                if len(track) > min_length}
        return self

    def select_topk_long_tracks(self, top_k: int = 1) -> 'Tracker':
        tracks: list[tuple[int, Track]] = sorted([(track_id, track) for track_id, track in self.tracks.items()],
                                                 key=lambda x: len(x[1]), reverse=True)
        self.selected_tracks = {track_id: track for track_id, track in tracks[:top_k]}
        return self

    def get_center_track(self) -> Track:
        if len(self.selected_tracks) > 0:
            tracks = self.selected_tracks
        else:
            tracks = self.tracks
        return sorted([track for _, track in tracks.items()],
                      key=lambda x: np.linalg.norm(x.center() - x.first_detection().frame_center()))[0]

    @classmethod
    def save_track_faces(cls, track: Track, output_dir: str | Path, sample_every_n: int = 1) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, detection in tqdm(enumerate(track), total=len(track), desc='Save track faces'):
            if i % sample_every_n != 0: continue
            image = detection.crop()
            cv2.imwrite(str(output_dir / Path(detection.frame_path).name), image)

'''
def detection_visualization(frame_paths: str | list[str],
                            detections: RetinafaceDetections | str,
                            output_dir: str,
                            sample_every_n: int = 25,
                            only_detected: bool = False):

    if isinstance(frame_paths, str):
        frame_paths = sorted([
            str(Path(frame_paths) / elem) for elem in os.listdir(frame_paths)
        ])

    if isinstance(detections, str):
        detections = DetLoader().load(detections)

    if only_detected:
        ids = detections.ids()
    else:
        ids = list(
            range(int(Path(frame_paths[0]).stem),
                  int(Path(frame_paths[-1]).stem)))

    ids = ids[::sample_every_n]

    for id in tqdm(ids,
                   total=len(ids),
                   desc=f'Save detections to {output_dir}'):
        frame_path = next(
            filter(lambda frame_path: int(Path(frame_path).stem) == id,
                   frame_paths), None)
        if frame_path is None: continue
        frame = cv2.imread(frame_path)
        cv2.putText(frame, f"frame id: {id:02d}", (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        dets = [
            detection for detection in detections if detection['frame'] == id
        ]
        for det in dets:
            bb_xyxy = xywh2xyxy(det['bb'])
            cv2.putText(frame, "score: {:.2f}".format(det['score']),
                        (bb_xyxy[0] - 5, bb_xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (bb_xyxy[0], bb_xyxy[1]),
                          (bb_xyxy[2], bb_xyxy[3]), (0, 255, 0), 2)
            cv2.imwrite(str(Path(output_dir) / f'{id:06d}.png'), frame)


def track_visualization(frame_paths: str | list[str],
                        tracks: OrderedDict,
                        output_dir: str,
                        sample_every_n: int = 25):

    if isinstance(frame_paths, str):
        frame_paths = sorted([
            str(Path(frame_paths) / elem) for elem in os.listdir(frame_paths)
        ])

    ids = list(
        range(int(Path(frame_paths[0]).stem), int(Path(frame_paths[-1]).stem)))
    ids = ids[::sample_every_n]

    for id in tqdm(ids, total=len(ids), desc=f'Save tracks to {output_dir}'):

        frame_path = next(
            filter(lambda frame_path: int(Path(frame_path).stem) == id,
                   frame_paths), None)
        if frame_path is None: continue
        frame = cv2.imread(frame_path)

        for _, track in tracks.items():

            if id not in track.location:
                continue

            detection = track.location[id]
            bb_xyxy = xywh2xyxy(detection['bb'])
            frame = bbv.draw_rectangle(frame, bb_xyxy)
            frame = bbv.add_label(
                frame, "{}|{:2d}".format(track.id,
                                         int(detection['score'] * 100)),
                bb_xyxy)
            cv2.imwrite(str(Path(output_dir) / f'{id:06d}.png'), frame)


def face_visualization(frames: str | list[str],
                       tracks: OrderedDict,
                       output_dir: str,
                       sample_every_n: int = 1,
                       extra_percent: float = 0.2):

    assert frames, 'Empty list of frames'

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(frames, str):
        frames = sorted(
            [str(Path(frames) / elem) for elem in os.listdir(frames)])

    h, w, _ = cv2.imread(frames[0]).shape
    ids = list(range(int(Path(frames[0]).stem),
                     int(Path(frames[-1]).stem) + 1))
    ids = ids[::sample_every_n]

    for id in tqdm(ids,
                   total=len(ids),
                   desc=f'Save face RGB images to {output_dir}'):
        frame_path = next(
            filter(lambda frame_path: int(Path(frame_path).stem) == id,
                   frames), None)
        if frame_path is None: continue

        frame = cv2.imread(frame_path)

        for _, track in tracks.items():
            bb_size = track.bb_size(extra_percent=extra_percent)

            if bb_size % 2 == 1:
                bb_size -= 1  # make it even

            if id not in track.location:
                face = np.zeros((bb_size, bb_size, 3))
            else:
                detection = track.location[id]
                bb_xyxy = xywh2xyxy(detection['bb'])
                #print('detected bb:', bb_xyxy)
                # centering
                cx = bb_xyxy[0] + abs(bb_xyxy[2] - bb_xyxy[0]) // 2
                cy = bb_xyxy[1] + abs(bb_xyxy[3] - bb_xyxy[1]) // 2
                #print('center of bb:', cx, cy)
                face_bb_xyxy = np.rint(
                    np.array([
                        cx - bb_size // 2, cy - bb_size // 2,
                        cx + bb_size // 2, cy + bb_size // 2
                    ])).astype(np.int32)
                #print('calculated bb:', face_bb_xyxy)
                # correct if necessary
                face_bb_xyxy[face_bb_xyxy < 0] = 0
                face_bb_xyxy[face_bb_xyxy[0] > w] = w
                face_bb_xyxy[face_bb_xyxy[2] > w] = w
                face_bb_xyxy[face_bb_xyxy[1] > h] = h
                face_bb_xyxy[face_bb_xyxy[3] > h] = h
                # cut face
                x1, y1, x2, y2 = face_bb_xyxy
                #print('thresholded bb:', face_bb_xyxy)
                face = frame[y1:y2, x1:x2, :]

                if face.shape != (bb_size, bb_size, 3):
                    face_resized = np.zeros((bb_size, bb_size, 3))
                    sh, sw = (bb_size - face.shape[0]) // 2, (
                        bb_size - face.shape[1]) // 2
                    face_resized[sh:sh + face.shape[0],
                                 sw:sw + face.shape[1], :] = face
                    face = face_resized

            cv2.imwrite(str(output_dir / f'{id:06d}.png'), face)


@timer_with_return
@load_or_create('pkl')
def find_and_track_deepface(frames_dir: str | Path,
                            num_tracks: int = 2,
                            batch_size: int = 32,
                            gpu_id: int = 0,
                            retinaface_arch: str = 'resnet50',
                            cache_det: str = 'test.det',
                            **kwargs):
    """ Refactor for general use case

    Args:
        frames_dir (str): _description_
        num_tracks (int, optional): _description_. Defaults to 1.
        batch_size (int, optional): _description_. Defaults to 32.
        gpu_id (int, optional): _description_. Defaults to 0.
        overwrite (bool, optional): _description_. Defaults to True.
        retinaface_arch (str, optional): _description_. Defaults to 'resnet50'.
        cache_det (str, optional): _description_. Defaults to 'test.det'.
    """
    assert retinaface_arch in {
        'mobilenet', 'resnet50'
    }, 'Invalid architecture choice for RetinaFace. Choose from {"mobilenet","resnet50"}.'

    # get frames
    frames = [
        str(Path(frames_dir) / frame)
        for frame in sorted(os.listdir(frames_dir))
    ]

    # detect face bounding boxes
    detections = detect_faces(frame_paths=frames,
                              detector=RetinaFace(gpu_id=gpu_id,
                                                  network=retinaface_arch),
                              batch_size=batch_size,
                              output_path=cache_det)

    #if visualize_det:
    #    Path(./).mkdir(parents=True, exist_ok=True)
    #    detection_visualization(frames, detections, detection_dir)
    #    frames2video(detection_dir, Path(detection_dir).parent / f'{Path(detection_dir).stem}.mp4')

    h, w, _ = cv2.imread(frames[0]).shape
    cx = w // 2
    cy = h // 2

    print('Run tracker...')
    # label, interpolate, merge and filter tracks
    tracker = IoUTracker().label_iou(detections, max_lost=90) \
                          .interpolate() \
                          .filter_min_length(min_length=10) \
                          .merge_deepface(frames, sample=10, threshold=0.6) \
                          .filter_topk_length(top_k=num_tracks) \
                          .select_center((cx, cy))

    print('[postprocess] number of tracks:', len(tracker.tracks),
          'lengths of tracks:',
          [(id, len(track)) for id, track in tracker.tracks.items()])

    return tracker.tracks
'''

def center_face_track(frame_dir: str | Path,
                      min_det_score: float = 0.7,
                      bb_iou_thr: float = 0.2,
                      interp_max_lost: int = 30,
                      merge_max_lost: int = -1,
                      merge_iou_thr: float = 0.2,
                      batch_size: int = 32,
                      gpu_id: int = 0,
                      retinaface_arch: str = 'resnet50',
                      verbose: bool = False,
                      cache_vdet: str = 'test.vdet',
                      **kwargs):

    assert retinaface_arch in {
        'mobilenet', 'resnet50'
    }, 'Invalid architecture choice for RetinaFace. Choose from ["mobilenet","resnet50"].'

    print('Run face detector...')
    face_detector = FaceDetector(gpu_id=gpu_id, batch_size=batch_size, verbose=verbose)
    if Path(frame_dir).is_dir():
        video_detections = face_detector.iterate_folder(frame_dir=frame_dir, output_path=cache_vdet)
    else: # video
        video_detections = face_detector.detect_video(video_path=frame_dir, output_path=cache_vdet)

    print('Run tracker...')
    # label, interpolate, merge and filter tracks
    tracker = Tracker().label_tracks_iou(video_detections, min_score=min_det_score, iou_threshold=bb_iou_thr, max_lost=interp_max_lost) \
                       .merge_iou(max_lost=merge_max_lost, iou_threshold=merge_iou_thr) 

    
    track: Track = tracker.select_topk_long_tracks(top_k=2) \
                          .get_center_track()

    print('[Track]\n\tnumber of tracks:', len(tracker.tracks),
          '\n\tlengths of tracks:', [(id, len(track)) for id, track in tracker.tracks.items()],
          '\n\tlength of selected center track:', len(track))

    return track


if __name__ == '__main__':

    video1 = 'data/videos/9KAqOrdiZ4I.001.mp4'
    video2 = 'data/videos/002003_FC1_A.mp4'

    center_face_track(frame_dir=video1,
                      gpu_id=0,
                      verbose=True,
                      cache_vdet='test.vdet')

    detection_dir = 'data/processed/faces/002003_FC1_A_360p'
    track_dir = 'data/processed/tracks/002003_FC1_A_360p'
