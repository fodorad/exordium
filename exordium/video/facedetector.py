import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from batch_face import RetinaFace
import cv2
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from exordium import PathType
from exordium.video.io import image2np
from exordium.video.bb import xyxy2xywh
from exordium.video.detection import FrameDetections, VideoDetections
from exordium.utils.decorator import load_or_create


class RetinaFaceLandmarks(Enum):
    RIGHT_EYE = 0
    LEFT_EYE = 1
    NOSE = 2
    MOUTH_RIGHT = 3
    MOUTH_LEFT = 4


class FaceDetector(ABC):
    """Face detector abstract class definition.

    notation:
        y is row (height), x is column (width)

    image:
        (0,0)---(0,w)
          |       |
          |       |
        (h,0)---(h,w)

    bounding box:
        (x_min, y_min, x_max, y_max)

    detection format
        (y_min, x_min) -- (y_min, x_max)
              |                 |
        (y_max, x_min) -- (y_max, x_max)
    """
    def __init__(self, batch_size: int = 32, verbose: bool = False):
        self.batch_size = batch_size
        self.verbose = verbose

    @abstractmethod
    def run_detector(self, images_rgb: list[np.ndarray]) -> list[list[tuple[np.ndarray, np.ndarray, float]]]:
        """Run detector on the images.

        Note:
            If no face is detected, then empty list [] will be at that frame index.

        Args:
            images_rgb (list[np.ndarray]): list of input images of shape (H, W, 3) and RGB channel order.

        Returns:
            list[list[tuple[np.ndarray, np.ndarray, float]]]: multiple detections can be per image.
        """

    @load_or_create('fdet')
    def detect_image_path(self, image_path: PathType, **kwargs) -> FrameDetections:
        """Run detector on a single image given by its path.

        Args:
            image_path (PathType): path to the image file.

        Returns:
            FrameDetections: detections within the image.
        """
        image_bgr = image2np(image_path, channel_order='RGB')
        return self.detect_image(image_bgr, image_path=str(image_path), **kwargs)

    @load_or_create('fdet')
    def detect_image(self, image_rgb: np.ndarray, **kwargs) -> FrameDetections:
        """Run detector on a single image.

        Args:
            image_rgb (np.ndarray): input image of shape (H, W, 3) and RGB channel order.

        Returns:
            FrameDetections: detections within the image.
        """
        frame_dets: list[list[tuple[np.ndarray, np.ndarray, float]]] = self.run_detector([image_rgb])

        # iterate over all the faces detected within a frame
        frame_detections = FrameDetections()
        for bb_xyxy, landmarks, score in frame_dets[0]:
            frame_detections.add_dict({
                'frame_id': -1,
                'source': kwargs.get('image_path', image_rgb),
                'score': score,
                'bb_xywh': xyxy2xywh(bb_xyxy),
                'landmarks': np.rint(np.array(landmarks)).astype(int)
            })

        return frame_detections

    @load_or_create('vdet')
    def detect_frame_dir(self, frame_dir: PathType, **kwargs) -> VideoDetections:
        """Run detector on ordered frames of a video in a single folder.

        Args:
            frame_dir (PathType): path to the frame directory.

        Returns:
            VideoDetections: detections within the images.
        """
        frame_paths = sorted(list(Path(frame_dir).iterdir()))
        video_detections = VideoDetections()

        for batch_ind in tqdm(range(0, len(frame_paths), self.batch_size), desc='RetinaFace detection', disable=not self.verbose):
            batch_frame_paths = frame_paths[batch_ind:batch_ind + self.batch_size]
            images = [image2np(frame_path, 'RGB') for frame_path in batch_frame_paths]
            frame_dets: list[list[tuple[np.ndarray, np.ndarray, float]]] = self.run_detector(images)

            # iterate over the frame detections
            for frame_det, frame_path in zip(frame_dets, batch_frame_paths):

                # iterate over all the faces detected within a frame
                frame_detections = FrameDetections()
                for bb_xyxy, landmarks, score in frame_det:
                    frame_detections.add_dict({
                        'frame_id': int(Path(frame_path).stem),
                        'source': str(frame_path),
                        'score': score,
                        'bb_xywh': xyxy2xywh(bb_xyxy),
                        'landmarks': np.rint(np.array(landmarks)).astype(int)
                    })

                video_detections.add(frame_detections)

        return video_detections

    @load_or_create('vdet')
    def detect_video(self, video_path: PathType, **kwargs) -> VideoDetections:
        """Run detector on a video given by its path.

        Args:
            video_path (PathType): path to the video file.

        Returns:
            VideoDetections: detections within the images.
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f'Video does not exist at {str(video_path)}')

        vr = VideoReader(str(video_path), ctx=cpu(0))

        video_detections = VideoDetections()
        for batch_ind in tqdm(range(0, len(vr), self.batch_size), desc='RetinaFace detection', disable=not self.verbose):
            frame_indices = [ind for ind in range(batch_ind, batch_ind + self.batch_size) if ind < len(vr)]
            images: np.ndarray = vr.get_batch(frame_indices).asnumpy() # (T, H, W, C)
            frame_list = [images[i] for i in range(images.shape[0])]
            frame_dets: list[list[tuple[np.ndarray, np.ndarray, float]]] = self.run_detector(frame_list)

            # iterate over the frame detections
            for frame_det, frame_id in zip(frame_dets, frame_indices):

                # iterate over all the faces detected within a frame
                frame_detections = FrameDetections()
                for bb_xyxy, landmarks, score in frame_det:
                    frame_detections.add_dict({
                        'frame_id': frame_id,
                        'source': str(video_path),
                        'score': score,
                        'bb_xywh': xyxy2xywh(bb_xyxy),
                        'landmarks': np.rint(np.array(landmarks)).astype(int)
                    })

                video_detections.add(frame_detections)

        return video_detections


class RetinaFaceDetector(FaceDetector):
    """Face detector wrapper class using RetinaFace CNN."""

    def __init__(self, gpu_id: int = 0, batch_size: int = 16, verbose: bool = False):
        super().__init__(batch_size=batch_size, verbose=verbose)
        self.batch_size = batch_size
        self.detector = RetinaFace(gpu_id=gpu_id, network='resnet50')
        logging.info('RetinaFace is loaded.')

    def run_detector(self, images_rgb: list[np.ndarray]) -> list[list[tuple[np.ndarray, np.ndarray, float]]]:
        images_bgr = [image2np(image, 'BGR') for image in images_rgb]
        return self.detector(images_bgr, cv=True) # RetinaFace expects BGR images


def align_face_and_landmarks(image: np.ndarray,
                             landmarks: np.ndarray,
                             left_eye_landmarks: np.ndarray,
                             right_eye_landmarks: np.ndarray,
                             eye_ratio: tuple[float, float] = (0.38, 0.38),
                             output_wh: tuple[int, int] = (192, 192)) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Aligns the x axis of the Head Coordinate System (HCS) to the x axis of the Camera Coordinate System (CCS).
    The face is rotated in the opposite direction of its current roll value of the headpose.

    Args:
        image (np.ndarray): input image of shape (H, W, C).
        landmarks (np.ndarray): input landmarks of shape (N, 2).
        left_eye_landmarks (np.ndarray): landmarks of the left eye of shape (2,)
        right_eye_landmarks (np.ndarray): landmarks of the right eye of shape (2,)
        eye_ratio (tuple[float, float], optional): location of the eye within the image and on the x axis. Defaults to (0.38, 0.38).
        output_wh (tuple[int, int], optional): output width and height. Defaults to (192, 192).

    Raises:
        Exception: invalid landmark shape.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float]: rotated face image, rotated landmarks, rotation angle in degree, rotation matrix
    """
    if landmarks.shape[1] != 2:
        raise Exception(f'Expected landmarks with shape (N, 2) got istead {landmarks.shape}.')

    landmarks = np.rint(landmarks).astype(int)
    output_width, output_height = output_wh
    right_eye_x, right_eye_y = right_eye_landmarks # participant's right eye
    left_eye_x, left_eye_y = left_eye_landmarks # participant's left eye

    dY = right_eye_y - left_eye_y
    dX = right_eye_x - left_eye_x
    rotation_degree = np.degrees(np.arctan2(dY, dX)) - 180
    center = (int((left_eye_x + right_eye_x) // 2),
              int((left_eye_y + right_eye_y) // 2))

    right_eye_ratio_x = 1.0 - eye_ratio[0]
    dist = np.sqrt((dX**2) + (dY**2))
    output_dist = (right_eye_ratio_x - eye_ratio[0])
    output_dist *= output_width
    scale = output_dist / dist

    R = cv2.getRotationMatrix2D(center, rotation_degree, scale)
    t_x = output_width * 0.5
    t_y = output_height * eye_ratio[1]
    R[0, 2] += (t_x - center[0])
    R[1, 2] += (t_y - center[1])

    rotated_face = cv2.warpAffine(image, R, (output_width, output_height), flags=cv2.INTER_CUBIC)
    _landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
    rotated_landmarks = np.rint(np.dot(R, _landmarks.T).T).astype(int)
    return rotated_face, rotated_landmarks, rotation_degree, R


def crop_eye_keep_ratio(img: np.ndarray, landmarks: np.ndarray, bb: tuple = (36, 60)):
    xy_m = np.mean(landmarks, axis=0)
    ratio = bb[0] / bb[1]
    xy_dx = np.linalg.norm(landmarks[0, :] - landmarks[3, :]) * 2
    xy_dy = xy_dx * ratio
    eye = img[int(xy_m[1] - xy_dy // 2):int(xy_m[1] + xy_dy // 2),
              int(xy_m[0] - xy_dx // 2):int(xy_m[0] + xy_dx // 2), :]
    eye = cv2.resize(eye, (bb[1], bb[0]), interpolation=cv2.INTER_AREA)
    return eye