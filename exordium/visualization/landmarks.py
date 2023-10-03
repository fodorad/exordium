import os
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from exordium import PathType
from exordium.video.bb import xywh2xyxy
from exordium.video.io import frames2video
from exordium.video.detection import Detection, FrameDetections, VideoDetections, Track

AXES_COLOR = [("x", (255, 0, 0)), ("y", (0, 255, 0)), ("z", (0, 0, 255))]


def visualize_detection(image: np.ndarray,
                        bb_xyxy: np.ndarray,
                        probability: float,
                        output_path: PathType | None = None) -> np.ndarray:
    if bb_xyxy.shape != (4,):
        raise Exception(f'Expected bounding box with shape (4,) got istead {bb_xyxy.shape}.')

    bb_xyxy = np.rint(bb_xyxy).astype(int)
    probability = np.round(probability, decimals=2)

    cv2.rectangle(image, bb_xyxy[:2], bb_xyxy[2:], (255,0,0), 2)
    cv2.putText(image, str(probability), bb_xyxy[:2] - 5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

    return image


def add_detections_to_frame(frame_detections: FrameDetections, frame: np.ndarray | None = None) -> np.ndarray:
    frame = frame or frame_detections[0].frame()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(frame, f"frame id: {frame_detections[0].frame_id:06d}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for detection in frame_detections:
        cv2.putText(frame, f"score: {detection.score:.2f}", (detection.bb_xywh[0] - 5, detection.bb_xywh[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        bb_xyxy = xywh2xyxy(detection.bb_xywh)
        cv2.rectangle(frame, (bb_xyxy[0], bb_xyxy[1]), (bb_xyxy[2], bb_xyxy[3]), (0, 255, 0), 2)
    return frame


def save_detections_to_video(video_detections: VideoDetections, frame_dir: str | Path, output_dir: str | Path, fps: int = 30, sample_every_n: int = 1, verbose: bool = False) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(list(Path(frame_dir).iterdir()))
    frame_ids = [int(Path(frame_path).stem) for frame_path in frame_paths]
    frame_detection_ids = video_detections.frame_ids()
    for frame_id, frame_path in tqdm(zip(frame_ids, frame_paths), total=len(frame_ids), desc='Save frames', disable=not verbose):
        frame = cv2.imread(str(frame_path))
        if frame_id in frame_detection_ids:
            frame_detection = video_detections.get_frame_detection(frame_id)
            frame = add_detections_to_frame(frame_detection, frame)
        cv2.imwrite(str(output_dir / f'{Path(frame_path).stem}.png'), frame)



def save_track_target_to_images(track: Track, output_dir: str | Path, bb_size: int = 224, fps: int = 30, sample_every_n: int = 1, verbose: bool = False) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, detection in tqdm(enumerate(track.detections), desc='Save track targets', disable=not verbose):
        if i % sample_every_n != 0: continue
        image = detection.crop()
        if bb_size != -1:
            image = cv2.resize(image, (bb_size, bb_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output_dir / f'{detection.frame_id:06d}.png'), image)

    frames2video(output_dir, output_dir.parent / f'{output_dir.stem}.mp4', fps)


def save_track_with_context_to_video(track: Track, frame_dir: str | Path, output_dir: str | Path, fps: int = 30, sample_every_n: int = 1, verbose: bool = False) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(list(Path(frame_dir).iterdir()))
    frame_ids = [int(Path(frame_path).stem) for frame_path in frame_paths]
    track_frame_ids = track.frame_ids()

    for frame_id, frame_path in tqdm(zip(frame_ids, frame_paths), total=len(frame_paths), desc='Save video frames', disable=not verbose):
        if frame_id % sample_every_n != 0: continue
        if frame_id not in track_frame_ids: continue

        frame = cv2.imread(str(frame_path))
        detection = track.get_detection(frame_id)

        #cv2.putText(frame, f"frame id: {frame_id:06d}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"score: {detection.score:.2f}", (detection.bb_xyxy[0] - 5, detection.bb_xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(frame, (detection.bb_xyxy[0], detection.bb_xyxy[1]), (detection.bb_xyxy[2], detection.bb_xyxy[3]), (0, 255, 0), 2)
        cv2.imwrite(str(Path(output_dir) / f'{frame_id:06d}.png'), frame)

    frames2video(output_dir, output_dir.parent / f'{output_dir.stem}.mp4', fps)


def visualize_landmarks(image: np.ndarray,
                        landmarks: np.ndarray,
                        output_path: str | os.PathLike | None = None,
                        show_indices: bool = False) -> np.ndarray:
    """Landmarks are drawn to the given image"""

    if not (landmarks.ndim == 2 and landmarks.shape[1] == 2):
        raise Exception(f'Expected landmakrs with shape (5,2) got istead {landmarks.shape}.')

    image_out = np.copy(image)
    landmarks = np.rint(landmarks).astype(int)

    radius = 1
    color = (0, 255, 0)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.3
    font_thickness = 1

    for index in range(len(landmarks)):
        cv2.circle(image_out, landmarks[index, :], radius, color, thickness)
        if show_indices:
            cv2.putText(image_out, str(index), landmarks[index, :],
                        font, fontScale, color, font_thickness, cv2.LINE_AA)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image_out)

    return image_out


def visualize_detection(detection: Detection,
                        output_path: PathType | None = None,
                        show_indices: bool = False):
    return visualize_landmarks(
        detection.frame,
        detection.landmarks,
        output_path,
        show_indices
    )


if __name__ == '__main__':
    from exordium.video.tddfa_v2 import TDDFA_V2
    face_path = 'data/processed_v2/face_rgb/Gdrgw7Z6tLg.003/000000.png'
    face = cv2.imread(face_path)
    tddfa = TDDFA_V2()
    output_dict = tddfa(face)
    visualize_landmarks(face, output_dict['landmarks'], 'out.png', True)