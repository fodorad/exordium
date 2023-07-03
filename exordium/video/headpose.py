import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from exordium.video.retinaface import detect_faces


def frames2headpose(frame_paths: str | np.ndarray | list, model: HeadposeExtractor | None = None):

    if model is None:
        model = HeadposeExtractor()

    if isinstance(frame_paths, str | np.ndarray):
        frame_paths = [frame_paths]

    detections = detect_faces(frame_paths=frame_paths, format='xyxys,lmks')

    poses = []
    for i, frame_path in enumerate(frame_paths):
        #detections.format = 'xyxys,lmks'
        det = detections.get_highest_score(frame_ind=i)

        if isinstance(frame_path, str):
            image = cv2.imread(frame_path)
        elif isinstance(frame_path, np.ndarray):
            image = frame_path

        if det is not None:
            headpose = model.estimate_headpose(image, [det[0]])
        else:
            headpose = (np.nan, np.nan, np.nan)

        elem = {'id': i, 'frame_path': frame_path, 'headpose': headpose} # [(yaw, pitch, roll), ...]
        poses.append(elem) 

    return poses


def face2headpose(face_paths: str | Path | np.ndarray | list, model: HeadposeExtractor | None = None, verbose: bool = False):

    if model is None:
        model = HeadposeExtractor()

    if isinstance(face_paths, str | Path | np.ndarray):
        face_paths = [face_paths]

    poses = []
    for face in tqdm(face_paths, desc='Extract headpose', disable=not verbose):

        id = int(Path(face).stem)

        if isinstance(face, str | Path):
            image = cv2.imread(str(face))
        elif isinstance(face, np.ndarray):
            image = face

        bb = [0, 0, image.shape[0], image.shape[1]]

        if len(np.unique(image)) == 1 and image.ravel()[0] in [0, np.nan]:
            headpose = (np.nan, np.nan, np.nan)
        else:
            headpose = model.estimate_headpose(image, [bb])

        elem = {'id': id, 'frame_path': None if isinstance(face, np.ndarray) else face, 'headpose': headpose} # [(yaw, pitch, roll), ...]
        poses.append(elem)

    return poses