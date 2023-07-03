import pickle
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import bbox_visualizer as bbv
from exordium.utils.shared import load_or_create
from exordium.video.retinaface import detect_faces
from exordium.video.tddfa_v2 import get_3DDFA_V2_landmarks
#from exordium.video.tddfa_v2 import TDDFA_V2, get_3DDFA_V2_landmarks, FACE_LANDMARKS
from exordium.video.bb import xywh2xyxy


def draw_landmarks(img: np.ndarray,
                   landmarks: np.ndarray,
                   output_file: str | Path,
                   format: str = 'BGR'):
    assert format in {'BGR', 'RGB'}
    assert landmarks.shape[1] == 2

    image = img.copy()

    for i in range(landmarks.shape[0]):
        points = np.rint(landmarks[i, :]).astype(int)
        cv2.circle(image, (points[0], points[1]), 1, (0, 0, 255), -1)

    if format == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_file), image)


class FaceAligner():

    def align(
            self,
            img: np.ndarray,
            landmarks: np.ndarray,
            left_eye_ratio=(0.38, 0.38),
            output_width=192,  # preferred for iris estimation
            output_height=None):
        # expected MTCNN implementation: https://github.com/timesler/facenet-pytorch
        # expected RetinaFace implementation: https://github.com/elliottzheng/batch-face
        assert landmarks.shape == (
            5, 2), f'Expected: (5,2), got istead: {landmarks.shape}'

        if output_height is None:
            output_height = output_width

        landmarks = np.rint(landmarks).astype(np.int32)

        left_eye_x, left_eye_y = landmarks[1, :]  # participant's left eye
        right_eye_x, right_eye_y = landmarks[0, :]  # participant's right eye

        dY = right_eye_y - left_eye_y
        dX = right_eye_x - left_eye_x

        angle = np.degrees(np.arctan2(dY, dX)) - 180
        center = (int((left_eye_x + right_eye_x) // 2),
                  int((left_eye_y + right_eye_y) // 2))

        right_eye_ratio_x = 1.0 - left_eye_ratio[0]
        dist = np.sqrt((dX**2) + (dY**2))
        output_dist = (right_eye_ratio_x - left_eye_ratio[0])
        output_dist *= output_width
        scale = output_dist / dist

        M = cv2.getRotationMatrix2D(center, angle, scale)
        t_x = output_width * 0.5
        t_y = output_height * left_eye_ratio[1]
        M[0, 2] += (t_x - center[0])
        M[1, 2] += (t_y - center[1])
        self.M = M

        rotated_face = cv2.warpAffine(img,
                                      M, (output_width, output_height),
                                      flags=cv2.INTER_CUBIC)
        rotated_landmarks = self.apply_transform(landmarks)
        return rotated_face, rotated_landmarks

    def apply_transform(self, landmarks):
        _landmarks = np.concatenate(
            [landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
        return np.rint(np.dot(self.M, _landmarks.T).T).astype(int)


def visualize_detection(img: np.ndarray, bb_xyxy: np.ndarray | list,
                        probability: float, landmarks: np.ndarray | None,
                        output_path: str | Path):
    assert np.array(bb_xyxy).shape == (
        4, ), f'Expected: (4,), got istead: {bb_xyxy.shape}'
    assert isinstance(
        probability,
        (float,
         np.float32)), f'Expected: float, got instead: {type(probability)}'
    assert landmarks is None or landmarks.shape == (
        5, 2
    ), f'Expected: None or ndarray with shape (5,2), got istead: {landmarks.shape}'

    bb_xyxy = np.rint(bb_xyxy).astype(np.int32)
    probability = np.round(probability, decimals=2)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0),
              (255, 255, 255)]
    img = bbv.draw_rectangle(img, bb_xyxy.astype(int))
    img = cv2.putText(img, str(probability), bb_xyxy[:2] - 5,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                      cv2.LINE_AA)

    if landmarks is not None:
        landmarks = np.rint(landmarks).astype(np.int32)
        for i in range(landmarks.shape[0]):
            img = cv2.circle(img, landmarks[i, :].astype(int), 1, colors[i],
                             -1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


@load_or_create('pkl')
def face_crop_with_landmarks(frame_paths: list | str | np.ndarray,
                             size: int = 192,
                             verbose: bool = True,
                             **kwargs):
    if isinstance(frame_paths, str):
        frame_paths = [frame_paths]

    fa = FaceAligner()
    detections = detect_faces(frame_paths=frame_paths,
                              format='xyxys,lmks',
                              verbose=verbose)

    dicts = []
    for i, frame_path in tqdm(enumerate(frame_paths),
                              total=len(frame_paths),
                              desc='Extract landmarks, align and transform',
                              disable=not verbose):
        det = detections.get_highest_score(i)

        if det is None:
            rotated_img, rotated_fine_landmarks = None, None
        else:
            bb, landmarks = det
            img = cv2.imread(frame_path)
            fine_landmarks = get_3DDFA_V2_landmarks(img, [bb])

            rotated_img, rotated_landmarks = fa.align(img.copy(),
                                                      landmarks,
                                                      output_width=size)
            rotated_fine_landmarks = fa.apply_transform(fine_landmarks)

        dicts.append({
            'frame_path': frame_path,
            'img': rotated_img,
            'landmarks': rotated_fine_landmarks
        })

    return dicts


def face_to_eye(img: np.ndarray, landmarks: np.ndarray, bb: tuple = (36, 60)):
    xy_m = np.mean(landmarks, axis=0)
    ratio = bb[0] / bb[1]
    xy_dx = np.linalg.norm(landmarks[0, :] - landmarks[3, :]) * 2
    xy_dy = xy_dx * ratio
    eye = img[int(xy_m[1] - xy_dy // 2):int(xy_m[1] + xy_dy // 2),
              int(xy_m[0] - xy_dx // 2):int(xy_m[0] + xy_dx // 2), :]
    eye = cv2.resize(eye, (bb[1], bb[0]), interpolation=cv2.INTER_AREA)
    return eye


def crop_eyes(frame: np.ndarray,
              left_eye_xy: np.ndarray,
              right_eye_xy: np.ndarray,
              extra_percent_eye: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Cut the eyes from a frame using the bigger side of the bounding box

    Args:
        frame (np.ndarray): image containing the participant's face
        left_eye_xy (np.ndarray | list): two annotated corner points of the left eye (x1, y1, x2, y2) order
        right_eye_xy (np.ndarray | list): two annotated corner points of the right eyes (x1, y1, x2, y2) order

    Returns:
        np.ndarray: cropped eyes

    """
    bb_size_eye = int(
        max([
            np.linalg.norm(left_eye_xy[:2] - left_eye_xy[2:]),
            np.linalg.norm(right_eye_xy[:2] - right_eye_xy[2:]),
        ]) * (1 + extra_percent_eye))
    # centre point
    left_cx = left_eye_xy[0] + abs(left_eye_xy[0] - left_eye_xy[2]) // 2
    # face is not aligned using the manual annotation
    left_cy = (min(left_eye_xy[1], left_eye_xy[3]) +
               abs(left_eye_xy[1] - left_eye_xy[3]) // 2)
    right_cx = right_eye_xy[0] + abs(right_eye_xy[0] - right_eye_xy[2]) // 2
    right_cy = (min(right_eye_xy[1], right_eye_xy[3]) +
                abs(right_eye_xy[1] - right_eye_xy[3]) // 2)
    # normalized bounding box
    left_nx1 = max([left_cx - bb_size_eye // 2, 0])
    left_nx2 = min([left_cx + bb_size_eye // 2, frame.shape[1]])
    left_ny1 = max([left_cy - bb_size_eye // 2, 0])
    left_ny2 = min([left_cy + bb_size_eye // 2, frame.shape[0]])
    right_nx1 = max([right_cx - bb_size_eye // 2, 0])
    right_nx2 = min([right_cx + bb_size_eye // 2, frame.shape[1]])
    right_ny1 = max([right_cy - bb_size_eye // 2, 0])
    right_ny2 = min([right_cy + bb_size_eye // 2, frame.shape[0]])
    return (
        frame[int(left_ny1):int(left_ny2),
              int(left_nx1):int(left_nx2), :],
        frame[int(right_ny1):int(right_ny2),
              int(right_nx1):int(right_nx2), :],
    )


def crop_and_align_face(frame_paths: list | str, size: int = 256, **kwargs):

    detections = detect_faces(frame_paths=frame_paths,
                              format='xyxys,lmks',
                              handle_skip=False,
                              verbose=kwargs.get('verbose', False))

    dicts = []
    fa = FaceAligner()
    for i, frame_path in tqdm(enumerate(frame_paths),
                              total=len(frame_paths),
                              desc='Extract landmarks, align and transform',
                              disable=kwargs.get('verbose', False)):
        det = detections[0]
        
        if det is None:
            rotated_img, rotated_fine_landmarks = None, None
        else:
            bb, landmarks = det
            img = cv2.imread(frame_path)
            fine_landmarks = get_3DDFA_V2_landmarks(img, [bb])
            rotated_img, rotated_landmarks = fa.align(img.copy(),
                                                      landmarks,
                                                      output_width=size)
            rotated_fine_landmarks = fa.apply_transform(fine_landmarks)

        dicts.append({
            'id': i,
            'frame_path': frame_path,
            'img': rotated_img,
            'landmarks': rotated_fine_landmarks
        })  # [(yaw, pitch, roll), ...]

    return dicts


@load_or_create('pkl')
def face_crop_with_lmrks_hp(frame_paths: list | str,
                            size: int = 192,
                            **kwargs):
    if isinstance(frame_paths, str):
        frame_paths = [frame_paths]

    fa = FaceAligner()
    hp = HeadposeExtractor()
    detections = detect_faces(frame_paths=frame_paths,
                              format='xyxys,lmks',
                              verbose=kwargs.get('verbose', False))

    dicts = []
    for i, frame_path in tqdm(enumerate(frame_paths),
                              total=len(frame_paths),
                              desc='Extract landmarks, align and transform',
                              disable=kwargs.get('verbose', False)):
        det = detections.get_highest_score(i)

        if det is None:
            rotated_img, rotated_fine_landmarks = None, None
            headpose = (np.nan, np.nan, np.nan)
        else:
            bb, landmarks = det
            img = cv2.imread(frame_path)
            headpose = hp.estimate_headpose(img, [bb])
            fine_landmarks = get_3DDFA_V2_landmarks(img, [bb])
            rotated_img, rotated_landmarks = fa.align(img.copy(),
                                                      landmarks,
                                                      output_width=size)
            rotated_fine_landmarks = fa.apply_transform(fine_landmarks)

        dicts.append({
            'id': i,
            'frame_path': frame_path,
            'img': rotated_img,
            'landmarks': rotated_fine_landmarks,
            'headpose': headpose
        })  # [(yaw, pitch, roll), ...]

    return dicts


@load_or_create('pkl')
def face_crop_with_lmrks_hp_using_track(frame_paths: list | str,
                                        track_path,
                                        size: int = 192,
                                        **kwargs) -> list[dict]:

    if isinstance(frame_paths, str):
        frame_paths = [frame_paths]

    ids = list(
        range(int(Path(frame_paths[0]).stem),
              int(Path(frame_paths[-1]).stem) + 1))

    with open(track_path, 'rb') as f:
        tracks = pickle.load(f)

    fa = FaceAligner()
    hp = HeadposeExtractor()

    dicts = []
    for id in tqdm(ids,
                   total=len(ids),
                   desc='Extract landmarks, align and transform',
                   disable=kwargs.get('verbose', False)):

        frame_path = next(
            filter(lambda frame_path: int(Path(frame_path).stem) == id,
                   frame_paths), None)

        for _, track in tracks.items():

            if id not in track.location:
                det = None
            else:
                det = track.location[id]

        if det is None or frame_path is None:
            rotated_img, rotated_fine_landmarks = None, None
            headpose = (np.nan, np.nan, np.nan)
        else:
            bb, landmarks = det['bb'], det['landmarks']
            bb = xywh2xyxy(bb)
            img = cv2.imread(frame_path)
            headpose = hp.estimate_headpose(img, [bb])
            fine_landmarks = get_3DDFA_V2_landmarks(img, [bb])
            rotated_img, rotated_landmarks = fa.align(img.copy(),
                                                      landmarks,
                                                      output_width=size)
            rotated_fine_landmarks = fa.apply_transform(fine_landmarks)

        dicts.append({
            'id': id,
            'frame_path': frame_path,
            'img': rotated_img,
            'landmarks': rotated_fine_landmarks,
            'headpose': headpose
        })  # [(yaw, pitch, roll), ...]

    return dicts


def blink_features(frame_paths: list | str,
                   size: int = 192,
                   **kwargs) -> list[dict]:

    if isinstance(frame_paths, str):
        frame_paths = [frame_paths]

    ids = list(
        range(int(Path(frame_paths[0]).stem),
              int(Path(frame_paths[-1]).stem) + 1))

    TODOs

    fa = FaceAligner()
    hp = HeadposeExtractor()

    dicts = []
    for id in tqdm(ids,
                   total=len(ids),
                   desc='Extract landmarks, align and transform',
                   disable=kwargs.get('verbose', False)):

        frame_path = next(
            filter(lambda frame_path: int(Path(frame_path).stem) == id,
                   frame_paths), None)

        for _, track in tracks.items():

            if id not in track.location:
                det = None
            else:
                det = track.location[id]

        if det is None or frame_path is None:
            rotated_img, rotated_fine_landmarks = None, None
            headpose = (np.nan, np.nan, np.nan)
        else:
            bb, landmarks = det['bb'], det['landmarks']
            bb = xywh2xyxy(bb)
            img = cv2.imread(frame_path)
            headpose = hp.estimate_headpose(img, [bb])
            fine_landmarks = get_3DDFA_V2_landmarks(img, [bb])
            rotated_img, rotated_landmarks = fa.align(img.copy(),
                                                      landmarks,
                                                      output_width=size)
            rotated_fine_landmarks = fa.apply_transform(fine_landmarks)

        dicts.append({
            'id': id,
            'frame_path': frame_path,
            'img': rotated_img,
            'landmarks': rotated_fine_landmarks,
            'headpose': headpose
        })  # [(yaw, pitch, roll), ...]

    return dicts


if __name__ == "__main__":

    img_path = 'data/processed/frames/h-jMFLm6U_Y.000/frame_00001.png'
    d = face_crop_with_landmarks(img_path)[0]
    print(d['img'].shape)
    print(d['landmarks'].shape)
    draw_landmarks(d['img'], d['landmarks'], 'test.png')
