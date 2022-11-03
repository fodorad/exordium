from pathlib import Path

import cv2
import numpy as np
import bbox_visualizer as bbv

from exordium.preprocess.video.retinaface import RetinaFace, detect_faces


def draw_landmarks(img, landmarks, output_file: str | Path, format: str = 'BGR'):
    assert format in {'BGR', 'RGB'}
    assert landmarks.shape[1] == 2

    image = img.copy()

    for i in range(landmarks.shape[0]):
        points = np.rint(landmarks[i,:]).astype(int)
        cv2.circle(image, (points[0], points[1]), 3, (0, 0, 255), -1)

    if format == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_file), image)


def face_alignment(img: np.ndarray,
                   landmarks: np.ndarray,
                   detector: str = 'retinaface',
                   left_eye_ratio=(0.38, 0.38),
                   output_width=224,
                   output_height=None):
    # expected MTCNN implementation: https://github.com/timesler/facenet-pytorch
    # expected RetinaFace implementation: https://github.com/elliottzheng/batch-face

    assert landmarks.shape == (5,2), f'Expected: (5,2), got istead: {landmarks.shape}'
    assert detector in {'mtcnn', 'retinaface'}, 'Only MTCNN and RetinaFace format is supported right now.'

    if output_height is None:
        output_height = output_width

    landmarks = np.rint(landmarks).astype(np.int32)

    left_eye_x, left_eye_y = landmarks[1,:] # participant's left eye
    right_eye_x, right_eye_y = landmarks[0,:] # participant's right eye

    dY = right_eye_y - left_eye_y
    dX = right_eye_x - left_eye_x

    angle = np.degrees(np.arctan2(dY, dX)) - 180
    center = (int((left_eye_x + right_eye_x) // 2),
              int((left_eye_y + right_eye_y) // 2))

    right_eye_ratio_x = 1.0 - left_eye_ratio[0]
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    output_dist = (right_eye_ratio_x - left_eye_ratio[0])
    output_dist *= output_width
    scale = output_dist / dist

    M = cv2.getRotationMatrix2D(center, angle, scale)
    t_x = output_width * 0.5
    t_y = output_height * left_eye_ratio[1]
    M[0, 2] += (t_x - center[0])
    M[1, 2] += (t_y - center[1])

    _landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0],1))], axis=1)

    rotated_face = cv2.warpAffine(img, M, (output_width, output_height), flags=cv2.INTER_CUBIC)
    rotated_landmarks = np.rint(np.dot(M, _landmarks.T).T).astype(int)
    return rotated_face, rotated_landmarks


def visualize_mtcnn(img: np.ndarray,
                    bb_xyxy: np.ndarray,
                    probability: float,
                    landmarks: np.ndarray,
                    output_path: str | Path):
    assert bb_xyxy.shape == (4,), f'Expected: (4,), got istead: {bb_xyxy.shape}'
    assert isinstance(probability, (float, np.float32)), f'Expected: float, got instead: {type(probability)}'
    assert landmarks.shape == (5,2), f'Expected: (5,2), got istead: {landmarks.shape}'

    bb_xyxy = np.rint(bb_xyxy).astype(np.int32)
    landmarks = np.rint(landmarks).astype(np.int32)
    probability = np.round(probability, decimals=2)

    colors = [(255,0,0),(0,255,0), (0,0,255), (0,0,0), (255,255,255)]
    img = bbv.draw_rectangle(img, bb_xyxy.astype(int))
    # img = bbv.add_label(img, "{:2f}".format(probability), bb_xyxy)
    img = cv2.putText(img, str(probability), bb_xyxy[:2]-5, cv2.FONT_HERSHEY_SIMPLEX,
                      0.5, (0,255,0), 1, cv2.LINE_AA)

    for i in range(landmarks.shape[0]):
        img = cv2.circle(img, landmarks[i,:].astype(int), 1, colors[i], -1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))



def frames_to_aligned_face_and_landmarks(frame_paths: list | str):
    if isinstance(frame_paths, str): frame_paths = [frame_paths]
    dicts = []
    detections = detect_faces(frame_paths=frame_paths, detector=RetinaFace(gpu_id=0, network='resnet50'))
    for i, frame_path in enumerate(frame_paths):
        dets = sorted([det for det in detections if det['frame'] == i], key=lambda d: d['score'])
        
        if len(dets) > 0:
            img = cv2.imread(frame_path)
            img_orig = img.copy()
            det = dets[0]

            print(img.shape)
            for points in [det['landmarks'][i,:] for i in range(5)]:
                cv2.circle(img, (points[0], points[1]), 3, (0, 0, 255), -1)
            
            rotated_img, rotated_landmarks = face_alignment(img_orig, det['landmarks'], detector='retinaface')
            dicts.append({'img': rotated_img, 'landmarks': rotated_landmarks})
    return dicts
            

if __name__ == "__main__":
    img_path = 'data/processed/frames/h-jMFLm6U_Y.000/frame_00001.png'
    #print(frames2headpose([img_path, img_path, img_path]))

    d = frames_to_aligned_face_and_landmarks(img_path)
    print(d)