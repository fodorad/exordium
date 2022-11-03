import csv
import cv2
import numpy as np
from tqdm import tqdm
from batch_face import RetinaFace # pip install git+https://github.com/elliottzheng/batch-face.git@master
from exordium.shared import load_or_create
from exordium.preprocess.video.bb import xyxy2xywh


class RetinafaceDetections():

    def __init__(self):
        self.detections = []

    def add(self, detection: dict):
        self.detections.append(detection)

    def __len__(self):
        return len(self.detections)

    def __getitem__(self, idx):
        return self.detections[idx]

    def save(self, output_file: str):

        with open(output_file, 'w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['frame', 'score',
                             'x', 'y', 'w', 'h',
                             'left_eye_x', 'left_eye_y',
                             'right_eye_x', 'right_eye_y',
                             'nose_x', 'nose_y',
                             'left_mouth_x', 'left_mouth_y',
                             'right_mouth_x', 'right_mouth_y'])

            for detection in self.detections:
                writer.writerow([detection['frame'],
                                 detection['score'],
                                 *detection['bb']] +
                                 list(np.reshape(detection['landmarks'], (10,))))


    def load(self, input_file: str):

        with open(input_file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')

            for line_count, record in enumerate(csv_reader):
                if line_count > 0:
                    self.detections.append({
                        'frame': int(record[0]),
                        'score': float(record[1]),
                        'bb': np.array(record[2:6], dtype=np.int32),
                        'landmarks': np.array(record[6:16], dtype=np.int32).reshape(5,2),
                    })
        return self


@load_or_create('det')
def detect_faces(frame_paths: list | str,
                 detector: RetinaFace = None,
                 batch_size: int = 32,
                 gpu_id: int = 0, **kwargs):

    if isinstance(frame_paths, str | np.ndarray):
        frame_paths = [frame_paths]
    
    if detector is None:
        detector = RetinaFace(gpu_id=gpu_id, network='resnet50') # 'mobilenet'

    detections = RetinafaceDetections()
    for batch_ind in tqdm(range(0, len(frame_paths), batch_size)):

        if isinstance(frame_paths[0], str): # list of image paths
            imgs = [cv2.imread(frame) for frame in frame_paths[batch_ind:batch_ind+batch_size]]
        else: # list of images: ndarray with shape (h, w, 3)
            imgs = frame_paths[batch_ind:batch_ind+batch_size]

        faces = detector(imgs, cv=True)
        for frame in range(len(faces)):
            for face_ind in range(len(faces[frame])):
                box, landmarks, score = faces[frame][face_ind]
                # image:
                # (0,0)---(0,w)
                #   |       |
                #   |       |
                # (h,0)---(h,w)
                #
                # bounding box:
                # (x_min, y_min, x_max, y_max)
                #
                # original detection format
                #    (y_min, x_min) -- (y_min, x_max)
                #           |               |
                #    (y_max, x_min) -- (y_max, x_max)
                #
                # y is row (height), x is column (width)
                box = np.rint(np.array(box)).astype(int)
                box = np.where(box < 0, np.zeros_like(box), box) # negative index outside of the picture
                xywh = xyxy2xywh(box).astype(np.int32)
                landmarks = np.rint(np.array(landmarks)).astype(np.int32)
                detections.add({'frame': batch_ind+frame,
                                'score': score,
                                'bb': xywh,
                                'landmarks': landmarks})
    return detections
