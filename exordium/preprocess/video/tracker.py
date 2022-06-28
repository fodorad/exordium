import os
import csv
import random
from pathlib import Path
from unicodedata import decimal
from PIL import Image
from collections import OrderedDict
import itertools

import cv2
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
from scipy import interpolate

from exordium.shared import timer, timer_with_return, load_or_create
from exordium.preprocess.video.frames import frames2video
from batch_face import RetinaFace # pip install git+https://github.com/elliottzheng/batch-face.git@master
# from motrackers.utils.misc import xyxy2xywh, xywh2xyxy, iou_xywh
import bbox_visualizer as bbv


def get_centroid(bboxes):
    """
    Calculate centroids for multiple bounding boxes.

    Args:
        bboxes (numpy.ndarray): Array of shape `(n, 4)` or of shape `(4,)` where
            each row contains `(xmin, ymin, width, height)`.

    Returns:
        numpy.ndarray: Centroid (x, y) coordinates of shape `(n, 2)` or `(2,)`.

    """

    one_bbox = False
    if len(bboxes.shape) == 1:
        one_bbox = True
        bboxes = bboxes[None, :]

    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    w, h = bboxes[:, 2], bboxes[:, 3]

    xc = xmin + 0.5*w
    yc = ymin + 0.5*h

    x = np.hstack([xc[:, None], yc[:, None]])

    if one_bbox:
        x = x.flatten()
    return x


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Source: https://github.com/bochinski/iou-tracker/blob/master/util.py

    Args:
        bbox1 (numpy.array or list[floats]): Bounding box of length 4 containing
            ``(x-top-left, y-top-left, x-bottom-right, y-bottom-right)``.
        bbox2 (numpy.array or list[floats]): Bounding box of length 4 containing
            ``(x-top-left, y-top-left, x-bottom-right, y-bottom-right)``.

    Returns:
        float: intersection-over-onion of bbox1, bbox2.
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1), (x0_2, y0_2, x1_2, y1_2) = bbox1, bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0.0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    iou_ = size_intersection / size_union

    return iou_


def iou_xywh(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Source: https://github.com/bochinski/iou-tracker/blob/master/util.py

    Args:
        bbox1 (numpy.array or list[floats]): bounding box of length 4 containing ``(x-top-left, y-top-left, width, height)``.
        bbox2 (numpy.array or list[floats]): bounding box of length 4 containing ``(x-top-left, y-top-left, width, height)``.

    Returns:
        float: intersection-over-onion of bbox1, bbox2.
    """
    bbox1 = bbox1[0], bbox1[1], bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]
    bbox2 = bbox2[0], bbox2[1], bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]

    iou_ = iou(bbox1, bbox2)

    return iou_


def xyxy2xywh(xyxy):
    """
    Convert bounding box coordinates from (xmin, ymin, xmax, ymax) format to (xmin, ymin, width, height).

    Args:
        xyxy (numpy.ndarray):

    Returns:
        numpy.ndarray: Bounding box coordinates (xmin, ymin, width, height).

    """

    if len(xyxy.shape) == 2:
        w, h = xyxy[:, 2] - xyxy[:, 0] + 1, xyxy[:, 3] - xyxy[:, 1] + 1
        xywh = np.concatenate((xyxy[:, 0:2], w[:, None], h[:, None]), axis=1)
        return xywh.astype("int")
    elif len(xyxy.shape) == 1:
        (left, top, right, bottom) = xyxy
        width = right - left + 1
        height = bottom - top + 1
        return np.array([left, top, width, height]).astype('int')
    else:
        raise ValueError("Input shape not compatible.")


def xywh2xyxy(xywh):
    """
    Convert bounding box coordinates from (xmin, ymin, width, height) to (xmin, ymin, xmax, ymax) format.

    Args:
        xywh (numpy.ndarray): Bounding box coordinates as `(xmin, ymin, width, height)`.

    Returns:
        numpy.ndarray : Bounding box coordinates as `(xmin, ymin, xmax, ymax)`.

    """

    if len(xywh.shape) == 2:
        x = xywh[:, 0] + xywh[:, 2]
        y = xywh[:, 1] + xywh[:, 3]
        xyxy = np.concatenate((xywh[:, 0:2], x[:, None], y[:, None]), axis=1).astype('int')
        return xyxy
    if len(xywh.shape) == 1:
        x, y, w, h = xywh
        xr = x + w
        yb = y + h
        return np.array([x, y, xr, yb]).astype('int')


def midwh2xywh(midwh):
    """
    Convert bounding box coordinates from (xmid, ymid, width, height) to (xmin, ymin, width, height) format.

    Args:
        midwh (numpy.ndarray): Bounding box coordinates (xmid, ymid, width, height).

    Returns:
        numpy.ndarray: Bounding box coordinates (xmin, ymin, width, height).
    """

    if len(midwh.shape) == 2:
        xymin = midwh[:, 0:2] - midwh[:, 2:] * 0.5
        wh = midwh[:, 2:]
        xywh = np.concatenate([xymin, wh], axis=1).astype('int')
        return xywh
    if len(midwh.shape) == 1:
        xmid, ymid, w, h = midwh
        xywh = np.array([xmid-w*0.5, ymid-h*0.5, w, h]).astype('int')
        return xywh


def intersection_complement_indices(big_set_indices, small_set_indices):
    """
    Get the complement of intersection of two sets of indices.

    Args:
        big_set_indices (numpy.ndarray): Indices of big set.
        small_set_indices (numpy.ndarray): Indices of small set.

    Returns:
        numpy.ndarray: Indices of set which is complementary to intersection of two input sets.
    """
    assert big_set_indices.shape[0] >= small_set_indices.shape[1]
    n = len(big_set_indices)
    mask = np.ones((n,), dtype=bool)
    mask[small_set_indices] = False
    intersection_complement = big_set_indices[mask]
    return intersection_complement


def nms(boxes, scores, overlapThresh, classes=None):
    """
    Non-maximum suppression. based on Malisiewicz et al.

    Args:
        boxes (numpy.ndarray): Boxes to process (xmin, ymin, xmax, ymax)
        scores (numpy.ndarray): Corresponding scores for each box
        overlapThresh (float):  Overlap threshold for boxes to merge
        classes (numpy.ndarray, optional): Class ids for each box.

    Returns:
        tuple: a tuple containing:
            - boxes (list): nms boxes
            - scores (list): nms scores
            - classes (list, optional): nms classes if specified

    """

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    if scores.dtype.kind == "i":
        scores = scores.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    if classes is not None:
        return boxes[pick], scores[pick], classes[pick]
    else:
        return boxes[pick], scores[pick]

@timer_with_return
def face_alignment(img: np.ndarray,
                   landmarks: np.ndarray,
                   detector: str = 'mtcnn',
                   desiredLeftEye=(0.38, 0.38),
                   desiredFaceWidth=224,
                   desiredFaceHeight=None):
    # expected MTCNN implementation: https://github.com/timesler/facenet-pytorch

    assert landmarks.shape == (5,2), f'Expected: (5,2), got istead: {landmarks.shape}'
    assert detector in {'mtcnn'}, 'Only MTCNN format is supported right now.'

    # if the desired face height is None, set it to be the
    # desired face width (normal behavior)
    if desiredFaceHeight is None:
        desiredFaceHeight = desiredFaceWidth

    landmarks = np.rint(landmarks).astype(np.int32)

    left_eye_x, left_eye_y = landmarks[1,:] # participant's left eye
    right_eye_x, right_eye_y = landmarks[0,:] # participant's right eye

    # compute the angle between the eye centroids
    dY = right_eye_y - left_eye_y
    dX = right_eye_x - left_eye_x
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    eyesCenter = (int((left_eye_x + right_eye_x) // 2),
                  int((left_eye_y + right_eye_y) // 2))

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    desiredRightEyeX = 1.0 - desiredLeftEye[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - desiredLeftEye[0])
    desiredDist *= desiredFaceWidth
    scale = desiredDist / dist

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    tX = desiredFaceWidth * 0.5 
    tY = desiredFaceHeight * desiredLeftEye[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    return cv2.warpAffine(img, M, (desiredFaceWidth, desiredFaceHeight), flags=cv2.INTER_CUBIC)


def face_alignment2(img: np.ndarray,
                   landmarks: np.ndarray,
                   bb_xyxy: np.ndarray = None,
                   detector: str = 'mtcnn'):
    # modified version of function in deepface repository:
    # https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py
    # left_eye, right_eye, nose: (x, y), (h, w)
    # img[x,y,:] = (0,255,0)
    # cv2 image, top left is 0,0
    #
    # expected MTCNN implementation: https://github.com/timesler/facenet-pytorch

    assert landmarks.shape == (5,2), f'Expected: (5,2), got istead: {landmarks.shape}'
    assert detector in {'mtcnn'}, 'Only MTCNN format is supported right now.'

    bb_xyxy = np.rint(bb_xyxy).astype(np.int32)
    landmarks = np.rint(landmarks).astype(np.int32)

    right_eye = landmarks[0,:]
    left_eye = landmarks[1,:]
    nose = landmarks[2,:]

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # find rotation direction
    if left_eye_y < right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 # rotate clcokwise
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 # rotate counter clockwise

    # Euclidean distance of p1 and p2:
    #     np.sqrt(np.sum(np.power(p1-p2, 2)))
    # or
    #     np.linalg.norm((p1-p2))
    a = np.linalg.norm(np.array(left_eye)  - np.array(point_3rd))
    b = np.linalg.norm(np.array(right_eye) - np.array(point_3rd))
    c = np.linalg.norm(np.array(right_eye) - np.array(left_eye))

    assert b != 0 and c != 0, 'Division by zero'

    cos_a = (b*b + c*c - a*a)/(2*b*c) # apply cosine rule
    cos_a = np.clip(cos_a, -1., 1.) # floating point errors can lead to NaN
    angle = (np.arccos(cos_a) * 180) / np.pi # radian to degree

    if direction == -1:
        angle = 90 - angle

    img = np.array(Image.fromarray(img).rotate(direction*angle, resample=Image.BICUBIC)) # rotate

    return img

def visualize_mtcnn(img: np.ndarray,
                    bb_xyxy: np.ndarray,
                    probability: float,
                    landmarks: np.ndarray,
                    output_path: str = 'test.png'):
    assert bb_xyxy.shape == (4,)
    assert isinstance(probability, (float, np.float32)), f'Expected: float, got instead: {type(probability)}'
    assert landmarks.shape == (5,2), f'Expected: (5,2), got istead: {landmarks.shape}'

    bb_xyxy = np.rint(bb_xyxy).astype(np.int32)
    landmarks = np.rint(landmarks).astype(np.int32)
    probability = np.round(probability, decimals=2)

    colors = [(255,0,0),(0,255,0), (0,0,255), (0,0,0), (255,255,255)]
    img = bbv.draw_rectangle(img, bb_xyxy.astype(int))
    #img = bbv.add_label(img, "{:2f}".format(probability), bb_xyxy)
    img = cv2.putText(img, str(probability), bb_xyxy[:2]-5, cv2.FONT_HERSHEY_SIMPLEX,
                      0.5, (0,255,0), 1, cv2.LINE_AA)
    for i in range(landmarks.shape[0]):
        img = cv2.circle(img, landmarks[i,:].astype(int), 1, colors[i], -1)

    if output_path is not None:
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def test_face_alignment(img_path: str = None):
    import torch
    from facenet_pytorch import MTCNN
    if img_path is None:
        img_path = 'data/processed/frames/h-jMFLm6U_Y.000/frame_00001.png'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    bb_xyxy, probs, landmarks = mtcnn.detect(img, landmarks=True)
    bb_xyxy = bb_xyxy[0]
    probability = probs[0]
    landmarks = landmarks[0]
    print(bb_xyxy)
    print(probs)
    print(landmarks)
    img_path = Path(img_path)
    output_path_orig = img_path.parents[2] / 'faces_aligned' / img_path.parent.name / img_path.name
    output_path_aligned = img_path.parents[2] / 'faces_aligned' / img_path.parent.name / f'{img_path.stem}_aligned.png'
    output_path_orig.parent.mkdir(parents=True, exist_ok=True)
    output_path_aligned.parent.mkdir(parents=True, exist_ok=True)
    visualize_mtcnn(img, bb_xyxy, probability, landmarks, output_path=str(output_path_orig))

    img = face_alignment(img, landmarks)
    #visualize_mtcnn(img, bb_xyxy, probability, landmarks, output_path='test_aligned.png')
    cv2.imwrite(str(output_path_aligned), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


class Track():

    def __init__(self, id: int, detection: dict, verbose: bool = False):
        # detection: score: float, bb: np.ndarray (xywh), landmarks: np.ndarray (5,2)
        self.id = id
        self.location = dict()
        self.update(detection)
        if verbose:
            print(f'{detection["frame"]}: new track ({self.id}) is started')

    def update(self, detection):
        self.location[detection['frame']] = detection

    def merge(self, track):
        for k, v in track.location.items():
            if k not in self.location:
                self.location[k] = v

    def sample(self, num: int = 5):
        if len(self.location) == 0: return None
        dets = []
        for k, v in self.location.items():
            if v['score'] != -1: # not an interpolated detection
                dets.append((k, v))
        if len(dets) < num: return dets
        return random.sample(dets, num)

    def center(self):
        if len(self.location) == 0: return None, None
        xs = []
        ys = []
        for _, v in self.location.items():
            if v['score'] != -1: # not an interpolated detection
                bb = v['bb']
                xs.append(bb[0]+bb[2]//2)
                ys.append(bb[1]+bb[3]//2)
        return np.array([np.array(xs).mean(), np.array(ys).mean()])

    def bb_size(self, extra_percent: float = 0.2):
        if len(self.location) == 0: return None, None
        ws = []
        hs = []
        for _, v in self.location.items():
            if v['score'] != -1: # not an interpolated detection
                bb = v['bb'] # xywh
                ws.append(bb[2])
                hs.append(bb[3])
        return int(max(ws+hs) * (1 + extra_percent))


    def __len__(self):
        return len(self.location)

    def __str__(self):
        return f'{self.id} track with {len(self.location)} dets from {list(self.location.keys())[0]} to {list(self.location.keys())[-1]}'


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
            line_count = 0
            for record in csv_reader:
                if line_count > 0:
                    self.detections.append({
                        'frame': int(record[0]),
                        'score': float(record[1]),
                        'bb': np.array(record[2:6], dtype=np.int32),
                        'landmarks': np.array(record[6:16], dtype=np.int32).reshape(5,2),
                    })
                line_count += 1
        return self


class IoUTracker():

    def __init__(self):
        self.new_track_id = 0
        self.tracks = dict()

    def label_iou(self, detections,
                        iou_thr: float = 0.2,
                        min_det_score: float = 0.7,
                        max_lost: int = 25):
        print('label iou')
        for detection in tqdm(detections):
            # drop low confidence detection
            if detection['score'] < min_det_score: continue

            # initialize the very first track
            if len(self.tracks) == 0:
                self.tracks[self.new_track_id] = Track(self.new_track_id, detection)
                self.new_track_id += 1
                continue

            # current state of the dictionary, otherwise "RuntimeError: dictionary changed size during iteration"
            tracks = list(self.tracks.items()).copy()

            print('number of active tracks:', len(tracks))

            tracks_to_add = []
            # if condition is met, assign detection to an unfinished track
            for _, track in tracks:
                # get track's last timestep's detection
                last_detection_frame = sorted(list(track.location.keys()))[-1]

                # same frame, different detection -> skip assignment
                if last_detection_frame == detection['frame']:
                    #print(f'{detection["frame"]}: same frame, different detection')
                    continue

                # calculate iou of the detection and the track's last detection
                iou = iou_xywh(track.location[last_detection_frame]['bb'], detection['bb'])

                # last frame is within lost frame tolerance and IoU threshold is met
                if detection['frame'] - last_detection_frame < max_lost and \
                    iou > iou_thr:

                    # add detection to track
                    tracks_to_add.append((track.id, iou))
                    #print(f'{detection["frame"]}: track ({track.id}) is extended')

            if len(tracks_to_add) == 0:
                # start new track
                self.tracks[self.new_track_id] = Track(self.new_track_id, detection)
                self.new_track_id += 1
            else:
                # get good iou tracks, sort by iou and get the highest one
                best_iou_track = sorted(tracks_to_add, key=lambda x: x[1], reverse=True)[0]
                # update the best track with the detection
                self.tracks[best_iou_track[0]].update(detection)
        return self

    def interpolate(self):
        print('interpolate')
        for _, track in self.tracks.items():
            # get frames with detections
            t_list = np.array(sorted(list(track.location.keys())))
            # find missing frames
            indices = np.where(np.diff(t_list)>1)[0][::-1]
            for ind in indices:
                # BB interp from t_start to t_end
                t_start = t_list[ind]
                t_end = t_list[ind+1]
                # get start and end detections
                d_start = track.location[t_start]
                d_end = track.location[t_end]

                # interpolate bb coords
                # bb shape == (4,2)
                bb_interp = interpolate.interp1d(np.array([t_start, t_end]), 
                                                 np.array([d_start['bb'], d_end['bb']]).T)
                new_bb = bb_interp(np.arange(t_start, t_end+1, 1))

                # interpolate lmks coords
                lmks_x_interp = interpolate.interp1d(np.array([t_start, t_end]), 
                                                     np.array([d_start['landmarks'][:,0], d_end['landmarks'][:,0]]).T)
                lmks_y_interp = interpolate.interp1d(np.array([t_start, t_end]), 
                                                     np.array([d_start['landmarks'][:,1], d_end['landmarks'][:,1]]).T)

                new_lmks_x = lmks_x_interp(np.arange(t_start, t_end+1, 1))
                new_lmks_y = lmks_y_interp(np.arange(t_start, t_end+1, 1))

                # round and change type
                new_frame_ids = np.arange(t_start+1, t_end, 1)
                new_bb = np.round(new_bb[:,1:-1]).astype(np.int32)
                new_lmks_x = np.round(new_lmks_x[:,1:-1]).astype(np.int32)
                new_lmks_y = np.round(new_lmks_y[:,1:-1]).astype(np.int32)

                # add interpolated detections to the tracks
                for ind, t in enumerate(new_frame_ids):
                    new_d = {
                        'frame': t,
                        'score': -1,
                        'bb': new_bb[:, ind],
                        'landmarks': np.stack([new_lmks_x[:, ind], new_lmks_y[:, ind]]).T,
                    }
                    track.update(new_d)
        return self

    def merge_deepface(self, frames: list, sample: int = 5, threshold: float = 0.85):
        print('DeepFace verification started...')
        # models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
        model_name = 'ArcFace'
        model = DeepFace.build_model(model_name)
        print(model_name, 'is loaded.')
        track_ids = list(self.tracks.keys())
        if len(track_ids) == 1: return self
        blacklist = []
        for k in track_ids:
            if k in blacklist: continue
            for t in track_ids:
                if k == t or t in blacklist: continue
                dets_k = self.tracks[k].sample(sample)
                dets_t = self.tracks[t].sample(sample)
                if dets_k is None or dets_t is None: continue
                combs = list(itertools.product(list(range(len(dets_k))), list(range(len(dets_t)))))
                #print(f'Number of detections checked for merge: {len(dets_k)}-{len(dets_t)}')
                frame_pairs = []
                for i, (k_i, t_i) in enumerate(combs):
                    print(len(blacklist), '/', len(track_ids), '|', k, 'vs', t, '|', i, '/', len(combs), '\r', end='', flush=True)
                    frame_id_k = dets_k[k_i][0]
                    frame_id_t = dets_t[t_i][0]
                    bb_k_xyxy = xywh2xyxy(dets_k[k_i][1]['bb'])
                    bb_t_xyxy = xywh2xyxy(dets_t[t_i][1]['bb'])
                    #print(frames[frame_id_k], '\n', frames[frame_id_t])
                    #print('k frame:', frame_id_k, 't frame:', frame_id_t)
                    #print('k bb:', dets_k[k_i][1]['bb'], 't_bb:', dets_t[t_i][1]['bb'])
                    #print('k bb:', bb_k_xyxy, 't_bb:', bb_t_xyxy)
                    frame_k = cv2.imread(frames[frame_id_k])[bb_k_xyxy[1]:bb_k_xyxy[3], bb_k_xyxy[0]:bb_k_xyxy[2], :]
                    frame_t = cv2.imread(frames[frame_id_t])[bb_t_xyxy[1]:bb_t_xyxy[3], bb_t_xyxy[0]:bb_t_xyxy[2], :]
                    # print(f'k shape: {frame_k.shape} t shape: {frame_t.shape}')
                    frame_pairs.append([frame_k, frame_t])
                #same = DeepFace.verify(img1_path=frame_k, img2_path=frame_t, 
                #                       model_name=model_name, model=model, 
                #                       enforce_detection=False, detector_backend='skip')['verified']
                DF_res = DeepFace.verify(img1_path=frame_pairs, 
                                         model_name=model_name, 
                                         model=model, 
                                         enforce_detection=False, 
                                         detector_backend='skip', 
                                         prog_bar=False)
                mean_verification_score = np.array([v['verified'] for _, v in DF_res.items()], dtype=np.float32).mean()
                #for i, (k, v) in enumerate(DF_res.items()):
                #    if v['verified'] == True:
                #        cv2.imwrite(f'FN_{k}_{t}_{i}.jpg', 
                #                    np.concatenate((cv2.resize(frame_pairs[i][0], (200, 200), interpolation=cv2.INTER_AREA),
                #                                    cv2.resize(frame_pairs[i][1], (200, 200), interpolation=cv2.INTER_AREA)), axis=0))
                if mean_verification_score > threshold:
                    # merge tracks
                    self.tracks[k].merge(self.tracks[t])
                    self.tracks.pop(t)
                    blacklist.append(t)
                    print(len(blacklist), '/', len(track_ids), '\r', end='', flush=True)
            blacklist.append(k)
            print(len(blacklist), '/', len(track_ids), '\r', end='', flush=True)
        return self


    def merge_iou(self, max_lost: int = 125, iou_thr: float = 0.2):
        print('merge iou')
        track_ids = list(self.tracks.keys())
        if len(track_ids) == 1: return self
        blacklist = []
        for k in track_ids:
            if k in blacklist: continue
            for t in track_ids:
                if k == t or t in blacklist: continue

                track_k = self.tracks[k]
                track_t = self.tracks[t]

                # get track's last timestep's detection
                track_k_last = sorted(list(track_k.location.keys()))[-1]
                track_t_first = sorted(list(track_t.location.keys()))[0]

                # same frame, different detection -> skip assignment
                if track_k_last == track_t_first:
                    continue

                # merge tracks if within lost frame tolerance and IoU threshold is met
                if abs(track_t_first - track_k_last) < max_lost and \
                   iou_xywh(track_k.location[track_k_last]['bb'], track_t.location[track_t_first]['bb']) > iou_thr:

                    # merge tracks
                    self.tracks[k].merge(self.tracks[t])
                    self.tracks.pop(t)
                    blacklist.append(t)
            blacklist.append(k)
        return self

    def filter_min_length(self, min_length: int = 250):
        print('filter, before, minl:', sorted([(k, len(track)) for k, track in self.tracks.items()], key=lambda x: x[1], reverse=True))
        keep = sorted([(k, len(track)) for k, track in self.tracks.items() if len(track) > min_length], key=lambda x: x[1], reverse=True)
        print('filter, after, minl:', keep)
        keep_ids = [k for k, _ in keep]
        for k in list(self.tracks.keys()):
            if k not in keep_ids:
                self.tracks.pop(k)
        return self

    def filter_topk_length(self, top_k: int = 1):
        keep = sorted([(k, len(track)) for k, track in self.tracks.items()], key=lambda x: x[1], reverse=True)
        print('filter, topk, keep:', keep)
        keep_ids = [k for k, _ in keep[:top_k]]
        for k in list(self.tracks.keys()):
            if k not in keep_ids:
                self.tracks.pop(k)
        return self

    def select_center(self, center_point: tuple, top_k: int = 1):
        center_point = np.array(center_point)
        keep = sorted([(k, np.linalg.norm(track.center()-center_point)) for k, track in self.tracks.items()], key=lambda x: x[1], reverse=False)
        print('center', center_point[0], center_point[1], 'keep:', keep)
        keep_ids = [k for k, _ in keep[:top_k]]
        for k in list(self.tracks.keys()):
            if k not in keep_ids:
                self.tracks.pop(k)
        return self


    def save(self, frames: list, output_dir: str, sample_every_n: int = 1):
        print(f'Save faces to {output_dir}...')
        assert len(frames) > 0
        h, w, _ = cv2.imread(frames[0])
        for frame_ind, frame_path in tqdm(enumerate(frames), total=len(frames)):
            if frame_ind % sample_every_n != 0: continue
            frame = cv2.imread(frame_path)

            for _, track in self.tracks.items():
                if not frame_ind in track.location:
                    frame = np.zeros((h, w, 3))
                    cv2.imwrite(str(Path(output_dir) / 'frame_{:05d}.png'.format(frame_ind)), frame)
                else:
                    bb_size = track.bb_size()
                    detection = track.location[frame_ind]
                    bb_xyxy = xywh2xyxy(detection['bb'])
                    # centering
                    cx, cy = bb_xyxy[0]-bb_xyxy[2], bb_xyxy[1]-bb_xyxy[3]
                    face_bb_xyxy = np.rint(np.array([cx-bb_size//2, cy-bb_size//2, 
                                                     cx+bb_size//2, cx-bb_size//2]))
                    # correct if necessary
                    face_bb_xyxy[face_bb_xyxy < 0] = 0
                    face_bb_xyxy[face_bb_xyxy[0] > w] = w
                    face_bb_xyxy[face_bb_xyxy[2] > w] = w
                    face_bb_xyxy[face_bb_xyxy[1] > h] = h
                    face_bb_xyxy[face_bb_xyxy[3] > h] = h
                    # cut face
                    x1, y1, x2, y2 = face_bb_xyxy
                    face = frame[y1:y2, x1:x2, :]
                    cv2.imwrite(str(Path(output_dir) / 'frame_{:05d}.png'.format(frame_ind)), face)

def detection_visualization(frame_paths: list, detections: RetinafaceDetections, output_dir: str, sample_every_n: int = 25):
    print(f'Save detections to {output_dir}...')
    for frame_ind, frame_path in tqdm(enumerate(frame_paths), total=len(frame_paths)):
        frame = cv2.imread(frame_path)
        for detection in detections[::sample_every_n]:
            if int(detection['frame']) == frame_ind:
                bb_xyxy = xywh2xyxy(detection['bb'])
                cv2.putText(frame, "score: {:.2f}".format(detection['score']), (bb_xyxy[0]-5, bb_xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (bb_xyxy[0], bb_xyxy[1]), (bb_xyxy[2], bb_xyxy[3]), (0,255,0), 2)
                cv2.imwrite(str(Path(output_dir) / 'frame_{:05d}.png'.format(frame_ind)), frame)


def face_visualization(frames: list, tracks: OrderedDict, output_dir: str, sample_every_n: int = 1, extra_percent: float = 0.2):
    print(f'Save faces to {output_dir}...')
    if Path(output_dir).exists(): return # skip already done samples
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    assert len(frames) > 0
    h, w, _ = cv2.imread(frames[0]).shape
    for frame_ind, frame_path in tqdm(enumerate(frames), total=len(frames)):
        #if frame_ind < 27861: continue
        if frame_ind % sample_every_n != 0: continue
        frame = cv2.imread(frame_path)
        output_file = Path(output_dir) / 'frame_{:05d}.png'.format(frame_ind)
        #if output_file.exists():
        #    f = cv2.imread(str(output_file))
        for _, track in tracks.items():
            bb_size = track.bb_size(extra_percent=extra_percent)
            if bb_size % 2 == 1: bb_size -= 1 # make it even
            #if f.shape == (bb_size, bb_size, 3): continue
            #print('bb size:', bb_size)
            if not frame_ind in track.location:
                frame = np.zeros((bb_size, bb_size, 3))
                cv2.imwrite(str(Path(output_dir) / 'frame_{:05d}.png'.format(frame_ind)), frame)
            else:
                detection = track.location[frame_ind]
                bb_xyxy = xywh2xyxy(detection['bb'])
                #print('detected bb:', bb_xyxy)
                # centering
                cx = bb_xyxy[0] + abs(bb_xyxy[2]-bb_xyxy[0])//2
                cy = bb_xyxy[1] + abs(bb_xyxy[3]-bb_xyxy[1])//2
                #print('center of bb:', cx, cy)
                face_bb_xyxy = np.rint(np.array([cx-bb_size//2, cy-bb_size//2, 
                                                 cx+bb_size//2, cy+bb_size//2])).astype(np.int32)
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
                    sh, sw = (bb_size - face.shape[0]) // 2, (bb_size - face.shape[1]) // 2
                    face_resized[sh:sh+face.shape[0], sw:sw+face.shape[1], :] = face
                cv2.imwrite(str(Path(output_dir) / 'frame_{:05d}.png'.format(frame_ind)), face)


def track_visualization(frame_paths: list, tracks: OrderedDict, output_dir: str, sample_every_n: int = 25):
    print(f'Save tracks to {output_dir}...')
    for frame_ind, frame_path in tqdm(enumerate(frame_paths), total=len(frame_paths)):
        if frame_ind % sample_every_n != 0: continue
        frame = cv2.imread(frame_path)
        
        fx, fy = frame.shape[1], frame.shape[0]
        #frame_center = (int(fx//2), int(fy//2))

        for _, track in tracks.items():
            if not frame_ind in track.location: continue
            detection = track.location[frame_ind]
            bb_xyxy = xywh2xyxy(detection['bb'])
            
            frame = bbv.draw_rectangle(frame, bb_xyxy)
            frame = bbv.add_label(frame, "{}|{:2d}".format(track.id, int(detection['score']*100)), bb_xyxy)

            #cx, cy = track.center()
            #print(frame.shape, fx, fy)
            #track_center = (int(cx), int(cy))
            #print(center_point)
            #import sys;sys.exit()
            #print(frame_ind, track_center, frame_center)
            #frame = cv2.circle(frame, track_center, 1, (0,0,255), -1)
            #frame = cv2.arrowedLine(frame, track_center, frame_center, (0,0,150), 1)

            cv2.imwrite(str(Path(output_dir) / 'frame_{:05d}.png'.format(frame_ind)), frame)


@load_or_create('det')
def detect_faces(frame_paths: list, detector: RetinaFace, batch_size: int = 32, **kwargs):
    detections = RetinafaceDetections()
    for batch_ind in tqdm(range(0, len(frame_paths), batch_size)):
        imgs = [cv2.imread(frame) for frame in frame_paths[batch_ind:batch_ind+batch_size]]
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
                # note: y is row, x is column
                box = np.rint(np.array(box)).astype(np.int32)
                box = np.where(box < 0, np.zeros_like(box), box) # negative index outside of the picture
                xywh = xyxy2xywh(box).astype(np.int32)
                landmarks = np.rint(np.array(landmarks)).astype(np.int32)
                detections.add({'frame': batch_ind+frame, 'score': score, 'bb': xywh, 'landmarks': landmarks})
    return detections


@timer_with_return
@load_or_create('pkl')
def find_and_track_deepface(frames_dir: str,
                            num_tracks: int = 2,
                            batch_size: int = 32,
                            gpu_id: int = 0,
                            retinaface_arch: str = 'resnet50',
                            cache_det: str = 'test.det',
                            visualize_det: bool = False,
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
    assert retinaface_arch in ['mobilenet', 'resnet50'], 'Invalid architecture choice for RetinaFace. Choose from ["mobilenet","resnet50"].'

    # get frames
    frames = [str(Path(frames_dir) / frame) for frame in sorted(os.listdir(str(frames_dir)))]
    # detect face bounding boxes
    detections = detect_faces(frame_paths=frames, detector=RetinaFace(gpu_id=gpu_id, network=retinaface_arch), batch_size=batch_size, output_path=cache_det)

    #if visualize_det:
    #    Path(./).mkdir(parents=True, exist_ok=True)
    #    detection_visualization(frames, detections, detection_dir)
    #    frames2video(detection_dir, Path(detection_dir).parent / f'{Path(detection_dir).stem}.mp4')

    h, w, _ = cv2.imread(frames[0]).shape
    cx = w // 2
    cy = h // 2

    print('Run tracker...')
    # label, interpolate, merge and filter tracks
    tracker = IoUTracker().label_iou(detections) \
                          .interpolate() \
                          .filter_min_length() \
                          .merge_deepface(frames) \
                          .filter_topk_length(top_k=num_tracks) \
                          .select_center((cx, cy))

    print('[postprocess] number of tracks:', len(tracker.tracks),
          'lengths of tracks:', [(id, len(track)) for id, track in tracker.tracks.items()])


    #if track_dir is not None:
    #    Path(track_dir).mkdir(parents=True, exist_ok=True)
    #    track_visualization(frames, tracker.tracks, track_dir)
    #    frames2video(track_dir, Path(track_dir).parent / f'{Path(track_dir).stem}.mp4')

    #break
    #landmarks = np.rint(np.array(landmarks)).astype(np.int32)[:,[1, 0]]   
    #aligned_face = face_alignment(imgs[i], landmarks[0,:], landmarks[1,:], landmarks[2,:])
    #
    #cv2.imwrite(f'{output_dir}/{Path(frames[i]).name}', aligned_face)
    #print(f'{output_dir}/{Path(frames[i]).name}')
    #img = img[:,:,::-1] # BGR
    # break
    #for img in imgs:
    #    cv2.imwrite(f'{output_dir}/{Path(frames[i]).name}', img)
    return tracker.tracks


@timer_with_return
@load_or_create('pkl')
def find_and_track_iou(frames_dir: str,
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
    assert retinaface_arch in ['mobilenet', 'resnet50'], 'Invalid architecture choice for RetinaFace. Choose from ["mobilenet","resnet50"].'

    # get frames
    frames = [str(Path(frames_dir) / frame) for frame in sorted(os.listdir(str(frames_dir)))]
    # detect face bounding boxes
    detections = detect_faces(frame_paths=frames, detector=RetinaFace(gpu_id=gpu_id, network=retinaface_arch), batch_size=batch_size, output_path=cache_det)

    h, w, _ = cv2.imread(frames[0]).shape
    cx = w // 2
    cy = h // 2

    print('Run tracker...')
    # label, interpolate, merge and filter tracks
    tracker = IoUTracker().label_iou(detections) \
                          .interpolate() \
                          .merge_iou(max_lost=125) \
                          .filter_topk_length(top_k=num_tracks) \
                          .select_center((cx, cy))

    print('[postprocess] number of tracks:', len(tracker.tracks),
          'lengths of tracks:', [(id, len(track)) for id, track in tracker.tracks.items()])

    return tracker.tracks


def nms(boxes, scores, overlapThresh, classes=None):
    """
    perform non-maximum suppression. based on Malisiewicz et al.
    Args:
        boxes (numpy.ndarray): boxes to process
        scores (numpy.ndarray): corresponding scores for each box
        overlapThresh (float): overlap threshold for boxes to merge
        classes (numpy.ndarray, optional): class ids for each box.

    Returns:
        (tuple): tuple containing:

        boxes (list): nms boxes
        scores (list): nms scores
        classes (list, optional): nms classes if specified
    """
    # # if there are no boxes, return an empty list
    # if len(boxes) == 0:
    #     return [], [], [] if classes else [], []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    if scores.dtype.kind == "i":
        scores = scores.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    #score = boxes[:, 4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    if classes is not None:
        return boxes[pick], scores[pick], classes[pick]
    else:
        return boxes[pick], scores[pick]

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


if __name__ == '__main__':
    test_face_alignment()
    quit()
    find_and_track_iou(frames_dir='data/processed/frames/002003_FC1_A_360p',
                       cache_det='data/processed/faces/002003_FC1_A_360p.det',
                       gpu_id=2)

    detection_dir='data/processed/faces/002003_FC1_A_360p'
    track_dir='data/processed/tracks/002003_FC1_A_360p'

    #if detection_dir is not None:
    #    Path(detection_dir).mkdir(parents=True, exist_ok=True)
    #    detection_visualization(frames, detections, detection_dir)
    #    frames2video(detection_dir, Path(detection_dir).parent / f'{Path(detection_dir).stem}.mp4')
    #
    #if track_dir is not None:
    #    Path(track_dir).mkdir(parents=True, exist_ok=True)
    #    track_visualization(frames, tracker.tracks, track_dir)
    #    frames2video(track_dir, Path(track_dir).parent / f'{Path(track_dir).stem}.mp4')