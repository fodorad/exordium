import os
import csv
import random
import itertools
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
from scipy import interpolate
import bbox_visualizer as bbv

from exordium.shared import timer, timer_with_return, load_or_create
from exordium.preprocess.video.frames import frames2video
from exordium.preprocess.video.bb import iou_xywh, xywh2xyxy, xyxy2xywh
from batch_face import RetinaFace # pip install git+https://github.com/elliottzheng/batch-face.git@master


class Track():

    def __init__(self, id: int, detection: dict, verbose: bool = False):
        # detection: score: float, bb: np.ndarray (xywh), landmarks: np.ndarray (5,2)
        self.id = id
        self.location = {}
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
        dets = [(k, v) for k, v in self.location.items() if v['score'] != -1]
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
            for line_count, record in enumerate(csv_reader):
                if line_count > 0:
                    self.detections.append({
                        'frame': int(record[0]),
                        'score': float(record[1]),
                        'bb': np.array(record[2:6], dtype=np.int32),
                        'landmarks': np.array(record[6:16], dtype=np.int32).reshape(5,2),
                    })
        return self


class IoUTracker():

    def __init__(self):
        self.new_track_id = 0
        self.tracks = {}

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

            if not tracks_to_add:
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
        assert frames, 'Empty list of frames'
        h, w, _ = cv2.imread(frames[0])
        for frame_ind, frame_path in tqdm(enumerate(frames), total=len(frames)):
            if frame_ind % sample_every_n != 0: continue
            frame = cv2.imread(frame_path)

            for _, track in self.tracks.items():
                if frame_ind not in track.location:
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
    assert frames, 'Empty list of frames'
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
            if frame_ind not in track.location:
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
        for _, track in tracks.items():
            if frame_ind not in track.location: continue
            detection = track.location[frame_ind]
            bb_xyxy = xywh2xyxy(detection['bb'])
            frame = bbv.draw_rectangle(frame, bb_xyxy)
            frame = bbv.add_label(frame, "{}|{:2d}".format(track.id, int(detection['score']*100)), bb_xyxy)
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
def find_and_track_deepface(frames_dir: str | Path,
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
    assert retinaface_arch in {'mobilenet', 'resnet50'}, 'Invalid architecture choice for RetinaFace. Choose from {"mobilenet","resnet50"}.'

    # get frames
    frames = [str(Path(frames_dir) / frame) for frame in sorted(os.listdir(frames_dir))]
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
def find_and_track_iou(frames_dir: str | Path,
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
    assert retinaface_arch in {'mobilenet', 'resnet50'}, 'Invalid architecture choice for RetinaFace. Choose from ["mobilenet","resnet50"].'

    # get frames
    frames = [str(Path(frames_dir) / frame) for frame in sorted(os.listdir(frames_dir))]
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


if __name__ == '__main__':
    find_and_track_iou(frames_dir='data/processed/frames/002003_FC1_A_360p',
                       cache_det='data/processed/faces/002003_FC1_A_360p.det',
                       gpu_id=2)

    detection_dir='data/processed/faces/002003_FC1_A_360p'
    track_dir='data/processed/tracks/002003_FC1_A_360p'

    #if detection_dir is not None:
    #    Path(detection_dir).mkdir(parents=True, exist_ok=True)
    #    detection_visualization(frames, detections, detection_dir)
    #    frames2video(detection_dir, Path(detection_dir).parent / f'{Path(detection_dir).stem}.mp4')

    #if track_dir is not None:
    #    Path(track_dir).mkdir(parents=True, exist_ok=True)
    #    track_visualization(frames, tracker.tracks, track_dir)
    #    frames2video(track_dir, Path(track_dir).parent / f'{Path(track_dir).stem}.mp4')