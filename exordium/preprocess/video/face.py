from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import bbox_visualizer as bbv


def face_alignment(img: np.ndarray,
                   landmarks: np.ndarray,
                   detector: str = 'mtcnn',
                   left_eye_ratio=(0.38, 0.38),
                   output_width=224,
                   output_height=None):
    # expected MTCNN implementation: https://github.com/timesler/facenet-pytorch

    assert landmarks.shape == (5,2), f'Expected: (5,2), got istead: {landmarks.shape}'
    assert detector in {'mtcnn'}, 'Only MTCNN format is supported right now.'

    # if the desired face height is None, set it to be the
    # desired face width (normal behavior)
    if output_height is None:
        output_height = output_width

    landmarks = np.rint(landmarks).astype(np.int32)

    left_eye_x, left_eye_y = landmarks[1,:] # participant's left eye
    right_eye_x, right_eye_y = landmarks[0,:] # participant's right eye

    # compute the angle between the eye centroids
    dY = right_eye_y - left_eye_y
    dX = right_eye_x - left_eye_x
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # compute center (x, y)-coordinates (i.e., the median point)
    # between the two eyes in the input image
    center = (int((left_eye_x + right_eye_x) // 2),
                  int((left_eye_y + right_eye_y) // 2))

    # compute the desired right eye x-coordinate based on the
    # desired x-coordinate of the left eye
    right_eye_ratio_x = 1.0 - left_eye_ratio[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the *desired* image
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    output_dist = (right_eye_ratio_x - left_eye_ratio[0])
    output_dist *= output_width
    scale = output_dist / dist

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(center, angle, scale)

    t_x = output_width * 0.5
    t_y = output_height * left_eye_ratio[1]
    M[0, 2] += (t_x - center[0])
    M[1, 2] += (t_y - center[1])

    # apply the affine transformation
    return cv2.warpAffine(img, M, (output_width, output_height), flags=cv2.INTER_CUBIC)


def face_alignment2(img: np.ndarray,
                   landmarks: np.ndarray,
                   bb_xyxy: np.ndarray = None,
                   detector: str = 'mtcnn'):
    # modified version of function in deepface repository:
    # https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py
    # left_eye, right_eye, nose: (x, y), (w, h)
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

    return np.array(Image.fromarray(img).rotate(direction*angle, resample=Image.BICUBIC)) # rotate


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
