import os, sys
import json
from pathlib import Path
from typing import Tuple, List

from mtcnn import MTCNN
import cv2
import numpy as np
# import tensorflow as tf


def frame2face(input_path: str, output_path: str, extra_space: int = 0.1, resize_dim: int = 256) -> None:
    """Extracts the face with highest confidence value from the given image

    Args:
        input_path (str): input image path
        output_path (str): output image path
        extra_space (int, optional): extra space relative to the length of the bounding box. Defaults to 0.1.
        resize_dim (int, optional): resize output image. Defaults to 256.
    """
    if Path(output_path).exists(): return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(img)
    max_height, max_width, _ = img.shape
    if len(faces) > 0:
        ind = np.argmax(np.array([face['confidence'] for face in faces]), axis=0)
        bb = faces[ind]['box']
        side = max([bb[2], bb[3]])
        diff = abs(bb[2] - bb[3])
        extra = int(side * extra_space)
        w = bb[0] - extra // 2
        h = bb[1] - extra // 2
        if bb[2] < bb[3]:
            w -= diff // 2
        else:
            h -= diff // 2
        if w < 0: w = 0
        if h < 0: h = 0
        W = w + side + extra
        H = h + side + extra
        if W > max_width: W = max_width
        if H > max_height: H = max_height
        out = cv2.resize(img[h:H, w:W], (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def frame2pose(input_path: str, output_path: str) -> None:
    """Sends frame to the service and saves the returned pose
    First, run docker image
    docker run -it -p 5000:5000 quay.io/codait/max-human-pose-estimator

    Args:
        data (Tuple[str, str]): frame path, pose path
    """
    CMD = f'curl -F "file=@{input_path}" -XPOST http://localhost:5000/model/predict > {output_path}'
    print(CMD)
    os.system(CMD)


def visualize_pose(frame_path: str, pose_path: str, output_path: str) -> None:
    """Visualize pose coords on the frame

    Args:
        frame_path (str): frame path
        pose_path (str): pose path
        output_path (str): output path
    """
    img = cv2.imread(frame_path)
    with open(pose_path) as f:
        pose = json.load(f)
    body_parts = pose['predictions'][0]['body_parts']
    colors = {'Nose': [255,0,0], 'Neck': [0,255,0], 'RShoulder': [0,0,255], 'LShoulder': [0,255,255], 'LEye': [128, 128, 0], 'REye': [128, 0, 128], 'LEar': [128, 0, 128], 'REar': [0, 0, 0]}
    for element in body_parts:
        if element['part_name'] in list(colors.keys()):
            img[element['y']-2:element['y']+2, element['x']-2:element['x']+2] = colors[element['part_name']]
    cv2.imwrite(output_path, img)


def center_crop(x: np.ndarray, center_crop_size: Tuple[int, int], **kwargs) -> np.ndarray:
    """Center crop

    Args:
        x (np.array): input image
        center_crop_size (Tuple[int, int]): crop size

    Returns:
        np.ndarray: cropped image
    """
    centerh, centerw = x.shape[0]//2, x.shape[1]//2
    halfh, halfw = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerh-halfh:centerh+halfh, centerw-halfw:centerw+halfw, :]


def apply_10_crop(img: np.ndarray, crop_size: Tuple[int, int] = (224,224)) -> np.ndarray:
    """Applies 10-crop method to image

    Args:
        img (np.ndarray): image. Expected shape is (H,W,C)
        crop_size (Tuple[int, ...], optional): crop size. Defaults to (224,224).

    Returns:
        np.ndarray: 10-crop with shape (10,crop_size[0],crop_size[1],C)
    """
    h = crop_size[0]
    w = crop_size[1]
    flipped_X = np.fliplr(img)
    crops = [
        img[:h,:w, :], # Upper Left
        img[:h, img.shape[1]-w:, :], # Upper Right
        img[img.shape[0]-h:, :w, :], # Lower Left
        img[img.shape[0]-h:, img.shape[1]-w:, :], # Lower Right
        center_crop(img, (h, w)),

        flipped_X[:h,:w, :],
        flipped_X[:h, flipped_X.shape[1]-w:, :],
        flipped_X[flipped_X.shape[0]-h:, :w, :],
        flipped_X[flipped_X.shape[0]-h:, flipped_X.shape[1]-w:, :],
        center_crop(flipped_X, (h, w))
    ]
    return np.array(crops)


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()
        if p_1 > p:
            return input_img
        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            if left + w <= img_w and top + h <= img_h:
                break
        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c
        return input_img
    return eraser


# @tf.function
def tf_crop_center(image):
    """Returns a cropped square image."""
    shape = tf.shape(image)
    new_shape = 128
    offset_y = tf.maximum(shape[0] - shape[1], 0) // 2
    offset_x = tf.maximum(shape[1] - shape[0], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image