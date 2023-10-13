from math import sqrt
import cv2
import numpy as np


def __calc_hypotenuse(pts):
    if pts.ndim != 2:
        raise Exception(f'Expected shape is (68, C) or (C, 68) where C in {{2,3}}, got instead {pts.shape}')

    if pts.shape[0] == 68:
        pts = pts.T

    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def __build_camera_box(rear_size=90):
    point_3d = []
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    front_size = int(4 / 3 * rear_size)
    front_depth = int(4 / 3 * rear_size)
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)
    return point_3d


def draw_pose_box(img: np.ndarray, P: np.ndarray, landmarks: np.ndarray, color: tuple[int,int,int] = (40, 255, 0), line_width: int = 2) -> np.ndarray:
    """ Draw a 3D box as annotation of pose.

    Example:
        >>> image = draw_pose_box(image, camera_matrix, landmarks)

    Reference: https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py

    Args:
        img: the input image with shape (H, W, C) and BGR channel order.
        P: Affine Camera Matrix with shape (3, 4).
        landmarks: (68, 2) or (68, 3)

    Returns:
        (np.ndarray):
    """
    llength = __calc_hypotenuse(landmarks)
    point_3d = __build_camera_box(llength)
    # Map to 2d image points
    point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
    point_2d = point_3d_homo.dot(P.T)[:, :2]

    point_2d[:, 1] = - point_2d[:, 1]
    point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(landmarks[:27, :2], 1)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    img = img.astype(np.uint8)

    # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    return img