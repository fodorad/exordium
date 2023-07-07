import numpy as np


def crop_mid(image: np.ndarray, mid: np.ndarray, bb_size: int) -> np.ndarray:
    image_height, image_width, _ = image.shape
    half_size = bb_size // 2

    # Adjust bounding box coordinates if they exceed image boundaries
    x1 = np.clip(mid[0] - half_size, 0, image_width)
    y1 = np.clip(mid[1] - half_size, 0, image_height)
    x2 = np.clip(mid[0] + half_size, 0, image_width)
    y2 = np.clip(mid[1] + half_size, 0, image_height)

    # Crop the image based on adjusted bounding box coordinates
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def crop_xyxy(image: np.ndarray, bb_xyxy: np.ndarray) -> np.ndarray:
    image_height, image_width, _ = image.shape

    # Adjust bounding box coordinates if they exceed image boundaries
    x1 = np.clip(bb_xyxy[0], 0, image_width)
    y1 = np.clip(bb_xyxy[1], 0, image_height)
    x2 = np.clip(bb_xyxy[2], 0, image_width)
    y2 = np.clip(bb_xyxy[3], 0, image_height)

    # Crop the image based on adjusted bounding box coordinates
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def iou_xywh(bb_xywh1: np.ndarray, bb_xywh2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes in xyxy format.

    Args:
        bb_xyxy1 (numpy.ndarray): First bounding box in xyxy format.
        bb_xyxy2 (numpy.ndarray): Second bounding box in xyxy format.

    Returns:
        float: Intersection over Union (IoU) value.

    """

    # Convert xywh format to xyxy format
    bb_xyxy1 = xywh2xyxy(bb_xywh1)
    bb_xyxy2 = xywh2xyxy(bb_xywh2)

    # Calculate the coordinates of the intersection rectangle
    x1_inter = np.maximum(bb_xyxy1[0], bb_xyxy2[0])
    y1_inter = np.maximum(bb_xyxy1[1], bb_xyxy2[1])
    x2_inter = np.minimum(bb_xyxy1[2], bb_xyxy2[2])
    y2_inter = np.minimum(bb_xyxy1[3], bb_xyxy2[3])

    # Calculate the areas of the two bounding boxes
    area1 = bb_xywh1[2] * bb_xywh1[3]
    area2 = bb_xywh2[2] * bb_xywh2[3]

    # Calculate the area of intersection rectangle
    intersection_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)

    # Calculate the Union area by subtracting the intersection area
    # and adding the areas of the two bounding boxes
    union_area = area1 + area2 - intersection_area

    # Calculate the Intersection over Union (IoU) value
    iou = intersection_area / union_area

    return iou


def iou_xyxy(bb_xyxy1: np.ndarray, bb_xyxy2: np.ndarray) -> float:
    bb_xywh1 = xyxy2xywh(bb_xyxy1)
    bb_xywh2 = xyxy2xywh(bb_xyxy2)
    return iou_xywh(bb_xywh1, bb_xywh2)


def xyxy2xywh(bb_xyxy: np.ndarray) -> np.ndarray:
    """
    Convert bounding box coordinates from (xmin, ymin, xmax, ymax) format to (xmin, ymin, width, height).

    Args:
        xyxy (numpy.ndarray):

    Returns:
        numpy.ndarray: Bounding box coordinates (xmin, ymin, width, height).

    """
    bb_xyxy = np.array(bb_xyxy)

    if bb_xyxy.ndim == 1:
        bb_xyxy = bb_xyxy[None, :]

    xmin, ymin, xmax, ymax = bb_xyxy[:, 0], bb_xyxy[:, 1], bb_xyxy[:, 2], bb_xyxy[:, 3]

    w = xmax - xmin
    h = ymax - ymin

    return np.round(np.column_stack((xmin, ymin, w, h))).astype(int).squeeze()


def xywh2xyxy(bb_xywh: np.ndarray) -> np.ndarray:
    """
    Convert bounding box coordinates from (xmin, ymin, width, height) to (xmin, ymin, xmax, ymax) format.

    Args:
        xywh (numpy.ndarray): Bounding box coordinates as `(xmin, ymin, width, height)`.

    Returns:
        numpy.ndarray : Bounding box coordinates as `(xmin, ymin, xmax, ymax)`.

    """
    bb_xywh = np.array(bb_xywh)

    if bb_xywh.ndim == 1:
        bb_xywh = bb_xywh[None, :]

    xmin, ymin, w, h = bb_xywh[:, 0], bb_xywh[:, 1], bb_xywh[:, 2], bb_xywh[:, 3]

    xmax = xmin + w
    ymax = ymin + h

    return np.round(np.column_stack((xmin, ymin, xmax, ymax))).astype(int).squeeze()


def midwh2xywh(bb_midwh: np.ndarray) -> np.ndarray:
    """
    Convert bounding box coordinates from (xmid, ymid, width, height) to (xmin, ymin, width, height) format.

    Args:
        midwh (numpy.ndarray): Bounding box coordinates (xmid, ymid, width, height).

    Returns:
        numpy.ndarray: Bounding box coordinates (xmin, ymin, width, height).
    """
    bb_midwh = np.array(bb_midwh)

    if bb_midwh.ndim == 1:
        bb_midwh = bb_midwh[None, :]

    xc, yc, w, h = bb_midwh[:, 0], bb_midwh[:, 1], bb_midwh[:, 2], bb_midwh[:, 3]

    xmin = xc - 0.5 * w
    ymin = yc - 0.5 * h

    return np.round(np.column_stack((xmin, ymin, w, h))).astype(int).squeeze()


def xywh2midwh(bb_xywh: np.ndarray) -> np.ndarray:
    """
    Calculate centroids for multiple bounding boxes.

    Args:
        bb_xywh (numpy.ndarray): List of 4 ints or array of shape `(n, 4)` or of shape `(4,)` where
            each row contains `(xmin, ymin, width, height)`.

    Returns:
        numpy.ndarray: Centroid (x, y) coordinates of shape `(n, 2)` or `(2,)`.

    """
    bb_xywh = np.array(bb_xywh)

    if bb_xywh.ndim == 1:
        bb_xywh = bb_xywh[None, :]

    xmin, ymin, w, h = bb_xywh[:, 0], bb_xywh[:, 1], bb_xywh[:, 2], bb_xywh[:, 3]

    xc = xmin + 0.5 * w
    yc = ymin + 0.5 * h

    return np.round(np.column_stack((xc, yc, w, h))).astype(int).squeeze()
