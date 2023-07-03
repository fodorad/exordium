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


def xywh2mid(bb_xywh: np.ndarray):
    """
    Calculate centroids for multiple bounding boxes.

    Args:
        bb_xywh (numpy.ndarray): Array of shape `(n, 4)` or of shape `(4,)` where
            each row contains `(xmin, ymin, width, height)`.

    Returns:
        numpy.ndarray: Centroid (x, y) coordinates of shape `(n, 2)` or `(2,)`.

    """
    if bb_xywh.ndim == 1:
        bb_xywh = bb_xywh[None, :]

    xmin, ymin, w, h = bb_xywh[:, 0], bb_xywh[:, 1], bb_xywh[:, 2], bb_xywh[:, 3]

    xc = xmin + 0.5 * w
    yc = ymin + 0.5 * h

    return np.column_stack((xc, yc))


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

    return size_intersection / size_union # IoU


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

    return iou(bbox1, bbox2)


def xyxy2xywh(xyxy):
    """
    Convert bounding box coordinates from (xmin, ymin, xmax, ymax) format to (xmin, ymin, width, height).

    Args:
        xyxy (numpy.ndarray):

    Returns:
        numpy.ndarray: Bounding box coordinates (xmin, ymin, width, height).

    """
    if xyxy.ndim == 1:
        left, top, right, bottom = xyxy
        width = right - left + 1
        height = bottom - top + 1
        return np.array([left, top, width, height], dtype='int')

    elif xyxy.ndim == 2:
        width = xyxy[:, 2] - xyxy[:, 0] + 1
        height = xyxy[:, 3] - xyxy[:, 1] + 1
        return np.column_stack((xyxy[:, 0], xyxy[:, 1], width, height)).astype("int")

    else:
        raise ValueError("Input shape not compatible.")


def xywh2xyxy(bb_xywh: np.ndarray):
    """
    Convert bounding box coordinates from (xmin, ymin, width, height) to (xmin, ymin, xmax, ymax) format.

    Args:
        xywh (numpy.ndarray): Bounding box coordinates as `(xmin, ymin, width, height)`.

    Returns:
        numpy.ndarray : Bounding box coordinates as `(xmin, ymin, xmax, ymax)`.

    """

    if bb_xywh.ndim == 1:
        x, y, w, h = bb_xywh
        xmax = x + w
        ymax = y + h
        return np.array([x, y, xmax, ymax], dtype='int')

    elif bb_xywh.ndim == 2:
        x, y, w, h = bb_xywh[:, 0], bb_xywh[:, 1], bb_xywh[:, 2], bb_xywh[:, 3]
        xmax = x + w
        ymax = y + h
        return np.column_stack((x, y, xmax, ymax)).astype('int')  # xyxy
    
    else:
        raise ValueError("Input shape not compatible.")

def midwh2xywh(midwh: np.ndarray):
    """
    Convert bounding box coordinates from (xmid, ymid, width, height) to (xmin, ymin, width, height) format.

    Args:
        midwh (numpy.ndarray): Bounding box coordinates (xmid, ymid, width, height).

    Returns:
        numpy.ndarray: Bounding box coordinates (xmin, ymin, width, height).
    """
    if midwh.ndim == 1:
        xmid, ymid, w, h = midwh
        return np.array([xmid - w * 0.5, ymid - h * 0.5, w, h], dtype='int')  # xywh

    elif midwh.ndim == 2:
        xymin = midwh[:, :2] - midwh[:, 2:] * 0.5
        wh = midwh[:, 2:]
        return np.concatenate([xymin, wh], axis=1).astype('int')  # xywh

    else:
        raise ValueError("Input shape not compatible.")


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

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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

        # Delete indexes from the index list that have overlap above the threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    if classes is not None:
        return boxes[pick], scores[pick], classes[pick]
    else:
        return boxes[pick], scores[pick]