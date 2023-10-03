import numpy as np


def crop_mid(image: np.ndarray, mid: np.ndarray, bb_size: int) -> np.ndarray:
    """Crops an image using the middle point and size of a bounding box.

    Args:
        image (np.ndarray): input image of shape (H, W, C).
        mid (np.array): middle point of shape (2,)
        bb_size (int): crop size.

    Returns:
        np.ndarray: cropped image.
    """
    image_height, image_width, _ = image.shape
    half_size = bb_size // 2

    # Adjust bounding box coordinates if they exceed image boundaries
    x1 = np.clip(mid[0] - half_size, 0, image_width)
    y1 = np.clip(mid[1] - half_size, 0, image_height)
    x2 = np.clip(mid[0] + half_size, 0, image_width)
    y2 = np.clip(mid[1] + half_size, 0, image_height)

    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def crop_xyxy(image: np.ndarray, bb_xyxy: np.ndarray) -> np.ndarray:
    """Crops an image using the top left and bottom right x, y coords of a bounding box.

    Args:
        image (np.ndarray): input image of shape (H, W, C).
        bb_xyxy (tuple[int, int]): bounding box of shape (4,).

    Returns:
        np.ndarray: cropped image.
    """
    image_height, image_width, _ = image.shape

    # Adjust bounding box coordinates if they exceed image boundaries
    x1 = np.clip(bb_xyxy[0], 0, image_width)
    y1 = np.clip(bb_xyxy[1], 0, image_height)
    x2 = np.clip(bb_xyxy[2], 0, image_width)
    y2 = np.clip(bb_xyxy[3], 0, image_height)

    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def center_crop(img: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    """Applies center crop to an image.

    Args:
        img (np.array): input image of shape (H, W, C).
        crop_size (tuple[int, int]): crop size.

    Returns:
        np.ndarray: cropped image.
    """
    c_h, c_w = img.shape[0] // 2, img.shape[1] // 2
    half_h, half_w = crop_size[0] // 2, crop_size[1] // 2
    return img[c_h-half_h:c_h+half_h, c_w-half_w:c_w+half_w, :]


def apply_10_crop(img: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
    """Applies 10-crop method to an image.

    Args:
        img (np.ndarray): input image of shape (H, W, C).
        crop_size (tuple[int, int]): crop size.

    Returns:
        np.ndarray: 10-crop output images of shape (10, crop_size[0], crop_size[1], C).
    """
    h = crop_size[0]
    w = crop_size[1]
    flipped_X = np.fliplr(img)
    crops = [
        img[:h,:w, :], # Upper Left
        img[:h, img.shape[1]-w:, :], # Upper Right
        img[img.shape[0]-h:, :w, :], # Lower Left
        img[img.shape[0]-h:, img.shape[1]-w:, :], # Lower Right
        center_crop(img, (h, w)), # Center
        flipped_X[:h,:w, :],
        flipped_X[:h, flipped_X.shape[1]-w:, :],
        flipped_X[flipped_X.shape[0]-h:, :w, :],
        flipped_X[flipped_X.shape[0]-h:, flipped_X.shape[1]-w:, :],
        center_crop(flipped_X, (h, w))
    ]
    return np.array(crops)


def iou_xywh(bb_xywh1: np.ndarray, bb_xywh2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) for two bounding boxes in xywh format.

    Args:
        bb_xywh1 (np.ndarray): bounding box in xywh format.
        bb_xywh2 (np.ndarray): bounding box in xywh format.

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
    """Calculate Intersection over Union (IoU) for two bounding boxes in xyxy format.

    Args:
        bb_xyxy1 (np.ndarray): bounding box in xyxy format.
        bb_xyxy2 (np.ndarray): bounding box in xyxy format.

    Returns:
        float: Intersection over Union (IoU) value.
    """
    bb_xywh1 = xyxy2xywh(bb_xyxy1)
    bb_xywh2 = xyxy2xywh(bb_xyxy2)
    return iou_xywh(bb_xywh1, bb_xywh2)


def xyxy2xywh(bb_xyxy: np.ndarray) -> np.ndarray:
    """Convert bounding box coordinates from (xmin, ymin, xmax, ymax) to (xmin, ymin, width, height).

    Args:
        xyxy (np.ndarray): bounding box in xyxy format.

    Returns:
        np.ndarray: bounding box coordinates in xywh format.
    """
    bb_xyxy = np.array(bb_xyxy)

    if bb_xyxy.ndim == 1:
        bb_xyxy = bb_xyxy[None, :]

    xmin, ymin, xmax, ymax = bb_xyxy[:, 0], bb_xyxy[:, 1], bb_xyxy[:, 2], bb_xyxy[:, 3]

    w = xmax - xmin
    h = ymax - ymin

    return np.round(np.column_stack((xmin, ymin, w, h))).astype(int).squeeze()


def xywh2xyxy(bb_xywh: np.ndarray) -> np.ndarray:
    """Convert bounding box coordinates from (xmin, ymin, width, height) to (xmin, ymin, xmax, ymax).

    Args:
        xywh (np.ndarray): bounding box in xywh format.

    Returns:
        np.ndarray : bounding box in xyxy format.
    """
    bb_xywh = np.array(bb_xywh)

    if bb_xywh.ndim == 1:
        bb_xywh = bb_xywh[None, :]

    xmin, ymin, w, h = bb_xywh[:, 0], bb_xywh[:, 1], bb_xywh[:, 2], bb_xywh[:, 3]

    xmax = xmin + w
    ymax = ymin + h

    return np.round(np.column_stack((xmin, ymin, xmax, ymax))).astype(int).squeeze()


def midwh2xywh(bb_midwh: np.ndarray) -> np.ndarray:
    """Convert bounding box coordinates from (xmid, ymid, width, height) to (xmin, ymin, width, height).

    Args:
        midwh (np.ndarray): bounding box in midwh format.

    Returns:
        np.ndarray: bounding box in xywh format.
    """
    bb_midwh = np.array(bb_midwh)

    if bb_midwh.ndim == 1:
        bb_midwh = bb_midwh[None, :]

    xc, yc, w, h = bb_midwh[:, 0], bb_midwh[:, 1], bb_midwh[:, 2], bb_midwh[:, 3]

    xmin = xc - 0.5 * w
    ymin = yc - 0.5 * h

    return np.round(np.column_stack((xmin, ymin, w, h))).astype(int).squeeze()


def xywh2midwh(bb_xywh: np.ndarray) -> np.ndarray:
    """Convert bounding box coordinates from (xmin, ymin, width, height) to (xmid, ymid, width, height).

    Args:
        bb_xywh (np.ndarray): bounding box in xywh format.

    Returns:
        np.ndarray: bounding box in midwh format.

    """
    bb_xywh = np.array(bb_xywh)

    if bb_xywh.ndim == 1:
        bb_xywh = bb_xywh[None, :]

    xmin, ymin, w, h = bb_xywh[:, 0], bb_xywh[:, 1], bb_xywh[:, 2], bb_xywh[:, 3]

    xc = xmin + 0.5 * w
    yc = ymin + 0.5 * h

    return np.round(np.column_stack((xc, yc, w, h))).astype(int).squeeze()