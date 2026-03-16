import cv2
import numpy as np

from exordium.video.core.bb import xyxy2full
from exordium.video.face.landmark.constants import FaceLandmarks
from exordium.video.face.landmark.facemesh import rotate_landmarks


def rotate_face(face: np.ndarray, rotation_degree: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a face image by an explicit angle.

    Args:
        face: Face image of shape ``(H, W, 3)``.
        rotation_degree: Rotation in degrees.

    Returns:
        Tuple of ``(rotated_face, rotation_matrix)``.

    """
    if rotation_degree == 0:
        return face, np.eye(2)
    height, width = face.shape[:2]
    R = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_degree, 1)
    face_rotated = cv2.warpAffine(face, R, (width, height))
    return face_rotated, R


def align_face(
    image: np.ndarray, bb_xyxy: np.ndarray, landmarks: np.ndarray
) -> dict[str, np.ndarray]:
    """Align face image to canonical orientation using landmarks.

    Aligns the x-axis of the Head Coordinate System (HCS) to the
    x-axis of the Camera Coordinate System (CCS). The face is rotated
    based on the eye positions to align it horizontally.

    Args:
        image: Input image of shape (H, W, C).
        bb_xyxy: Input bounding box of shape (4,).
        landmarks: Input landmarks of shape (5, 2).

    Raises:
        Exception: If landmarks do not have shape (5, 2).

    Returns:
        Dictionary with keys: rotated_image, rotated_face, rotated_bb_xyxy,
        rotation_degree, and rotation_matrix.

    """
    if landmarks.shape != (5, 2):
        raise Exception(f"Expected landmarks with shape (5, 2) got instead {landmarks.shape}.")

    landmarks = np.rint(landmarks).astype(int)
    left_eye_x, left_eye_y = landmarks[FaceLandmarks.LEFT_EYE.value, :]
    right_eye_x, right_eye_y = landmarks[FaceLandmarks.RIGHT_EYE.value, :]

    dY = right_eye_y - left_eye_y
    dX = right_eye_x - left_eye_x
    rotation_degree = np.degrees(np.arctan2(dY, dX)) - 180

    height, width = image.shape[:2]
    image_center = (width // 2, height // 2)
    R = cv2.getRotationMatrix2D(image_center, rotation_degree, 1.0)
    abs_cos = abs(R[0, 0])
    abs_sin = abs(R[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    R[0, 2] += bound_w / 2 - image_center[0]
    R[1, 2] += bound_h / 2 - image_center[1]
    rotated_image = cv2.warpAffine(image, R, (bound_w, bound_h))

    rotated_bb_full = rotate_landmarks(xyxy2full(bb_xyxy), R)
    min_x, min_y = np.min(rotated_bb_full, axis=0)
    max_x, max_y = np.max(rotated_bb_full, axis=0)
    rotated_bb_xyxy = np.array([min_x, min_y, max_x, max_y], dtype=int)
    rotated_face = rotated_image[min_y:max_y, min_x:max_x]

    return {
        "rotated_image": rotated_image,
        "rotated_face": rotated_face,
        "rotated_bb_xyxy": rotated_bb_xyxy,
        "rotation_degree": rotation_degree,
        "rotation_matrix": R,
    }


def crop_eye_keep_ratio(
    img: np.ndarray, landmarks: np.ndarray, bb: tuple[int, int] = (36, 60)
) -> np.ndarray:
    """Crop an eye region from an image preserving the given aspect ratio.

    Args:
        img: Input image of shape ``(H, W, C)``.
        landmarks: Eye landmark coordinates of shape ``(N, 2)``.
        bb: Target output size as ``(height, width)``. Defaults to ``(36, 60)``.

    Returns:
        Cropped and resized eye image of shape ``(bb[0], bb[1], C)``.

    """
    xy_m = np.mean(landmarks, axis=0)
    ratio = bb[0] / bb[1]
    xy_dx = np.linalg.norm(landmarks[0, :] - landmarks[3, :]) * 2
    xy_dy = xy_dx * ratio
    eye = img[
        int(xy_m[1] - xy_dy // 2) : int(xy_m[1] + xy_dy // 2),
        int(xy_m[0] - xy_dx // 2) : int(xy_m[0] + xy_dx // 2),
        :,
    ]
    return cv2.resize(eye, (bb[1], bb[0]), interpolation=cv2.INTER_AREA)
