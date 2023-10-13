import cv2
import numpy as np
from exordium.video.gaze import pitchyaw_to_pixel
from exordium.video.transform import rotate_vector


def draw_vector(image: np.ndarray, origin: np.ndarray, end_point: np.ndarray) -> np.ndarray:
    """Draws an arrow to the image.

    Args:
        image (np.ndarray): image of shape (H, W, C).
        origin (np.ndarray): XY coords.
        end_point (np.ndarray): XY coords.

    Returns:
        np.ndarray: image with a vector drawn on it.
    """
    image_out = np.copy(image)
    end_point = tuple(np.round([origin[0] + end_point[0],
                                origin[1] + end_point[1]]).astype(int))
    cv2.arrowedLine(image_out, tuple(np.round(origin).astype(np.int32)), end_point,
                    (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.18)
    return image_out


def convert_rotate_draw_vector(image: np.ndarray,
                               yaw: float,
                               pitch: float,
                               rotation_degree: float,
                               origin: np.ndarray,
                               length: float = 400) -> np.ndarray:
    """Converts the yaw and pitch angles to XY coords, rotates the vector, then draws to the image.

    Args:
        image (np.ndarray): image to draw on.
        yaw (float): yaw angle in degrees.
        pitch (float): pitch angle in degrees.
        rotation_degree (float): rotation for the vector in degrees.
        origin (np.ndarray): origin XY coords of the vector.
        length (float, optional): length of the vector. Defaults to 400.

    Returns:
        np.ndarray: image with the rotated vector.
    """
    xy = pitchyaw_to_pixel(pitch, yaw, length)
    xy_rot = rotate_vector(xy, rotation_degree)
    image_out = draw_vector(image, origin, xy_rot)
    return image_out


def convert_draw_vector(image: np.ndarray,
                        yaw: float,
                        pitch: float,
                        origin: np.ndarray,
                        length: float = 200) -> np.ndarray:
    """Converts the yaw and pitch angles to XY coords, then draws to the image.

    Args:
        image (np.ndarray): image to draw on.
        yaw (float): yaw angle in degrees.
        pitch (float): pitch angle in degrees.
        origin (np.ndarray): origin XY coords of the vector.
        length (float, optional): length of the vector. Defaults to 200.

    Returns:
        np.ndarray: image with the vector.
    """
    xy = pitchyaw_to_pixel(pitch, yaw, length)
    image_out = draw_vector(image, origin, xy)
    return image_out


def visualize_target_gaze(frame: np.ndarray,
                          aligned_face: np.ndarray,
                          gaze_vector_normed: np.ndarray,
                          gaze_vector: np.ndarray,
                          gaze_vector_origin: np.ndarray,
                          threshold: float = 0.45) -> np.ndarray:
    frame_out = np.copy(frame)
    target_size = frame_out.shape[0] // 3
    face_ratio = max(aligned_face.shape[:2]) / target_size
    # draw normed
    target_face = cv2.resize(aligned_face, (448, 448), interpolation=cv2.INTER_AREA)
    target_face_radius = target_face.shape[0] // 2 # max gaze vector length
    target_face_origin = tuple(np.array(target_face.shape[:2]) // 2)
    target_face = cv2.circle(target_face, target_face_origin, int(target_face_radius), (0,255,0), 2)
    target_face = cv2.circle(target_face, target_face_origin, int(target_face_radius * threshold), (255,0,0), 2)
    target_face = cv2.circle(target_face, target_face_origin, 2, (0,255,0), 2)
    target_face = draw_vector(target_face, target_face_origin, gaze_vector_normed * target_face_radius)
    target_face = cv2.resize(target_face, (target_size, target_size), interpolation=cv2.INTER_AREA)
    frame_out[-target_size:, :target_size,:] = target_face
    # draw original
    frame_out = cv2.circle(frame_out, gaze_vector_origin, int(target_face_radius * face_ratio), (0,255,0), 2)
    frame_out = cv2.circle(frame_out, gaze_vector_origin, int(target_face_radius * face_ratio * threshold), (255,0,0), 2)
    frame_out = draw_vector(frame_out, gaze_vector_origin, gaze_vector * face_ratio * target_face_radius)
    return frame_out

def visualize_normed_space(image: np.ndarray, aligned_face: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    image_out = np.copy(image)
    H, _ = image.shape[:2]
    target_size = H // 3
    length = 200
    face_out = cv2.resize(aligned_face, (448, 448), interpolation=cv2.INTER_AREA)
    center = tuple(np.array(face_out.shape[:2]) // 2)
    face_out = cv2.circle(face_out, center, length, (0,255,0), 2)
    face_out = cv2.circle(face_out, center, length // 2, (255,0,0), 2)
    face_out = cv2.circle(face_out, center, 2, (0,255,0), 2)
    face_out = convert_draw_vector(face_out, yaw, pitch, center, length)
    face_out = cv2.resize(face_out, (target_size, target_size), interpolation=cv2.INTER_AREA)
    image_out[-target_size:,:target_size,:] = face_out
    return image_out