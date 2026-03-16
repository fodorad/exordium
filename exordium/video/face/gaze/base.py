import cv2
import numpy as np
import torch


def rotate_vector(xy: np.ndarray, rotation_degree: float) -> np.ndarray:
    """Rotate a 2D gaze vector by an angle.

    Args:
        xy: Vector represented as XY coordinates.
        rotation_degree: Rotation in degrees.

    Returns:
        Rotated vector.

    """
    if rotation_degree == 0:
        return xy
    theta = np.deg2rad(rotation_degree)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(R, xy)


def vector_to_pitchyaw(vectors):
    """Converts normalised 3D gaze vectors to (pitch, yaw) angles in radians.

    Args:
        vectors (np.ndarray): Gaze vectors of shape (N, 3).

    Returns:
        np.ndarray: Array of shape (N, 2) containing (pitch, yaw) in radians.

    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def gazeto3d(gaze: np.ndarray) -> np.ndarray:
    """Convert the gaze pitch and yaw angles into 3D vector."""
    if not gaze.shape == (2,):
        raise ValueError(
            f"Invalid gaze vector. The values should be pitch and yaw angles. \
                Expected shape is (2,) got instead {gaze.shape}"
        )

    gaze_gt = np.zeros([3])
    gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
    gaze_gt[1] = -np.sin(gaze[1])
    gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
    return gaze_gt


def pitchyaw_to_pixel(pitch: float, yaw: float, length: float = 1.0) -> np.ndarray:
    """Convert the gaze pitch and yaw angles to XY coords.

    Args:
        pitch (float): pitch angle (looking vertically) in degree.
        yaw (float): yaw angle (looking horizontally) in degree.
        length (float, optional): length of the vector. 1.0 means unit length. Defaults to 1.0.

    Returns:
        np.ndarray: XY coords

    """
    dx = -length * np.sin(yaw) * np.cos(pitch)
    dy = -length * np.sin(pitch)
    return np.array([dx, dy])


def spherical2cartesial(x):
    """Converts spherical (pitch, yaw) angles to 3D Cartesian unit vectors.

    Args:
        x (torch.Tensor): Tensor of shape (N, 2) containing (pitch, yaw) in
            radians.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) containing (x, y, z) unit vectors.

    """
    output = torch.zeros(x.size(0), 3)
    output[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
    output[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
    output[:, 1] = torch.sin(x[:, 1])
    return output


def compute_angular_error(input, target):
    """Computes the mean angular error between predicted and target gaze directions.

    Converts both inputs from spherical to Cartesian coordinates and computes
    the mean arc-cosine of their dot products in degrees.

    Args:
        input (torch.Tensor): Predicted gaze angles of shape (N, 2) in radians
            (pitch, yaw).
        target (torch.Tensor): Ground-truth gaze angles of shape (N, 2) in
            radians (pitch, yaw).

    Returns:
        torch.Tensor: Scalar mean angular error in degrees.

    """
    input = spherical2cartesial(input)
    target = spherical2cartesial(target)
    input = input.view(-1, 3, 1)
    target = target.view(-1, 1, 3)
    output_dot = torch.bmm(target, input)
    output_dot = output_dot.view(-1)
    output_dot = torch.acos(output_dot)
    output_dot = output_dot.data
    output_dot = 180 * torch.mean(output_dot) / np.pi
    return output_dot


def softmax_temperature(tensor, temperature):
    """Applies temperature-scaled softmax along dimension 1.

    Args:
        tensor (torch.Tensor): Input logits of shape (N, C).
        temperature (float): Temperature scaling factor. Values less than 1
            sharpen the distribution; values greater than 1 flatten it.

    Returns:
        torch.Tensor: Temperature-scaled softmax probabilities of shape (N, C).

    """
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result


def looking_at_camera_yaw_pitch(yaw, pitch, thr: float = 0.5) -> bool:
    """Determines whether a gaze direction points toward the camera.

    Converts (yaw, pitch) angles to a unit XY gaze vector and checks whether
    its magnitude is within the threshold.

    Args:
        yaw (float): Yaw angle in radians.
        pitch (float): Pitch angle in radians.
        thr (float, optional): Distance threshold in normalised gaze space.
            Defaults to 0.5.

    Returns:
        bool: True if the gaze is directed at the camera.

    """
    xy = pitchyaw_to_pixel(pitch, yaw, length=1)
    return looking_at_camera_xy(xy, thr)


def looking_at_camera_xy(xy: np.ndarray, thr: float = 0.5) -> bool:
    """Determines whether a 2D gaze vector points toward the camera.

    Args:
        xy (np.ndarray): 2D gaze vector of shape (2,).
        thr (float, optional): Magnitude threshold below which gaze is
            considered on-camera. Defaults to 0.5.

    Returns:
        bool: True if the L2 norm of xy is less than thr.

    """
    return bool(np.linalg.norm(xy) < thr)


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
    end_point = tuple(np.round([origin[0] + end_point[0], origin[1] + end_point[1]]).astype(int))
    cv2.arrowedLine(
        image_out,
        tuple(np.round(origin).astype(np.int32)),
        end_point,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.18,
    )
    return image_out


def convert_rotate_draw_vector(
    image: np.ndarray,
    yaw: float,
    pitch: float,
    rotation_degree: float,
    origin: np.ndarray,
    length: float = 400,
) -> np.ndarray:
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


def convert_draw_vector(
    image: np.ndarray, yaw: float, pitch: float, origin: np.ndarray, length: float = 200
) -> np.ndarray:
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


def visualize_target_gaze(
    frame: np.ndarray,
    aligned_face: np.ndarray,
    gaze_vector_normed: np.ndarray,
    gaze_vector: np.ndarray,
    gaze_vector_origin: np.ndarray,
    threshold: float = 0.45,
) -> np.ndarray:
    """Draws gaze vectors and threshold circles on a frame and an inset face crop.

    Renders the gaze vector on the full frame at the face location and also
    embeds a normalised view of the gaze in the bottom-left corner of the frame.

    Args:
        frame (np.ndarray): Full video frame of shape (H, W, C) in BGR format.
        aligned_face (np.ndarray): Aligned face crop of shape (h, w, C).
        gaze_vector_normed (np.ndarray): Unit gaze vector of shape (2,) in
            normalised image coordinates.
        gaze_vector (np.ndarray): Gaze vector in original image coordinates of
            shape (2,).
        gaze_vector_origin (np.ndarray): Pixel coordinates (x, y) of the gaze
            origin on the full frame.
        threshold (float, optional): Radius threshold for the on-camera circle
            as a fraction of the face radius. Defaults to 0.45.

    Returns:
        np.ndarray: Annotated frame of the same shape as the input frame.

    """
    frame_out = np.copy(frame)
    target_size = frame_out.shape[0] // 3
    face_ratio = max(aligned_face.shape[:2]) / target_size
    # prepare normed
    target_face = cv2.resize(aligned_face, (448, 448), interpolation=cv2.INTER_AREA)
    target_face_radius = target_face.shape[0] // 2  # max gaze vector length
    target_face_origin = tuple(np.array(target_face.shape[:2]) // 2)
    # draw original
    frame_out = cv2.circle(
        frame_out, gaze_vector_origin, int(target_face_radius * face_ratio), (0, 255, 0), 2
    )
    frame_out = cv2.circle(
        frame_out,
        gaze_vector_origin,
        int(target_face_radius * face_ratio * threshold),
        (255, 0, 0),
        2,
    )
    frame_out = draw_vector(
        frame_out, gaze_vector_origin, gaze_vector * face_ratio * target_face_radius
    )
    # draw normed
    target_face = cv2.circle(
        target_face, target_face_origin, int(target_face_radius), (0, 255, 0), 2
    )
    target_face = cv2.circle(
        target_face, target_face_origin, int(target_face_radius * threshold), (255, 0, 0), 2
    )
    target_face = cv2.circle(target_face, target_face_origin, 2, (0, 255, 0), 2)
    target_face = draw_vector(
        target_face, target_face_origin, gaze_vector_normed * target_face_radius
    )
    target_face = cv2.resize(target_face, (target_size, target_size), interpolation=cv2.INTER_AREA)
    frame_out[-target_size:, :target_size, :] = target_face
    return frame_out


def visualize_normed_space(
    image: np.ndarray, aligned_face: np.ndarray, yaw: float, pitch: float
) -> np.ndarray:
    """Draws a normalised gaze visualisation inset into the bottom-left corner.

    Resizes the aligned face to 448x448, draws concentric threshold circles and
    a gaze arrow, then composites the result into the image.

    Args:
        image (np.ndarray): Full frame of shape (H, W, C) in BGR format.
        aligned_face (np.ndarray): Aligned face crop of shape (h, w, C).
        yaw (float): Yaw angle in radians.
        pitch (float): Pitch angle in radians.

    Returns:
        np.ndarray: Copy of the input image with the gaze inset drawn in the
            bottom-left corner.

    """
    image_out = np.copy(image)
    H, _ = image.shape[:2]
    target_size = H // 3
    length = 200
    face_out = cv2.resize(aligned_face, (448, 448), interpolation=cv2.INTER_AREA)
    center = tuple(np.array(face_out.shape[:2]) // 2)
    face_out = cv2.circle(face_out, center, length, (0, 255, 0), 2)
    face_out = cv2.circle(face_out, center, length // 2, (255, 0, 0), 2)
    face_out = cv2.circle(face_out, center, 2, (0, 255, 0), 2)
    face_out = convert_draw_vector(face_out, yaw, pitch, center, length)
    face_out = cv2.resize(face_out, (target_size, target_size), interpolation=cv2.INTER_AREA)
    image_out[-target_size:, :target_size, :] = face_out
    return image_out


# ---------------------------------------------------------------------------
# GazeWrapper abstract base class
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod


class GazeWrapper(ABC):
    """Abstract base class for gaze estimation wrappers.

    Provides shared :meth:`looking_at_camera` and :meth:`visualize` static
    methods so that :class:`~exordium.video.unigaze.UnigazeWrapper` and
    :class:`~exordium.video.l2csnet.L2csNetWrapper` are interchangeable.

    Subclasses must implement :meth:`__call__` (preprocessed tensor →
    (yaw, pitch) tensors) and :meth:`predict_pipeline` (image sequence →
    (yaw, pitch) numpy arrays).

    """

    @abstractmethod
    def __call__(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run gaze estimation on a preprocessed batch tensor.

        Args:
            samples: Preprocessed face tensor on the model device.

        Returns:
            Tuple of ``(yaw, pitch)`` tensors each of shape ``(B,)`` in
            radians.

        """

    @abstractmethod
    def predict_pipeline(
        self,
        faces: "Sequence[str | Path | np.ndarray]",
        roll_angles: "Sequence[float] | None" = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict gaze from face images with optional roll correction.

        Args:
            faces: Face images (file paths or RGB numpy arrays).
            roll_angles: Per-face roll angles in degrees.  If provided each
                face is rotated by ``-roll_angle`` before inference.
                ``None`` skips rotation correction.

        Returns:
            Tuple of ``(yaw, pitch)`` numpy arrays each of shape ``(B,)``
            in radians.

        """

    @staticmethod
    def looking_at_camera(yaw: np.ndarray, pitch: np.ndarray, thr: float = 0.3) -> np.ndarray:
        """Determine whether each face is looking at the camera.

        Computes the L2 magnitude of the ``(yaw, pitch)`` vector and checks
        whether it is below ``thr``.

        Args:
            yaw: Yaw angles in radians, shape ``(B,)``.
            pitch: Pitch angles in radians, shape ``(B,)``.
            thr: Angle magnitude threshold in radians.  Smaller → stricter.

        Returns:
            Boolean array of shape ``(B,)``.

        """
        return np.sqrt(yaw**2 + pitch**2) < thr

    @staticmethod
    def visualize(
        face_crops: "Sequence[np.ndarray]",
        yaw: np.ndarray,
        pitch: np.ndarray,
        roll_angles: "Sequence[float] | None" = None,
        thr: float = 0.3,
    ) -> "list[np.ndarray]":
        """Draw gaze vectors on face crops with threshold circles.

        For each face draws:

        * A blue outer circle for the maximum gaze magnitude.
        * A green inner circle for the looking-at-camera threshold.
        * A red arrow for the gaze direction (rotated back when
          ``roll_angles`` are supplied).

        Args:
            face_crops: RGB face images each of shape ``(H, W, 3)``.
            yaw: Yaw angles in radians, shape ``(B,)``.
            pitch: Pitch angles in radians, shape ``(B,)``.
            roll_angles: Per-face roll angles in degrees used to rotate the
                arrow back into the original orientation.  ``None`` means no
                rotation (arrow is in the upright-head frame).
            thr: Threshold fraction for the inner circle radius.

        Returns:
            List of annotated RGB images.

        """
        if roll_angles is None:
            roll_angles = [0.0] * len(face_crops)

        results = []
        for face, y, p, roll in zip(face_crops, yaw, pitch, roll_angles):
            h, w = face.shape[:2]
            center = np.array([w / 2, h / 2])
            center_int = tuple(np.round(center).astype(int))
            length = min(h, w) / 2

            image_out = np.copy(face)
            cv2.circle(image_out, center_int, int(length), (0, 0, 255), 2)
            cv2.circle(image_out, center_int, int(length * thr), (0, 255, 0), 2)
            cv2.circle(image_out, center_int, 2, (0, 255, 0), 2)
            image_out = convert_rotate_draw_vector(image_out, y, p, roll, center, length)
            results.append(image_out)
        return results
