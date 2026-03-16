"""Head pose estimation wrapper using SixDRepNet via the ``sixdrepnet`` package.

Weights are downloaded automatically on first use by the ``sixdrepnet`` package.

Example::

    wrapper = SixDRepNetWrapper(device_id=0)
    result = wrapper.predict_single(face_rgb)
    yaw, pitch, roll = result["headpose"]  # degrees

"""

import cv2
import numpy as np
from sixdrepnet import SixDRepNet


class SixDRepNetWrapper:
    """6D rotation representation head pose estimator.

    Thin wrapper around the ``sixdrepnet`` package.  Uses a RepVGG-B1g2
    backbone with a 6-dimensional rotation output head trained for unconstrained
    head pose estimation.  Weights are downloaded automatically by the package
    on first use.

    Args:
        device_id: GPU device index.  ``None`` or ``-1`` uses CPU.

    Reference:
        "6D Rotation Representation For Unconstrained Head Pose Estimation"
        Hempel et al., ICASSP 2022.
        https://github.com/thohemp/6DRepNet

    Example::

        wrapper = SixDRepNetWrapper(device_id=0)
        result = wrapper.predict_single(face_rgb)
        yaw, pitch, roll = result["headpose"]  # degrees

    """

    def __init__(self, device_id: int | None = None) -> None:
        _device_id = -1 if (device_id is None or device_id < 0) else device_id
        self.model = SixDRepNet(gpu_id=_device_id)

    def __call__(self, faces_rgb: list[np.ndarray]) -> np.ndarray:
        """Predict head pose for a batch of face crops.

        Args:
            faces_rgb: List of RGB face crops each of shape ``(H, W, 3)``.

        Returns:
            Array of shape ``(B, 3)`` with ``[yaw, pitch, roll]`` in degrees
            per face.

        """
        results = []
        for face_rgb in faces_rgb:
            face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
            pitch, yaw, roll = self.model.predict(face_bgr)
            results.append([float(yaw), float(pitch), float(roll)])
        return np.array(results, dtype=np.float32)

    def predict_single(self, face_rgb: np.ndarray) -> dict[str, np.ndarray]:
        """Predict head pose for a single face crop.

        Args:
            face_rgb: RGB face crop of shape ``(H, W, 3)``.

        Returns:
            ``dict`` with key ``"headpose"``: ``np.ndarray`` of shape ``(3,)``
            containing ``[yaw, pitch, roll]`` in degrees.

        """
        return {"headpose": self([face_rgb])[0]}


def draw_headpose_axis(
    img: np.ndarray,
    headpose: tuple[float, float, float] | np.ndarray,
    tdx: int | None = None,
    tdy: int | None = None,
    size: int = 100,
) -> np.ndarray:
    """Draw 3D head-pose axes (X/Y/Z) projected onto a 2D image.

    Args:
        img: Input image in BGR format.
        headpose: Head pose angles ``(yaw, pitch, roll)`` in degrees.
        tdx: X-coordinate of the axis origin. ``None`` uses the image centre.
        tdy: Y-coordinate of the axis origin. ``None`` uses the image centre.
        size: Axis length in pixels. Defaults to 100.

    Returns:
        Image with head-pose axes drawn (red=X, green=Y, blue=Z).

    """
    yaw, pitch, roll = headpose

    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(-roll)

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]]
    )
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])

    R = Rz @ Ry @ Rx

    if tdx is None or tdy is None:
        height, width = img.shape[:2]
        tdx = width // 2
        tdy = height // 2

    axis = np.float32([[size, 0, 0], [0, -size, 0], [0, 0, -size]])
    axis_rotated = R.dot(axis.T).T

    points = [
        (int(tdx + axis_rotated[i, 0]), int(tdy + axis_rotated[i, 1]))
        for i in range(axis_rotated.shape[0])
    ]

    img = cv2.line(img, (tdx, tdy), points[0], (0, 0, 255), 3)  # X — red
    img = cv2.line(img, (tdx, tdy), points[1], (0, 255, 0), 3)  # Y — green
    img = cv2.line(img, (tdx, tdy), points[2], (255, 0, 0), 2)  # Z — blue

    return img
