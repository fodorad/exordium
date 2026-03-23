"""Gaze estimation base class and shared utility functions."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------


def rotate_vector(
    xy: tuple[float, float] | np.ndarray, rotation_degree: float
) -> tuple[float, float]:
    """Rotate a 2D gaze vector by an angle.

    Args:
        xy: Vector represented as XY coordinates, shape ``(2,)``.
        rotation_degree: Rotation in degrees (counter-clockwise).

    Returns:
        Rotated ``(x, y)`` tuple.

    """
    if rotation_degree == 0:
        return float(xy[0]), float(xy[1])
    theta = math.radians(rotation_degree)
    c, s = math.cos(theta), math.sin(theta)
    return c * float(xy[0]) - s * float(xy[1]), s * float(xy[0]) + c * float(xy[1])


def pitchyaw_to_pixel(pitch: float, yaw: float, length: float = 1.0) -> tuple[float, float]:
    """Convert gaze pitch and yaw angles to a 2D XY direction vector.

    Args:
        pitch: Pitch angle (vertical gaze) in radians.
        yaw: Yaw angle (horizontal gaze) in radians.
        length: Scale factor applied to the unit vector. Defaults to ``1.0``.

    Returns:
        ``(dx, dy)`` direction tuple.

    """
    dx = -length * math.sin(yaw) * math.cos(pitch)
    dy = -length * math.sin(pitch)
    return dx, dy


def vector_to_pitchyaw(vectors: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert normalised 3D gaze vectors to ``(pitch, yaw)`` angles in radians.

    Args:
        vectors: Gaze vectors of shape ``(N, 3)`` — torch tensor or numpy array.

    Returns:
        Tensor of shape ``(N, 2)`` containing ``(pitch, yaw)`` in radians.

    """
    if not isinstance(vectors, torch.Tensor):
        vectors = torch.as_tensor(vectors, dtype=torch.float32)
    norms = vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
    vectors = vectors / norms
    pitch = torch.asin(vectors[:, 1])
    yaw = torch.atan2(vectors[:, 0], vectors[:, 2])
    return torch.stack([pitch, yaw], dim=1)


def gazeto3d(gaze: np.ndarray) -> np.ndarray:
    """Convert ``(pitch, yaw)`` angles to a 3D unit vector.

    Args:
        gaze: Array of shape ``(2,)`` containing ``(pitch, yaw)`` in radians.

    Returns:
        3D unit vector of shape ``(3,)``.

    Raises:
        ValueError: If ``gaze`` does not have shape ``(2,)``.

    """
    if gaze.shape != (2,):
        raise ValueError(f"Expected shape (2,), got {gaze.shape}")
    pitch, yaw = float(gaze[0]), float(gaze[1])
    return np.array(
        [
            -math.cos(yaw) * math.sin(pitch),
            -math.sin(yaw),
            -math.cos(yaw) * math.cos(pitch),
        ]
    )


def spherical2cartesial(x: torch.Tensor) -> torch.Tensor:
    """Convert spherical ``(pitch, yaw)`` angles to 3D Cartesian unit vectors.

    Args:
        x: Tensor of shape ``(N, 2)`` containing ``(pitch, yaw)`` in radians.

    Returns:
        Tensor of shape ``(N, 3)`` containing ``(x, y, z)`` unit vectors.

    """
    output = torch.zeros(x.size(0), 3)
    output[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
    output[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
    output[:, 1] = torch.sin(x[:, 1])
    return output


def compute_angular_error(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute mean angular error between predicted and target gaze directions.

    Args:
        input: Predicted gaze angles of shape ``(N, 2)`` in radians ``(pitch, yaw)``.
        target: Ground-truth gaze angles of shape ``(N, 2)`` in radians ``(pitch, yaw)``.

    Returns:
        Scalar mean angular error in degrees.

    """
    input = spherical2cartesial(input)
    target = spherical2cartesial(target)
    input = input.view(-1, 3, 1)
    target = target.view(-1, 1, 3)
    output_dot = torch.bmm(target, input).view(-1).clamp(-1, 1)
    return 180 * torch.mean(torch.acos(output_dot)) / math.pi


def softmax_temperature(tensor: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature-scaled softmax along dimension 1.

    Args:
        tensor: Input logits of shape ``(N, C)``.
        temperature: Scaling factor. Values < 1 sharpen; values > 1 flatten.

    Returns:
        Temperature-scaled softmax probabilities of shape ``(N, C)``.

    """
    result = torch.exp(tensor / temperature)
    return torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def draw_vector(
    image: np.ndarray | torch.Tensor,
    origin,
    end_point,
) -> np.ndarray | torch.Tensor:
    """Draw a gaze arrow onto an image.

    Accepts ``(H, W, 3)`` uint8 RGB numpy arrays or ``(3, H, W)`` uint8
    RGB torch tensors; returns the same type.

    Args:
        image: RGB image — ``np.ndarray (H, W, 3)`` or
            ``torch.Tensor (3, H, W)`` uint8.
        origin: Arrow origin as XY pixel coordinates ``(x, y)``.
        end_point: Arrow tip offset as XY ``(dx, dy)``.  The actual tip
            is drawn at ``origin + end_point``.

    Returns:
        Copy of the image with the arrow drawn in red, same type as input.

    """
    if isinstance(image, torch.Tensor):
        img: np.ndarray = image.permute(1, 2, 0).cpu().numpy()
    else:
        img = cast("np.ndarray", image)
    img_out = img.copy()
    ox, oy = float(origin[0]), float(origin[1])
    tip = (int(round(ox + float(end_point[0]))), int(round(oy + float(end_point[1]))))
    cv2.arrowedLine(
        img_out,
        (int(round(ox)), int(round(oy))),
        tip,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.18,
    )
    if isinstance(image, torch.Tensor):
        return torch.from_numpy(img_out).permute(2, 0, 1)
    return img_out


def convert_rotate_draw_vector(
    image: np.ndarray | torch.Tensor,
    yaw: float,
    pitch: float,
    rotation_degree: float,
    origin,
    length: float = 400,
) -> np.ndarray | torch.Tensor:
    """Convert gaze angles to XY, rotate, then draw onto an image.

    Accepts ``(H, W, 3)`` uint8 RGB numpy arrays or ``(3, H, W)`` uint8
    RGB torch tensors; returns the same type.

    Args:
        image: RGB image — ``np.ndarray (H, W, 3)`` or
            ``torch.Tensor (3, H, W)`` uint8.
        yaw: Yaw angle in radians.
        pitch: Pitch angle in radians.
        rotation_degree: Degrees to rotate the gaze vector (for roll correction).
        origin: Arrow origin as XY pixel coordinates.
        length: Arrow length in pixels. Defaults to ``400``.

    Returns:
        Copy of the image with the rotated gaze arrow, same type as input.

    """
    xy = pitchyaw_to_pixel(pitch, yaw, length)
    xy_rot = rotate_vector(xy, rotation_degree)
    return draw_vector(image, origin, xy_rot)


def convert_draw_vector(
    image: np.ndarray | torch.Tensor,
    yaw: float,
    pitch: float,
    origin,
    length: float = 200,
) -> np.ndarray | torch.Tensor:
    """Convert gaze angles to XY and draw onto an image.

    Accepts ``(H, W, 3)`` uint8 RGB numpy arrays or ``(3, H, W)`` uint8
    RGB torch tensors; returns the same type.

    Args:
        image: RGB image — ``np.ndarray (H, W, 3)`` or
            ``torch.Tensor (3, H, W)`` uint8.
        yaw: Yaw angle in radians.
        pitch: Pitch angle in radians.
        origin: Arrow origin as XY pixel coordinates.
        length: Arrow length in pixels. Defaults to ``200``.

    Returns:
        Copy of the image with the gaze arrow, same type as input.

    """
    xy = pitchyaw_to_pixel(pitch, yaw, length)
    return draw_vector(image, origin, xy)


# ---------------------------------------------------------------------------
# Camera-gaze utilities
# ---------------------------------------------------------------------------


def looking_at_camera_yaw_pitch(yaw: float, pitch: float, thr: float = 0.5) -> bool:
    """Check whether a gaze direction points toward the camera.

    Args:
        yaw: Yaw angle in radians.
        pitch: Pitch angle in radians.
        thr: Distance threshold in normalised gaze space. Defaults to ``0.5``.

    Returns:
        ``True`` if the gaze is directed at the camera.

    """
    dx, dy = pitchyaw_to_pixel(pitch, yaw, length=1)
    return math.sqrt(dx * dx + dy * dy) < thr


def looking_at_camera_xy(xy: tuple[float, float] | np.ndarray, thr: float = 0.5) -> bool:
    """Check whether a 2D gaze vector points toward the camera.

    Args:
        xy: 2D gaze vector ``(x, y)``.
        thr: Magnitude threshold. Defaults to ``0.5``.

    Returns:
        ``True`` if the L2 norm of ``xy`` is less than ``thr``.

    """
    return math.sqrt(float(xy[0]) ** 2 + float(xy[1]) ** 2) < thr


# ---------------------------------------------------------------------------
# GazeWrapper base class
# ---------------------------------------------------------------------------


class GazeWrapper(ABC):
    """Abstract base class for gaze estimation wrappers.

    Subclasses must implement :meth:`preprocess`, :meth:`inference`, and
    :meth:`postprocess`.  The shared :meth:`__call__`, :meth:`predict`,
    :meth:`looking_at_camera`, and :meth:`visualize` are provided here so
    that :class:`~exordium.video.face.gaze.l2csnet.L2csNetWrapper` and
    :class:`~exordium.video.face.gaze.unigaze.UnigazeWrapper` are
    interchangeable.

    Supported input types for :meth:`__call__` and :meth:`predict`:

    * ``torch.Tensor`` — ``(C, H, W)`` or ``(B, C, H, W)`` uint8 RGB; fastest
      path, no copies until device transfer.
    * ``np.ndarray`` — ``(H, W, 3)`` or ``(B, H, W, 3)`` uint8 RGB.
    * ``Sequence[np.ndarray]`` — list of ``(H, W, 3)`` uint8 arrays.
    * ``Sequence[str | Path]`` — list of image file paths.

    Design contract:

    * :meth:`preprocess` — convert any input to a model-ready float tensor on
      ``self.device``.
    * :meth:`inference` — model forward pass **plus** all output conversion
      (softmax, bin mapping, unit conversion); returns ``(yaw, pitch)`` in
      radians on ``self.device``.
    * :meth:`__call__` — chains the two above under ``torch.inference_mode``;
      returns ``(yaw, pitch)`` tensors.
    * :meth:`predict` — same as :meth:`__call__` but returns numpy arrays;
      accepts an optional ``roll_angles`` list for head-roll correction.

    """

    device: torch.device
    """Torch device used for model inference (CPU or GPU)."""

    @staticmethod
    def _to_uint8_tensor(frames) -> torch.Tensor:
        """Convert any supported input to a uint8 ``(B, 3, H, W)`` CPU tensor.

        Delegates to :func:`~exordium.video.core.io.to_uint8_tensor`.

        Args:
            frames: One of:

                * ``torch.Tensor (C, H, W)`` or ``(B, C, H, W)`` uint8
                * ``np.ndarray (H, W, 3)`` or ``(B, H, W, 3)`` uint8
                * ``str | Path`` — single image file path
                * ``Sequence[np.ndarray]`` of ``(H, W, 3)`` arrays
                * ``Sequence[str | Path]`` of image file paths

        Returns:
            uint8 tensor of shape ``(B, 3, H, W)`` on CPU.

        """
        from exordium.video.core.io import to_uint8_tensor

        return to_uint8_tensor(frames)

    @abstractmethod
    def preprocess(self, frames) -> torch.Tensor:
        """Convert any supported input to a model-ready tensor on ``self.device``.

        Call :meth:`_to_uint8_tensor` first to normalise the input type, then
        apply model-specific resize and normalisation as tensor operations.

        Args:
            frames: Any supported input (see class docstring).

        Returns:
            Preprocessed float tensor on ``self.device``.

        """

    @abstractmethod
    def inference(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model and return ``(yaw, pitch)`` angles in radians.

        Receives a preprocessed float tensor from :meth:`preprocess`, runs
        the model forward pass, converts the raw output to angles, and
        returns the final result.  All conversion logic (softmax, bin
        mapping, dict extraction, unit conversion) belongs here.

        Args:
            tensor: Float tensor of shape ``(B, 3, H, W)`` on ``self.device``,
                already resized and normalised by :meth:`preprocess`.

        Returns:
            Tuple of ``(yaw, pitch)`` tensors each of shape ``(B,)`` in
            radians on ``self.device``.

        """

    def __call__(self, frames) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess and run inference, returning gaze tensors.

        Args:
            frames: Any supported input (see class docstring).

        Returns:
            Tuple of ``(yaw, pitch)`` tensors each of shape ``(B,)`` in
            radians on ``self.device``.

        """
        with torch.inference_mode():
            return self.inference(self.preprocess(frames))

    def predict(
        self,
        frames,
        roll_angles: Sequence[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict gaze angles with optional roll correction, returning tensors.

        When ``roll_angles`` is given each face is rotated by ``-roll`` on
        ``self.device`` before inference so no numpy roundtrip is needed.

        Args:
            frames: Any supported input (see class docstring).
            roll_angles: Per-face roll angles in degrees.  Each face is
                rotated by ``-roll`` to align upright before inference.
                ``None`` skips correction. Defaults to ``None``.

        Returns:
            Tuple of ``(yaw, pitch)`` tensors each of shape ``(B,)``
            in radians on ``self.device``.

        """
        if roll_angles is not None:
            from exordium.video.core.transform import rotate_face

            x = self._to_uint8_tensor(frames).to(self.device)  # (B, 3, H, W) on device
            rotated = [
                cast("torch.Tensor", rotate_face(x[i], -roll)[0])
                for i, roll in enumerate(roll_angles)
            ]
            frames = torch.stack(rotated)  # (B, 3, H, W) on self.device

        return self(frames)

    @staticmethod
    def looking_at_camera(
        yaw: torch.Tensor | np.ndarray,
        pitch: torch.Tensor | np.ndarray,
        thr: float = 0.3,
    ) -> torch.Tensor:
        """Determine whether each face is looking at the camera.

        Args:
            yaw: Yaw angles in radians, shape ``(B,)`` — tensor or numpy array.
            pitch: Pitch angles in radians, shape ``(B,)`` — numpy array or tensor.
            thr: Angle magnitude threshold in radians.  Smaller is stricter.
                Defaults to ``0.3``.

        Returns:
            Boolean tensor of shape ``(B,)``.

        """
        if not isinstance(yaw, torch.Tensor):
            yaw = torch.as_tensor(yaw, dtype=torch.float32)
        if not isinstance(pitch, torch.Tensor):
            pitch = torch.as_tensor(pitch, dtype=torch.float32)
        return (yaw**2 + pitch**2).sqrt() < thr

    @staticmethod
    def visualize(
        faces,
        yaw: np.ndarray | torch.Tensor,
        pitch: np.ndarray | torch.Tensor,
        roll_angles: Sequence[float] | None = None,
        thr: float = 0.3,
        output_path: str | Path | None = None,
    ) -> list[np.ndarray] | list[torch.Tensor]:
        """Draw gaze vectors on face crops.

        Accepts any supported face input (tensor, ndarray, list of images or
        paths).  For each face draws:

        * A red outer circle for the maximum gaze magnitude.
        * A green inner circle for the looking-at-camera threshold.
        * A red arrow for the gaze direction (rotated back into the original
          orientation when ``roll_angles`` are supplied).

        Args:
            faces: Any input accepted by :meth:`_to_uint8_tensor` — uint8
                ``(B, 3, H, W)`` tensor, ``(B, H, W, 3)`` ndarray, or a list
                of images / file paths.
            yaw: Yaw angles in radians, shape ``(B,)`` — tensor or numpy.
            pitch: Pitch angles in radians, shape ``(B,)`` — tensor or numpy.
            roll_angles: Per-face roll angles in degrees.  The gaze arrow is
                rotated back by ``+roll`` to account for the upright
                alignment done before inference.  ``None`` means no rotation.
            thr: Threshold fraction for the inner circle radius. Defaults to
                ``0.3``.
            output_path: Path to save a grid of annotated faces (numpy/cv2
                write).  ``None`` skips saving.

        Returns:
            List of annotated RGB images — ``(3, H, W)`` uint8 tensors when the
            input was a tensor batch, ``(H, W, 3)`` uint8 numpy arrays otherwise.

        """
        input_is_tensor = isinstance(faces, torch.Tensor)
        x = GazeWrapper._to_uint8_tensor(faces)  # (B, 3, H, W) uint8

        yaw_list: list = (
            yaw.detach().cpu().tolist()
            if isinstance(yaw, torch.Tensor)
            else (yaw.tolist() if hasattr(yaw, "tolist") else list(yaw))
        )
        pitch_list: list = (
            pitch.detach().cpu().tolist()
            if isinstance(pitch, torch.Tensor)
            else (pitch.tolist() if hasattr(pitch, "tolist") else list(pitch))
        )
        if roll_angles is None:
            roll_angles = [0.0] * len(x)

        results = []
        for i, (y, p, roll) in enumerate(zip(yaw_list, pitch_list, roll_angles)):
            face_np = x[i].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
            h, w = face_np.shape[:2]
            cx, cy = w / 2, h / 2
            center_int = (int(round(cx)), int(round(cy)))
            length = min(h, w) / 2
            origin = (cx, cy)

            img = face_np.copy()
            cv2.circle(img, center_int, int(length), (0, 0, 255), 2)
            cv2.circle(img, center_int, int(length * thr), (0, 255, 0), 2)
            cv2.circle(img, center_int, 2, (0, 255, 0), 2)
            img = convert_rotate_draw_vector(img, float(y), float(p), float(roll), origin, length)
            results.append(img)

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            grid = np.concatenate(results, axis=1)
            cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

        if input_is_tensor:
            return [torch.from_numpy(r).permute(2, 0, 1) for r in results]
        return results
