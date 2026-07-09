"""MediaPipe Iris landmark detector wrapper."""

import math
import os
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from exordium import WEIGHT_DIR
from exordium.utils.ckpt import download_weight
from exordium.utils.device import get_torch_device
from exordium.video.face.landmark.constants import FaceMeshLandmarks, IrisLandmarks


def _norm2d(a, b) -> float:
    """2-D Euclidean distance for numpy arrays or plain scalars — returns ``float``."""
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.sqrt(dx * dx + dy * dy)


def _norm2d_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """2-D Euclidean distance for torch tensors — returns scalar ``torch.Tensor``."""
    return torch.linalg.norm(a.float() - b.float())


def calculate_iris_diameters(
    iris_landmarks: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Calculate iris diameters from MediaPipe Iris landmarks.

    Computes the horizontal (left–right) and vertical (top–bottom) diameter
    of the iris from the 5-point iris landmark set.

    Type-preserving: ``np.ndarray`` input returns ``np.ndarray``; ``torch.Tensor``
    input returns ``torch.Tensor``.

    Args:
        iris_landmarks: Iris landmarks of shape ``(5, 2)``.

    Returns:
        1-D array / tensor of shape ``(2,)`` containing
        ``[horizontal_diameter, vertical_diameter]``.

    """
    if isinstance(iris_landmarks, torch.Tensor):
        il = iris_landmarks.float()
        return torch.stack(
            [
                _norm2d_t(il[IrisLandmarks.LEFT.value], il[IrisLandmarks.RIGHT.value]),
                _norm2d_t(il[IrisLandmarks.TOP.value], il[IrisLandmarks.BOTTOM.value]),
            ]
        )
    return np.array(
        [
            _norm2d(
                iris_landmarks[IrisLandmarks.LEFT.value], iris_landmarks[IrisLandmarks.RIGHT.value]
            ),
            _norm2d(
                iris_landmarks[IrisLandmarks.TOP.value], iris_landmarks[IrisLandmarks.BOTTOM.value]
            ),
        ]
    )


def calculate_eyelid_pupil_distances(
    iris_landmarks: np.ndarray | torch.Tensor,
    eye_landmarks: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Calculate eyelid-to-pupil distances from iris and eye landmarks.

    Measures the Euclidean distance from the iris centre to the top and
    bottom eyelid landmarks, giving an estimate of how open the eye is.

    Type-preserving: ``np.ndarray`` inputs return ``np.ndarray``;
    ``torch.Tensor`` inputs return ``torch.Tensor``.

    Args:
        iris_landmarks: Iris landmarks of shape ``(5, 2)``.
        eye_landmarks: FaceMesh eye landmarks of shape ``(71, 2)`` or ``(16, 2)``.

    Returns:
        1-D array / tensor of shape ``(2,)`` containing
        ``[top_eyelid_distance, bottom_eyelid_distance]``.

    """
    if isinstance(iris_landmarks, torch.Tensor) and isinstance(eye_landmarks, torch.Tensor):
        center = iris_landmarks[IrisLandmarks.CENTER.value].float()
        top = eye_landmarks[FaceMeshLandmarks.TOP].squeeze().float()
        bot = eye_landmarks[FaceMeshLandmarks.BOTTOM].squeeze().float()
        return torch.stack([_norm2d_t(center, top), _norm2d_t(center, bot)])
    center = iris_landmarks[IrisLandmarks.CENTER.value]
    return np.array(
        [
            _norm2d(center, eye_landmarks[FaceMeshLandmarks.TOP].squeeze()),
            _norm2d(center, eye_landmarks[FaceMeshLandmarks.BOTTOM].squeeze()),
        ]
    )


def calculate_eye_aspect_ratio(
    landmarks: np.ndarray | torch.Tensor,
) -> float | torch.Tensor:
    """Calculate the Eye Aspect Ratio (EAR).

    EAR is defined as the mean of three vertical landmark distances divided
    by the horizontal landmark distance.  A value close to zero indicates
    a closed eye; typical open-eye values are in the range 0.25–0.40.

    Usage::

        ear_right = calculate_eye_aspect_ratio(right_eye_lmks)
        ear_left  = calculate_eye_aspect_ratio(left_eye_lmks)
        ear_mean  = (ear_right + ear_left) / 2.0

    Type-preserving: ``np.ndarray`` input returns ``float``; ``torch.Tensor``
    input returns a scalar ``torch.Tensor``.

    Args:
        landmarks: Eye landmarks of shape ``(71, 2)`` or ``(16, 2)``.
            Both shapes correspond to MediaPipe FaceMesh eye subsets.

    Returns:
        Eye aspect ratio — ``float`` for numpy input, scalar
        ``torch.Tensor`` for tensor input.

    Raises:
        ValueError: If ``landmarks.shape`` is not ``(71, 2)`` or ``(16, 2)``.

    """
    if tuple(landmarks.shape) not in {(71, 2), (16, 2)}:
        raise ValueError(
            "Invalid eye landmarks. Only FaceMesh is supported. "
            f"Expected (71, 2) or (16, 2), got {landmarks.shape}"
        )

    if isinstance(landmarks, torch.Tensor):
        lm = landmarks.float()
        tb1 = _norm2d_t(
            lm[FaceMeshLandmarks.TOP_LEFT].squeeze(), lm[FaceMeshLandmarks.BOTTOM_LEFT].squeeze()
        )
        tb2 = _norm2d_t(lm[FaceMeshLandmarks.TOP].squeeze(), lm[FaceMeshLandmarks.BOTTOM].squeeze())
        tb3 = _norm2d_t(
            lm[FaceMeshLandmarks.TOP_RIGHT].squeeze(), lm[FaceMeshLandmarks.BOTTOM_RIGHT].squeeze()
        )
        lr = _norm2d_t(lm[FaceMeshLandmarks.LEFT].squeeze(), lm[FaceMeshLandmarks.RIGHT].squeeze())
        return (tb1 + tb2 + tb3) / (3.0 * lr)

    tb1 = _norm2d(
        landmarks[FaceMeshLandmarks.TOP_LEFT].squeeze(),
        landmarks[FaceMeshLandmarks.BOTTOM_LEFT].squeeze(),
    )
    tb2 = _norm2d(
        landmarks[FaceMeshLandmarks.TOP].squeeze(),
        landmarks[FaceMeshLandmarks.BOTTOM].squeeze(),
    )
    tb3 = _norm2d(
        landmarks[FaceMeshLandmarks.TOP_RIGHT].squeeze(),
        landmarks[FaceMeshLandmarks.BOTTOM_RIGHT].squeeze(),
    )
    lr = _norm2d(
        landmarks[FaceMeshLandmarks.LEFT].squeeze(),
        landmarks[FaceMeshLandmarks.RIGHT].squeeze(),
    )
    return (tb1 + tb2 + tb3) / (3.0 * lr)


class IrisWrapper:
    """MediaPipe Iris landmark detector wrapper.

    Detects 71 eye landmarks and 5 iris landmarks from 64×64 eye patches
    using a PyTorch port of the MediaPipe Iris model.

    Supported input types for :meth:`__call__`:

    * ``torch.Tensor`` — ``(3, H, W)`` or ``(B, 3, H, W)`` uint8 RGB;
      fastest path, stays on ``self.device`` end-to-end.
    * ``np.ndarray`` — ``(H, W, 3)`` or ``(B, H, W, 3)`` uint8 RGB.
    * ``Sequence[np.ndarray]`` — list of ``(H, W, 3)`` uint8 arrays.
    * ``Sequence[str | Path]`` — list of image file paths.

    Design contract:

    * :meth:`preprocess` — convert any input to a ``(B, 3, 64, 64)``
      float32 tensor in ``[0, 1]`` on ``self.device``.
    * :meth:`inference` — model forward pass; returns
      ``(eye_landmarks, iris_landmarks)`` as ``(B, 71, 2)`` and
      ``(B, 5, 2)`` tensors on ``self.device``.
    * :meth:`__call__` — chains both under ``torch.inference_mode``.

    Args:
        device_id: GPU device index.  ``None`` or ``-1`` uses CPU.

    """

    def __init__(self, device_id: int | None = None):
        self.local_path = download_weight("iris_weights.pth", WEIGHT_DIR / "iris")
        self.device = get_torch_device(device_id)
        self.model = MediaPipeIris()
        self.model.load_state_dict(
            torch.load(self.local_path, map_location=torch.device("cpu"), weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _to_uint8_tensor(frames) -> torch.Tensor:
        """Convert any supported input to a uint8 ``(B, 3, H, W)`` CPU tensor.

        Delegates to :func:`~exordium.video.core.io.to_uint8_tensor`.

        Args:
            frames: One of:

                * ``torch.Tensor (3, H, W)`` or ``(B, 3, H, W)`` uint8 RGB
                * ``np.ndarray (H, W, 3)`` or ``(B, H, W, 3)`` uint8 RGB
                * ``str | Path`` — single image file path
                * ``Sequence[np.ndarray]`` of ``(H, W, 3)`` arrays
                * ``Sequence[str | Path]`` of image file paths

        Returns:
            uint8 tensor of shape ``(B, 3, H, W)`` on CPU.

        """
        from exordium.video.core.io import to_uint8_tensor

        return to_uint8_tensor(frames)

    def preprocess(self, frames) -> torch.Tensor:
        """Resize eye patches to 64×64 and normalise to ``[0, 1]``.

        Accepts variable-size inputs — each patch is resized individually
        before stacking, so eye crops of different sizes are handled correctly.

        Args:
            frames: Any input accepted by :meth:`_to_uint8_tensor`.

        Returns:
            Float32 tensor of shape ``(B, 3, 64, 64)`` on ``self.device``
            with values in ``[0, 1]``.

        """
        if isinstance(frames, (list, tuple)) and not isinstance(frames[0], (str, Path)):
            resized = [
                TF.resize(self._to_uint8_tensor(f).to(self.device), [64, 64], antialias=True)
                for f in frames
            ]
            x = torch.cat(resized, dim=0)
        else:
            x = self._to_uint8_tensor(frames).to(self.device)
            x = TF.resize(x, [64, 64], antialias=True)

        return x.float().div(255.0)

    def inference(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the Iris model and return landmark tensors.

        Args:
            tensor: Float32 tensor of shape ``(B, 3, 64, 64)`` on
                ``self.device`` with values in ``[0, 1]``.

        Returns:
            Tuple of:

            * ``eye_landmarks`` — ``(B, 71, 2)`` float32 tensor containing
              ``(x, y)`` pixel coordinates of the 71 eye landmarks.
            * ``iris_landmarks`` — ``(B, 5, 2)`` float32 tensor containing
              ``(x, y)`` pixel coordinates of the 5 iris landmarks.

            Both tensors are on ``self.device``.

        """
        eye_raw, iris_raw = self.model(tensor)  # (B, 213), (B, 15)
        eye = eye_raw.view(-1, 71, 3)[..., :2]  # (B, 71, 2) — drop z
        iris = iris_raw.view(-1, 5, 3)[..., :2]  # (B, 5,  2)
        return eye, iris

    def __call__(self, frames) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess and run inference on eye patches.

        Args:
            frames: Any supported input (see class docstring).

        Returns:
            Tuple of ``(eye_landmarks, iris_landmarks)`` tensors of shape
            ``(B, 71, 2)`` and ``(B, 5, 2)`` on ``self.device``.

        """
        with torch.inference_mode():
            return self.inference(self.preprocess(frames))

    def eye_to_feature(self, eye: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute iris and eye features from a single eye crop.

        Runs the full pipeline — preprocess → inference → metric computation —
        for a single eye patch and returns all intermediate and derived features
        as torch tensors.  Landmarks are in 64×64 model space.

        Args:
            eye: Eye crop of shape ``(3, H, W)`` uint8 RGB.  Any spatial size
                is accepted; the model resizes to 64×64 internally.

        Returns:
            Dictionary of torch tensors:

            * ``"eye_original"``          — ``(3, H, W)``  uint8  — original input crop
            * ``"eye"``                   — ``(3, 64, 64)`` uint8  — 64×64 resized crop
            * ``"eye_region_landmarks"``  — ``(71, 2)`` float32   — eye landmarks in 64×64 space
            * ``"iris_landmarks"``        — ``(5, 2)``  float32   — iris landmarks in 64×64 space
            * ``"iris_diameters"``        — ``(2,)``    float32   — [horizontal, vertical] diameter
            * ``"eyelid_pupil_distances"`` — ``(2,)`` float32 — [top, bottom] eyelid–pupil dist
            * ``"ear"``                   — scalar      float32   — eye aspect ratio

        """
        preprocessed = self.preprocess(eye)  # (1, 3, 64, 64) float32 [0,1]
        eye_64 = preprocessed[0].mul(255).to(torch.uint8).cpu()  # (3, 64, 64) uint8

        with torch.inference_mode():
            eye_lmks_b, iris_lmks_b = self.inference(preprocessed)  # (1,71,2), (1,5,2)

        eye_region_lmks = eye_lmks_b[0].cpu()  # (71, 2) float32
        iris_lmks = iris_lmks_b[0].cpu()  # (5, 2)  float32

        return {
            "eye_original": eye,  # (3, H, W) uint8
            "eye": eye_64,  # (3, 64, 64) uint8
            "eye_region_landmarks": eye_region_lmks,  # (71, 2) float32
            "iris_landmarks": iris_lmks,  # (5, 2)  float32
            "iris_diameters": cast("torch.Tensor", calculate_iris_diameters(iris_lmks)),
            "eyelid_pupil_distances": cast(
                "torch.Tensor",
                calculate_eyelid_pupil_distances(iris_lmks, eye_region_lmks),
            ),
            "ear": cast("torch.Tensor", calculate_eye_aspect_ratio(eye_region_lmks)),
        }


def visualize_iris(
    image: np.ndarray | torch.Tensor,
    landmarks: np.ndarray | torch.Tensor,
    iris_landmarks: np.ndarray | torch.Tensor,
    output_path: str | os.PathLike | None = None,
    show_indices: bool = False,
) -> np.ndarray | torch.Tensor:
    """Draw face landmarks and iris landmarks onto an image.

    Face landmarks are rendered in green and iris landmarks in blue.
    Accepts ``(H, W, C)`` numpy arrays or ``(C, H, W)`` uint8 torch tensors;
    returns the same type.

    Args:
        image: Input image — ``np.ndarray (H, W, C)`` or
            ``torch.Tensor (C, H, W)`` uint8.
        landmarks: Face landmark coordinates of shape ``(N, 2)`` —
            ``np.ndarray`` or ``torch.Tensor``.
        iris_landmarks: Iris landmark coordinates of shape ``(5, 2)`` —
            ``np.ndarray`` or ``torch.Tensor``.
        output_path: Path to save the output image. ``None`` skips saving.
        show_indices: Draw landmark indices next to each point.

    Returns:
        Copy of the image with face and iris landmarks drawn, same type as input.

    """
    if not (landmarks.ndim == 2 and landmarks.shape[1] == 2):
        raise Exception(f"Expected landmarks with shape (N, 2) got instead {landmarks.shape}.")

    if not (iris_landmarks.ndim == 2 and iris_landmarks.shape[1] == 2):
        raise Exception(
            f"Expected iris_landmarks with shape (5, 2) got instead {iris_landmarks.shape}."
        )

    if isinstance(image, torch.Tensor):
        img_np: np.ndarray = image.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = cast("np.ndarray", image)

    image_out = img_np.copy()
    # Convert landmarks to numpy at cv2 boundary
    lmks = np.rint(
        landmarks.cpu().numpy() if isinstance(landmarks, torch.Tensor) else landmarks
    ).astype(int)
    iris_lmks = np.rint(
        iris_landmarks.cpu().numpy() if isinstance(iris_landmarks, torch.Tensor) else iris_landmarks
    ).astype(int)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for index in range(lmks.shape[0]):
        cv2.circle(image_out, tuple(lmks[index, :]), 0, (0, 255, 0), 1)
        if show_indices:
            cv2.putText(
                image_out, str(index), tuple(lmks[index, :]), font, 0.3, (0, 0, 0), 1, cv2.LINE_AA
            )

    for index in range(iris_lmks.shape[0]):
        cv2.circle(image_out, tuple(iris_lmks[index, :].astype(int)), 0, (255, 0, 0), 1)
        if show_indices:
            cv2.putText(
                image_out,
                str(index),
                tuple(iris_lmks[index, :].astype(int)),
                font,
                0.3,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image_out)

    if isinstance(image, torch.Tensor):
        return torch.from_numpy(image_out).permute(2, 0, 1)
    return image_out


#################################################################################
#                                                                               #
#   Code: https://github.com/cedriclmenard/irislandmarks.pytorch                #
#   Author: Cedric Menard                                                       #
#   Paper: Real-time Pupil Tracking from Monocular Video for Digital Puppetry   #
#   Authors: Artsiom Ablavatski, Andrey Vakunov, Ivan Grishchenko,              #
#            Karthik Raveendran, Matsvei Zhdanovich                             #
#                                                                               #
#################################################################################


class IrisBlock(nn.Module):  # pragma: no cover
    """Main building block for Iris architecture."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.channel_pad = out_channels - in_channels

        padding = (kernel_size - 1) // 2
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

        self.convAct = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=int(out_channels / 2),
                kernel_size=stride,
                stride=stride,
                padding=0,
                bias=True,
            ),
            nn.PReLU(int(out_channels / 2)),
        )
        self.dwConvConv = nn.Sequential(
            nn.Conv2d(
                in_channels=int(out_channels / 2),
                out_channels=int(out_channels / 2),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,  # Padding might be wrong here
                groups=int(out_channels / 2),
                bias=True,
            ),
            nn.Conv2d(
                in_channels=int(out_channels / 2),
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        """Forward pass through Iris block.

        Args:
            x: Input tensor.

        Returns:
            Tensor after convolution, pooling, and activation.

        """
        h = self.convAct(x)
        if self.stride == 2:
            x = self.max_pool(x)

        h = self.dwConvConv(h)

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(h + x)


class MediaPipeIris(nn.Module):  # pragma: no cover
    """The IrisLandmark face landmark model from MediaPipe.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.
    The conversion to PyTorch is fairly straightforward, but there are
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.
    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.

    """

    def __init__(self):
        super().__init__()

        # self.num_coords = 228
        # self.x_scale = 64.0
        # self.y_scale = 64.0
        self.min_score_thresh = 0.75

        self._define_layers()

    def _define_layers(self):
        """Define neural network layers.

        Creates backbone, eye landmarks, and iris landmarks detection branches.

        """
        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0, bias=True
            ),
            nn.PReLU(64),
            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 64),
            IrisBlock(64, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
        )
        self.split_eye = nn.Sequential(
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            nn.Conv2d(
                in_channels=128, out_channels=213, kernel_size=2, stride=1, padding=0, bias=True
            ),
        )
        self.split_iris = nn.Sequential(
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            IrisBlock(128, 128, stride=2),
            IrisBlock(128, 128),
            IrisBlock(128, 128),
            nn.Conv2d(
                in_channels=128, out_channels=15, kernel_size=2, stride=1, padding=0, bias=True
            ),
        )

    def forward(self, x):
        """Forward pass through MediaPipe Iris model.

        Args:
            x: Input tensor of shape (b, c, h, w).

        Returns:
            List of [eye_landmarks, iris_landmarks] predictions.

        """
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, [0, 1, 0, 1], "constant", 0)
        b = x.shape[0]  # batch size, needed for reshaping later

        x = self.backbone(x)  # (b, 128, 8, 8)

        e = self.split_eye(x)  # (b, 213, 1, 1)
        e = e.view(b, -1)  # (b, 213)

        i = self.split_iris(x)  # (b, 15, 1, 1)
        i = i.reshape(b, -1)  # (b, 15)
        return [e, i]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.backbone[0].weight.device

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        # return x.float() / 127.5 - 1.0
        return x.float() / 255.0  # NOTE: [0.0, 1.0] range seems to give better results

    def predict_on_image(self, img):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be
                 64 pixels.

        Returns:
            A tensor with face detections.

        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))

        return self.predict_on_batch(img.unsqueeze(0))

    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 64 pixels.

        Returns:
            A list containing a tensor of face detections for each image in
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).
        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score

        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        assert x.shape[1] == 3
        assert x.shape[2] == 64
        assert x.shape[3] == 64

        # 1. Preprocess the images into tensors:
        x = x.to(self._device())
        x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)

        # 3. Postprocess the raw predictions:
        eye, iris = out

        return eye.view(-1, 71, 3), iris.view(-1, 5, 3)
