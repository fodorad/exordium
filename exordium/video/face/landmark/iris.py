"""MediaPipe Iris landmark detector wrapper."""

import os
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance

from exordium import WEIGHT_DIR
from exordium.utils.ckpt import download_file
from exordium.utils.device import get_torch_device
from exordium.video.core.io import image_to_np, images_to_np
from exordium.video.face.landmark.constants import FaceMeshLandmarks, IrisLandmarks, TddfaLandmarks


def calculate_iris_diameters(iris_landmarks: np.ndarray) -> np.ndarray:
    """Calculates iris diameters from MediaPipe Iris landmarks.

    Args:
        iris_landmarks (np.ndarray): iris landmarks of shape (5, 2).

    Returns:
        np.ndarray: vector of shape (2,). Horizontal distance and vertical distance.

    """
    return np.array(
        [
            np.linalg.norm(
                iris_landmarks[IrisLandmarks.LEFT.value, :]
                - iris_landmarks[IrisLandmarks.RIGHT.value, :]
            ),
            np.linalg.norm(
                iris_landmarks[IrisLandmarks.TOP.value, :]
                - iris_landmarks[IrisLandmarks.BOTTOM.value, :]
            ),
        ]
    )


def calculate_eyelid_pupil_distances(
    iris_landmarks: np.ndarray, eye_landmarks: np.ndarray
) -> np.ndarray:
    """Calculates eyelid-pupil distances from MediaPipe Iris and FaceMesh landmarks.

    Args:
        iris_landmarks (np.ndarray): Iris landmarks of shape (5, 2).
        eye_landmarks (np.ndarray): FaceMesh landmarks of shape (71, 2) or (16, 2).

    Returns:
        np.ndarray: vector of shape (2,). Top-center distance and bottom-center distance.

    """
    return np.array(
        [
            np.linalg.norm(
                iris_landmarks[IrisLandmarks.CENTER.value, :]
                - eye_landmarks[FaceMeshLandmarks.TOP, :]
            ),
            np.linalg.norm(
                iris_landmarks[IrisLandmarks.CENTER.value, :]
                - eye_landmarks[FaceMeshLandmarks.BOTTOM, :]
            ),
        ]
    )


def calculate_eye_aspect_ratio(landmarks: np.ndarray) -> float:
    """Calculates Eye Aspect Ratio feature.

    Usage:
        ear_left = eye_aspect_ratio(xy_left)
        ear_right = eye_aspect_ratio(xy_right)
        ear_mean = (ear_left + ear_right) / 2.0

    Args:
        landmarks (np.ndarray): eye landmarks of shape (N, 2).
            N == 6 if the landmark detector is the 3DDFA_V2.
            N == 16 if the landmark detector is the FaceMesh.

    Returns:
        float: eye aspect ratio.

    """
    if landmarks.shape not in {(6, 2), (71, 2), (16, 2)}:
        raise ValueError(
            "Invalid eye landmarks. Only 3DDFA_V2 or FaceMesh is supported currently."
            f"Expected (6, 2) or (71, 2) or (16, 2), but got instead {landmarks.shape}"
        )

    if landmarks.shape == (6, 2):  # 3DDFA_V2
        tb1 = distance.euclidean(
            landmarks[TddfaLandmarks.TOP_LEFT.value], landmarks[TddfaLandmarks.BOTTOM_LEFT.value]
        )
        tb2 = distance.euclidean(
            landmarks[TddfaLandmarks.TOP_RIGHT.value], landmarks[TddfaLandmarks.BOTTOM_RIGHT.value]
        )
        lr = distance.euclidean(
            landmarks[TddfaLandmarks.LEFT.value], landmarks[TddfaLandmarks.RIGHT.value]
        )
        return (tb1 + tb2) / (2.0 * lr)
    else:  # FaceMesh
        tb1 = distance.euclidean(
            landmarks[FaceMeshLandmarks.TOP_LEFT].squeeze(axis=0),
            landmarks[FaceMeshLandmarks.BOTTOM_LEFT].squeeze(axis=0),
        )
        tb2 = distance.euclidean(
            landmarks[FaceMeshLandmarks.TOP].squeeze(axis=0),
            landmarks[FaceMeshLandmarks.BOTTOM].squeeze(axis=0),
        )
        tb3 = distance.euclidean(
            landmarks[FaceMeshLandmarks.TOP_RIGHT].squeeze(axis=0),
            landmarks[FaceMeshLandmarks.BOTTOM_RIGHT].squeeze(axis=0),
        )
        lr = distance.euclidean(
            landmarks[FaceMeshLandmarks.LEFT].squeeze(axis=0),
            landmarks[FaceMeshLandmarks.RIGHT].squeeze(axis=0),
        )
        return (tb1 + tb2 + tb3) / (3.0 * lr)


class IrisWrapper:
    """MediaPipe Iris wrapper class."""

    def __init__(self, device_id: int = 0):
        self.remote_path = (
            "https://github.com/fodorad/exordium/releases/download/v1.0.0/iris_weights.pth"
        )
        self.local_path = WEIGHT_DIR / "iris" / Path(self.remote_path).name
        download_file(self.remote_path, self.local_path)
        self.device = get_torch_device(device_id)
        self.model = MediaPipeIris()
        self.model.load_state_dict(torch.load(self.local_path))
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, eyes: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Predict iris landmarks from eye patches.

        Args:
            eyes: Eye patches as RGB numpy arrays.

        Returns:
            Tuple of (eye_landmarks, iris_landmarks) as numpy arrays.

        """
        sample = images_to_np(eyes, "RGB", (64, 64))  # (B, H, W, C) == (B, 64, 64, 3)
        eye, iris = self.model.predict_on_batch(sample)
        eye = eye.detach().cpu().numpy()[:, :, :2]  # (B, 71, 3) -> (B, 71, 2)
        iris = iris.detach().cpu().numpy()[:, :, :2]  # (B, 5, 3) -> (B, 5, 2)
        return eye.squeeze(), iris.squeeze()  # (71, 2) and (5, 2)

    def eye_to_features(self, eye: str | Path | np.ndarray) -> dict:
        """Calculates features from an eye patch.

        Features as key-value pairs:
            'eye': eye image of shape (H, W, 3) == (64, 64, 3) and RGB channel order.
            'eye_original': eye image of shape (H, W, 3) and RGB channel order.
            'landmarks': FaceMesh landmarks of shape (71, 2).
            'iris_landmarks': MediaPipe Iris landmarks of shape (5, 2).
            'iris_diameter': horizontal and vertical distances of the MediaPipe Iris
                landmarks of shape (2,).
            'iris_eyelid_distance': Top and bottom landmarks of MediaPipe Iris to pupil center
                distances of shape (2,).
            'ear': eye aspect ratio of shape ().

        Args:
            eye (os.PathLike | np.ndarray): eye patch image path or np.ndarray of shape (H, W, 3).

        Returns:
            dict: features as a dictionary

        """
        eye_original = image_to_np(eye, "RGB")
        eye = cv2.resize(eye_original, (64, 64), interpolation=cv2.INTER_AREA)

        # (71, 2) eye landmarks xy, (5, 2) iris landmarks xy
        eye_landmarks, iris_landmarks = self([eye])  # (B, H, W, C)

        # (2,) iris diameters hv
        iris_diameters = calculate_iris_diameters(iris_landmarks)

        # (2,) eyelid-pupil distances tb
        eyelid_pupil_distances = calculate_eyelid_pupil_distances(iris_landmarks, eye_landmarks)
        ear = calculate_eye_aspect_ratio(eye_landmarks)

        return {
            "eye_original": eye_original,  # (H, W, 3)
            "eye": eye,  # (64, 64, 3)
            "landmarks": eye_landmarks,  # (71, 2)
            "iris_landmarks": iris_landmarks,  # (5, 2)
            "iris_diameters": iris_diameters,  # (2,)
            "eyelid_pupil_distances": eyelid_pupil_distances,  # (2,)
            "ear": ear,  # ()
        }


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


def visualize_iris(
    image: np.ndarray,
    landmarks: np.ndarray,
    iris_landmarks: np.ndarray,
    output_path: str | os.PathLike | None = None,
    show_indices: bool = False,
) -> np.ndarray:
    """Draw face landmarks and iris landmarks onto an image.

    Face landmarks are rendered in green and iris landmarks in blue.

    Args:
        image: Input image of shape ``(H, W, C)``.
        landmarks: Face landmark coordinates of shape ``(N, 2)``.
        iris_landmarks: Iris landmark coordinates of shape ``(5, 2)``.
        output_path: Path to save the output image. ``None`` skips saving.
        show_indices: Draw landmark indices next to each point.

    Returns:
        Copy of the image with face and iris landmarks drawn on it.

    """
    if not (landmarks.ndim == 2 and landmarks.shape[1] == 2):
        raise Exception(f"Expected landmarks with shape (N, 2) got instead {landmarks.shape}.")

    if not (iris_landmarks.ndim == 2 and iris_landmarks.shape[1] == 2):
        raise Exception(
            f"Expected iris_landmarks with shape (5, 2) got instead {iris_landmarks.shape}."
        )

    image_out = np.copy(image)
    landmarks = np.rint(landmarks).astype(int)
    iris_landmarks = np.rint(iris_landmarks).astype(int)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for index in range(landmarks.shape[0]):
        cv2.circle(image_out, landmarks[index, :], 0, (0, 255, 0), 1)
        if show_indices:
            cv2.putText(
                image_out, str(index), landmarks[index, :], font, 0.3, (0, 0, 0), 1, cv2.LINE_AA
            )

    for index in range(iris_landmarks.shape[0]):
        cv2.circle(image_out, iris_landmarks[index, :], 0, (255, 0, 0), 1)
        if show_indices:
            cv2.putText(
                image_out,
                str(index),
                iris_landmarks[index, :],
                font,
                0.3,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image_out)

    return image_out
