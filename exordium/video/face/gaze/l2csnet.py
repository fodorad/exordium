"""L2CS-Net gaze estimation model wrapper."""

from collections.abc import Sequence
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from exordium import WEIGHT_DIR
from exordium.utils.ckpt import download_file
from exordium.utils.device import get_torch_device
from exordium.video.core.io import images_to_np
from exordium.video.face.gaze.base import GazeWrapper
from exordium.video.face.transform import rotate_face


class L2csNetWrapper(GazeWrapper):
    """L2CS-Net gaze estimation wrapper.

    Predicts gaze direction (pitch and yaw) from face crops.

    Args:
        device_id: Device index. ``None`` or negative for CPU.

    """

    def __init__(self, device_id: int | None = None):
        self.device = get_torch_device(device_id)
        self.remote_path = (
            "https://github.com/fodorad/exordium/releases/download/v1.0.0/l2csnet_weights.pkl"
        )
        self.local_path = WEIGHT_DIR / "l2csnet" / Path(self.remote_path).name
        download_file(self.remote_path, self.local_path)
        saved_state_dict = torch.load(self.local_path, map_location=self.device)
        del saved_state_dict["fc_finetune.weight"]
        del saved_state_dict["fc_finetune.bias"]

        self.model = L2CS_Builder(arch="ResNet50", bins=90)
        self.model.load_state_dict(saved_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = torch.FloatTensor([idx for idx in range(90)]).to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(448),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform_inference = transforms.Compose(
            [
                transforms.Resize(448),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def __call__(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict gaze angles from a preprocessed face tensor.

        Args:
            samples: Preprocessed face tensor of shape ``(B, 3, 448, 448)``
                on the model device.

        Returns:
            Tuple of ``(yaw, pitch)`` tensors each of shape ``(B,)`` in
            radians.

        """
        gaze_yaw, gaze_pitch = self.model(samples)
        yaw_predicted = self.softmax(gaze_yaw)
        pitch_predicted = self.softmax(gaze_pitch)

        pitch_predicted = torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180

        pitch_normed = pitch_predicted * np.pi / 180.0
        yaw_normed = yaw_predicted * np.pi / 180.0
        return yaw_normed, pitch_normed

    def predict_pipeline(
        self,
        faces: Sequence[str | Path | np.ndarray],
        roll_angles: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict gaze from face images with optional head-roll correction.

        Args:
            faces: Face images (paths or RGB numpy arrays).
            roll_angles: Per-face roll angles in degrees.  If provided each
                face is rotated to align upright before inference.

        Returns:
            Tuple of ``(yaw, pitch)`` numpy arrays each of shape ``(B,)``
            in radians.

        """
        faces_rgb = images_to_np(faces, "RGB", resize=None)

        if roll_angles is not None:
            faces_rgb = np.stack(
                [rotate_face(face, roll)[0] for face, roll in zip(faces_rgb, roll_angles)]
            )

        samples = (torch.from_numpy(faces_rgb).permute(0, 3, 1, 2).float() / 255.0).to(self.device)
        samples = F.interpolate(samples, size=(448, 448), mode="bilinear", align_corners=False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        samples = (samples - mean) / std

        yaw_normed, pitch_normed = self(samples)
        return yaw_normed.detach().cpu().numpy(), pitch_normed.detach().cpu().numpy()


####################################################################################
#                                                                                  #
#   Code: https://github.com/Ahmednull/L2CS-Net                                    #
#   Author: Ahmed A. Abdelrahman                                                   #
#   Paper: L2CS-Net : Fine-Grained Gaze Estimation in Unconstrained Environments   #
#   Authors: Ahmed A. Abdelrahman; Thorsten Hempel; Aly Khalifa;                   #
#            Ayoub Al-Hamadi; Laslo Dinges                                         #
#                                                                                  #
####################################################################################


class L2CS(nn.Module):
    """L2CS-Net architecture for gaze estimation."""

    def __init__(self, block, layers, num_bins):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """Build residual layer.

        Args:
            block: Building block class.
            planes: Number of output channels.
            blocks: Number of blocks in the layer.
            stride: Stride for the first block. Defaults to 1.

        Returns:
            Sequential module containing the layer blocks.

        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Predict gaze angles from face image.

        Args:
            x: Input face tensor of shape (b, 3, h, w).

        Returns:
            Tuple of (yaw_logits, pitch_logits) tensors of shape (b, num_bins).

        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw_gaze = self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze


def L2CS_Builder(arch: str = "ResNet50", bins: int = 90):
    """Build L2CS-Net model with specified architecture.

    Args:
        arch: Architecture name. One of "ResNet18", "ResNet34", "ResNet50",
            "ResNet101", "ResNet152". Defaults to "ResNet50".
        bins: Number of gaze angle bins. Defaults to 90.

    Returns:
        L2CS model instance.

    Raises:
        ValueError: If architecture name is invalid.

    """
    match arch:
        case "ResNet18":
            model = L2CS(models.resnet.BasicBlock, [2, 2, 2, 2], bins)
        case "ResNet34":
            model = L2CS(models.resnet.BasicBlock, [3, 4, 6, 3], bins)
        case "ResNet50":
            model = L2CS(models.resnet.Bottleneck, [3, 4, 6, 3], bins)
        case "ResNet101":
            model = L2CS(models.resnet.Bottleneck, [3, 4, 23, 3], bins)
        case "ResNet152":
            model = L2CS(models.resnet.Bottleneck, [3, 8, 36, 3], bins)
        case _:
            raise ValueError("Invalid architecture")
    return model
