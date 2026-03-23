"""L2CS-Net gaze estimation model wrapper."""

from math import sqrt

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import models

from exordium import WEIGHT_DIR
from exordium.utils.ckpt import download_weight
from exordium.utils.device import get_torch_device
from exordium.video.deep.base import _IMAGENET_MEAN, _IMAGENET_STD
from exordium.video.face.gaze.base import GazeWrapper


class L2csNetWrapper(GazeWrapper):
    """L2CS-Net gaze estimation wrapper.

    Predicts gaze direction (pitch and yaw) from face crops using a
    ResNet-50 backbone with two 90-bin classification heads.

    Weights are downloaded automatically from ``fodorad/exordium-weights``
    on Hugging Face Hub on first use.

    Args:
        device_id: Device index.  ``None`` or negative for CPU.

    """

    def __init__(self, device_id: int | None = None):
        self.device = get_torch_device(device_id)
        self.local_path = download_weight("l2csnet_weights.pkl", WEIGHT_DIR / "l2csnet")
        saved_state_dict = torch.load(self.local_path, map_location=self.device, weights_only=True)
        del saved_state_dict["fc_finetune.weight"]
        del saved_state_dict["fc_finetune.bias"]

        self.model = L2CS_Builder(arch="ResNet50", bins=90)
        self.model.load_state_dict(saved_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._softmax = nn.Softmax(dim=1)
        self._idx_tensor = torch.arange(90, dtype=torch.float32, device=self.device)
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def preprocess(self, frames) -> torch.Tensor:
        """Resize and normalise face crops to L2CS-Net input convention.

        L2CS-Net expects 448×448 inputs normalised to ImageNet mean/std.

        Args:
            frames: Any input accepted by :meth:`~GazeWrapper._to_uint8_tensor`.

        Returns:
            Float tensor of shape ``(B, 3, 448, 448)`` on ``self.device``.

        """
        x = self._to_uint8_tensor(frames).to(self.device)
        x = TF.resize(x, [448, 448], antialias=True)
        x = x.float().div(255)
        return (x - self._mean) / self._std

    def inference(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run L2CS-Net and return ``(yaw, pitch)`` angles in radians.

        Performs the full forward pass and converts bin logits to angles:
        softmax over 90 bins → weighted expected bin index → degrees → radians.

        Args:
            tensor: Float tensor of shape ``(B, 3, 448, 448)`` on
                ``self.device``, normalised to ImageNet mean/std.

        Returns:
            Tuple of ``(yaw, pitch)`` tensors each of shape ``(B,)``
            in radians on ``self.device``.

        """
        gaze_yaw, gaze_pitch = self.model(tensor)
        yaw_prob = self._softmax(gaze_yaw)
        pitch_prob = self._softmax(gaze_pitch)
        yaw_deg = torch.sum(yaw_prob * self._idx_tensor, dim=1) * 4 - 180
        pitch_deg = torch.sum(pitch_prob * self._idx_tensor, dim=1) * 4 - 180
        return yaw_deg * (torch.pi / 180.0), pitch_deg * (torch.pi / 180.0)


####################################################################################
#                                                                                  #
#   Code: https://github.com/Ahmednull/L2CS-Net                                    #
#   Author: Ahmed A. Abdelrahman                                                   #
#   Paper: L2CS-Net : Fine-Grained Gaze Estimation in Unconstrained Environments   #
#   Authors: Ahmed A. Abdelrahman; Thorsten Hempel; Aly Khalifa;                   #
#            Ayoub Al-Hamadi; Laslo Dinges                                         #
#                                                                                  #
####################################################################################


class L2CS(nn.Module):  # pragma: no cover
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
        """Predict gaze bin logits from a face image batch.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Tuple of ``(yaw_logits, pitch_logits)`` each of shape
            ``(B, num_bins)``.

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
        return self.fc_yaw_gaze(x), self.fc_pitch_gaze(x)


def L2CS_Builder(arch: str = "ResNet50", bins: int = 90):  # pragma: no cover
    """Build an L2CS-Net model with the specified ResNet backbone.

    Args:
        arch: One of ``"ResNet18"``, ``"ResNet34"``, ``"ResNet50"``
            (default), ``"ResNet101"``, ``"ResNet152"``.
        bins: Number of gaze angle bins. Defaults to ``90``.

    Returns:
        :class:`L2CS` model instance.

    Raises:
        ValueError: If ``arch`` is not a recognised architecture name.

    """
    match arch:
        case "ResNet18":
            return L2CS(models.resnet.BasicBlock, [2, 2, 2, 2], bins)
        case "ResNet34":
            return L2CS(models.resnet.BasicBlock, [3, 4, 6, 3], bins)
        case "ResNet50":
            return L2CS(models.resnet.Bottleneck, [3, 4, 6, 3], bins)
        case "ResNet101":
            return L2CS(models.resnet.Bottleneck, [3, 4, 23, 3], bins)
        case "ResNet152":
            return L2CS(models.resnet.Bottleneck, [3, 8, 36, 3], bins)
        case _:
            raise ValueError(f"Invalid L2CS architecture: {arch!r}")
