"""ResNet architectures for deep residual learning.

Implements ResNet-18/34/50/101/152 as described in:
    He, K., Zhang, X., Ren, S., & Sun, J. (2016).
    Deep Residual Learning for Image Recognition. CVPR.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from exordium import WEIGHT_DIR
from exordium.utils.ckpt import download_file
from exordium.video.deep.base import VisualModelWrapper


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "ResNetWrapper",
]

_FEATURE_DIM: dict[str, int] = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
}

_model_urls: dict[str, str] = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class ResNetWrapper(VisualModelWrapper):
    """ResNet feature extractor wrapper.

    Loads a pretrained ResNet backbone and extracts global average-pooled
    spatial features, producing a ``(B, D)`` array where ``D`` is the
    channel dimension of the final convolutional stage.

    Feature dimensions by architecture:

    * ``resnet18`` / ``resnet34``: 512-d
    * ``resnet50`` / ``resnet101`` / ``resnet152``: 2048-d

    Args:
        arch: One of ``"resnet18"``, ``"resnet34"``, ``"resnet50"``,
            ``"resnet101"``, ``"resnet152"``. Defaults to ``"resnet50"``.
        pretrained: Load ImageNet-pretrained weights. Defaults to ``True``.
        device_id: GPU device index. ``None`` or negative uses CPU.

    """

    def __init__(
        self,
        arch: str = "resnet50",
        pretrained: bool = True,
        device_id: int | None = None,
    ) -> None:
        super().__init__(device_id)
        if arch not in _FACTORY:
            raise ValueError(f"Unknown arch '{arch}'. Choose from {list(_FACTORY)}")
        self.arch = arch
        self.feat_dim = _FEATURE_DIM[arch]
        self.model = _FACTORY[arch](pretrained=pretrained)
        self.model.eval().to(self.device)
        self.transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    def _preprocess(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Resize, crop and normalise RGB frames to ImageNet convention.

        Args:
            frames: RGB uint8 arrays each of shape ``(H, W, 3)``.

        Returns:
            Tensor of shape ``(B, 3, 224, 224)`` on ``self.device``.

        """
        return torch.stack([self.transform(Image.fromarray(f)) for f in frames]).to(self.device)

    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """ResNet forward pass with global average pooling.

        The backbone ``forward`` returns spatial features of shape
        ``(B, spatial, C)``. This method averages over the spatial dimension
        to produce ``(B, C)`` embeddings suitable for downstream tasks.

        Args:
            tensor: Preprocessed tensor of shape ``(B, 3, 224, 224)``
                on ``self.device``.

        Returns:
            Feature tensor of shape ``(B, D)``.

        """
        return self.model(tensor).mean(dim=1)


def conv3x3(in_planes, out_planes, stride=1):  # pragma: no cover
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):  # pragma: no cover
    """Residual block with two 3x3 convolutions used in ResNet-18 and ResNet-34."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """Initializes a BasicBlock.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels for the two conv layers.
            stride (int, optional): Stride for the first convolution.
                Defaults to 1.
            downsample (nn.Module | None, optional): Optional downsampling
                module applied to the residual shortcut. Defaults to None.

        """
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):  # pragma: no cover
    """Bottleneck residual block with 1x1, 3x3, 1x1 convolutions used in ResNet-50+."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """Initializes a Bottleneck block.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of channels for the 3x3 convolution (the
                output has planes * 4 channels due to the expansion factor).
            stride (int, optional): Stride for the 3x3 convolution.
                Defaults to 1.
            downsample (nn.Module | None, optional): Optional downsampling
                module applied to the residual shortcut. Defaults to None.

        """
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):  # pragma: no cover
    """Generic ResNet architecture that supports BasicBlock and Bottleneck variants."""

    def __init__(self, block, layers, num_classes=1000):
        """Initializes the ResNet model.

        Args:
            block (type): Residual block class, either BasicBlock or Bottleneck.
            layers (list[int]): Number of blocks in each of the four stages.
            num_classes (int, optional): Number of output classes for the final
                linear layer. Defaults to 1000.

        """
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
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
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
        """Forward pass through ResNet.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)

        return x


def _load_pretrained(model: ResNet, key: str) -> None:
    remote_path = _model_urls[key]
    local_path = WEIGHT_DIR / "resnet" / Path(remote_path).name
    download_file(remote_path, local_path)
    model.load_state_dict(torch.load(local_path, map_location=torch.device("cpu")))


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool, optional): If True, loads weights pre-trained on
            ImageNet. Defaults to True.
        **kwargs: Additional keyword arguments forwarded to ResNet.

    Returns:
        ResNet: ResNet-18 model instance.

    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet18")
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool, optional): If True, loads weights pre-trained on
            ImageNet. Defaults to True.
        **kwargs: Additional keyword arguments forwarded to ResNet.

    Returns:
        ResNet: ResNet-34 model instance.

    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet34")
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool, optional): If True, loads weights pre-trained on
            ImageNet. Defaults to True.
        **kwargs: Additional keyword arguments forwarded to ResNet.

    Returns:
        ResNet: ResNet-50 model instance.

    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet50")
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool, optional): If True, loads weights pre-trained on
            ImageNet. Defaults to True.
        **kwargs: Additional keyword arguments forwarded to ResNet.

    Returns:
        ResNet: ResNet-101 model instance.

    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet101")
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool, optional): If True, loads weights pre-trained on
            ImageNet. Defaults to True.
        **kwargs: Additional keyword arguments forwarded to ResNet.

    Returns:
        ResNet: ResNet-152 model instance.

    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, "resnet152")
    return model


_FACTORY = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}
