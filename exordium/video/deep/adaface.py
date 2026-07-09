"""AdaFace face recognition model wrapper (CVPR 2022).

Provides a :class:`VisualModelWrapper` that produces 512-dimensional
identity-discriminative face embeddings using the AdaFace method with
an IResNet backbone.

AdaFace uses a quality-adaptive angular margin loss that makes it
especially robust to degraded face images (blur, occlusion, extreme
head poses, masks, glasses, and non-frontal views).

Reference:
    Kim et al., "AdaFace: Quality Adaptive Margin for Face Recognition",
    CVPR 2022.  https://github.com/mk-minchul/AdaFace

Weights are downloaded from HuggingFace Hub on first use.
"""

import logging
from collections import namedtuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

from exordium import WEIGHT_DIR
from exordium.video.deep.base import VisualModelWrapper

logger = logging.getLogger(__name__)
"""Module-level logger."""

_FEATURE_DIM = 512
"""Output embedding dimensionality for all AdaFace IResNet backbones."""

_INPUT_SIZE = 112
"""Spatial input size expected by the IResNet backbone."""

_HF_REPO_IDS: dict[str, str] = {
    "ir_18": "minchul/cvlface_adaface_ir18_vgg2",
    "ir_50": "minchul/cvlface_adaface_ir50_ms1mv2",
    "ir_101": "minchul/cvlface_adaface_ir101_webface4m",
}
"""HuggingFace repo IDs for each supported backbone."""

_NUM_LAYERS: dict[str, int] = {
    "ir_18": 18,
    "ir_50": 50,
    "ir_101": 100,
}
"""Number of IResNet layers for each backbone variant."""

_STATE_DICT_PREFIX = "model.net."
"""Prefix to strip from safetensors state dict keys."""


class AdaFaceWrapper(VisualModelWrapper):
    """AdaFace face recognition model.

    Produces 512-dimensional L2-normalised identity embeddings from face
    crops.  Designed for robustness to low-quality and occluded faces.

    Supported backbones:

    * ``"ir_18"`` — fastest, ~43 MB (trained on VGGFace2)
    * ``"ir_50"`` — balanced, ~167 MB (trained on MS1MV2)
    * ``"ir_101"`` — most accurate, ~249 MB (trained on WebFace4M)

    Args:
        backbone: One of ``"ir_18"``, ``"ir_50"``, ``"ir_101"``.
        device_id: GPU device index.  ``None`` or negative uses CPU.

    Raises:
        ValueError: If ``backbone`` is not a supported variant.

    """

    def __init__(self, backbone: str = "ir_50", device_id: int | None = None):
        if backbone not in _HF_REPO_IDS:
            msg = f"Unknown backbone {backbone!r}. Choose from {sorted(_HF_REPO_IDS)}."
            raise ValueError(msg)

        super().__init__(device_id)
        self.backbone_name = backbone
        self.feature_dim = _FEATURE_DIM

        num_layers = _NUM_LAYERS[backbone]
        self.model = _IResNetBackbone(
            input_size=(_INPUT_SIZE, _INPUT_SIZE),
            num_layers=num_layers,
            output_dim=_FEATURE_DIM,
        )

        state_dict = self._load_weights(backbone)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"AdaFace ({backbone}) loaded to {self.device}.")

    def _load_weights(self, backbone: str) -> dict[str, torch.Tensor]:
        """Download and load AdaFace weights from HuggingFace Hub.

        Args:
            backbone: Backbone variant name.

        Returns:
            State dict with ``model.net.`` prefix stripped.

        """
        from huggingface_hub import hf_hub_download

        repo_id = _HF_REPO_IDS[backbone]
        cache_dir = WEIGHT_DIR / "adaface" / backbone
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / "model.safetensors"

        if not local_path.exists():
            logger.info(f"Downloading {repo_id}/model.safetensors → {local_path}")
            hf_hub_download(
                repo_id=repo_id,
                filename="model.safetensors",
                local_dir=str(cache_dir),
            )

        raw = load_file(str(local_path))
        prefix_len = len(_STATE_DICT_PREFIX)
        return {
            (k[prefix_len:] if k.startswith(_STATE_DICT_PREFIX) else k): v for k, v in raw.items()
        }

    def preprocess(self, frames) -> torch.Tensor:
        """Resize RGB face crops to 112x112 and normalise to [-1, 1].

        Args:
            frames: Any input supported by
                :meth:`~exordium.video.deep.base.VisualModelWrapper._to_uint8_tensor`.

        Returns:
            Float tensor of shape ``(B, 3, 112, 112)`` on ``self.device``.

        """
        x = self._to_uint8_tensor(frames).to(self.device)
        x = TF.resize(x, [_INPUT_SIZE, _INPUT_SIZE], antialias=True)
        return x.float().div(255).sub(0.5).div(0.5)

    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """IResNet forward pass with L2 normalisation.

        Args:
            tensor: Preprocessed face tensor of shape ``(B, 3, 112, 112)``
                on ``self.device`` with values in [-1, 1].

        Returns:
            L2-normalised identity embedding of shape ``(B, 512)``.

        """
        features = self.model(tensor)
        return nn.functional.normalize(features, p=2, dim=1)


# ---------------------------------------------------------------------------
# IResNet backbone (adapted from AdaFace / InsightFace)
# ---------------------------------------------------------------------------
# Simplified from the original code to remove fvcore / omegaconf / yaml
# dependencies.  Only the inference-relevant parts are kept.
# ---------------------------------------------------------------------------


def _initialize_weights(modules: nn.Module) -> None:
    """Kaiming-initialise conv, linear, and batch-norm layers."""
    for m in modules.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()


class _SEModule(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight.data)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SE attention."""
        w = self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))
        return x * w


class _BasicBlockIR(nn.Module):
    """Basic residual block for IResNet."""

    def __init__(self, in_channel: int, depth: int, stride: int):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, 1, stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, 3, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, 3, stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        return self.res_layer(x) + self.shortcut_layer(x)


class _BasicBlockIRSE(_BasicBlockIR):
    """Basic residual block for IResNet with SE attention."""

    def __init__(self, in_channel: int, depth: int, stride: int):
        super().__init__(in_channel, depth, stride)
        self.res_layer.add_module("se_block", _SEModule(depth, 16))


_Bottleneck = namedtuple("_Bottleneck", ["in_channel", "depth", "stride"])


def _get_block(in_channel: int, depth: int, num_units: int, stride: int = 2) -> list[_Bottleneck]:
    """Build a list of bottleneck specs for one stage."""
    return [_Bottleneck(in_channel, depth, stride)] + [
        _Bottleneck(depth, depth, 1) for _ in range(num_units - 1)
    ]


_BLOCK_CONFIGS: dict[int, list[list[_Bottleneck]]] = {
    18: [
        _get_block(64, 64, 2),
        _get_block(64, 128, 2),
        _get_block(128, 256, 2),
        _get_block(256, 512, 2),
    ],
    50: [
        _get_block(64, 64, 3),
        _get_block(64, 128, 4),
        _get_block(128, 256, 14),
        _get_block(256, 512, 3),
    ],
    100: [
        _get_block(64, 64, 3),
        _get_block(64, 128, 13),
        _get_block(128, 256, 30),
        _get_block(256, 512, 3),
    ],
}
"""Block configurations for supported IResNet depths."""


class _IResNetBackbone(nn.Module):
    """Improved ResNet backbone for face recognition.

    Supports 18, 50, and 100-layer variants with optional SE attention.

    Args:
        input_size: Spatial input dimensions ``(H, W)``.  Must be ``(112, 112)``.
        num_layers: Number of layers (18, 50, or 100).
        mode: ``"ir"`` (default) or ``"ir_se"`` for SE attention blocks.
        output_dim: Embedding dimensionality.

    """

    def __init__(
        self,
        input_size: tuple[int, int] = (112, 112),
        num_layers: int = 50,
        mode: str = "ir",
        output_dim: int = 512,
    ):
        super().__init__()

        if input_size != (112, 112):
            msg = f"Only (112, 112) input is supported, got {input_size}."
            raise ValueError(msg)
        if num_layers not in _BLOCK_CONFIGS:
            msg = f"Unsupported num_layers={num_layers}. Choose from {sorted(_BLOCK_CONFIGS)}."
            raise ValueError(msg)
        if mode not in ("ir", "ir_se"):
            msg = f"mode must be 'ir' or 'ir_se', got {mode!r}."
            raise ValueError(msg)

        unit_module = _BasicBlockIRSE if mode == "ir_se" else _BasicBlockIR

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        body_modules: list[nn.Module] = []
        for block in _BLOCK_CONFIGS[num_layers]:
            for bottleneck in block:
                body_modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth, bottleneck.stride)
                )
        self.body = nn.Sequential(*body_modules)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, output_dim),
            nn.BatchNorm1d(output_dim, affine=False),
        )

        _initialize_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract face embeddings.

        Args:
            x: Input tensor ``(B, 3, 112, 112)`` with values in [-1, 1].

        Returns:
            Embedding tensor ``(B, output_dim)``.

        """
        x = self.input_layer(x)
        x = self.body(x)
        return self.output_layer(x)
