"""EmotiEffNet facial expression feature extractor wrapper.

Wraps the EmotiEffNet models from the EmotiEffLib project for penultimate-layer
feature extraction.  Models are EfficientNet backbones (B0 or B2) pre-trained on
VGGFace2 and fine-tuned on AffectNet for facial expression recognition.

Weights are downloaded on first use from the EmotiEffLib GitHub repository and
cached locally.  The final classification head is replaced with an identity
layer so that :meth:`inference` returns penultimate features rather than class
logits.

The checkpoints from EmotiEffLib are full pickled models saved with an older
``timm`` version (0.9.x).  To avoid compatibility issues with newer ``timm``
releases, this wrapper creates a fresh architecture via ``timm.create_model``
and loads only the ``state_dict`` extracted from the pickle.

References:
    * https://github.com/sb-ai-lab/EmotiEffLib
    * Savchenko (2022), *HSEmotion: High-Speed Emotion Recognition Library*
"""

import logging
from collections.abc import Sequence

import timm
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from exordium import WEIGHT_DIR
from exordium.utils.ckpt import download_file
from exordium.video.deep.base import _IMAGENET_MEAN, _IMAGENET_STD, VisualModelWrapper

logger = logging.getLogger(__name__)
"""Module-level logger."""

_WEIGHT_URL = (
    "https://github.com/sb-ai-lab/EmotiEffLib/blob/main/"
    "models/affectnet_emotions/{name}.pt?raw=true"
)
"""URL template for downloading EmotiEffNet checkpoint files."""

_MODELS: dict[str, dict] = {
    "enet_b0_8_best_vgaf": {
        "timm_name": "tf_efficientnet_b0.ns_jft_in1k",
        "num_classes": 8,
        "img_size": 224,
        "feature_dim": 1280,
    },
    "enet_b0_8_best_afew": {
        "timm_name": "tf_efficientnet_b0.ns_jft_in1k",
        "num_classes": 8,
        "img_size": 224,
        "feature_dim": 1280,
    },
    "enet_b0_8_va_mtl": {
        "timm_name": "tf_efficientnet_b0.ns_jft_in1k",
        "num_classes": 8,
        "img_size": 224,
        "feature_dim": 1280,
    },
    "enet_b2_8": {
        "timm_name": "tf_efficientnet_b2.ns_jft_in1k",
        "num_classes": 8,
        "img_size": 260,
        "feature_dim": 1408,
    },
    "enet_b2_7": {
        "timm_name": "tf_efficientnet_b2.ns_jft_in1k",
        "num_classes": 7,
        "img_size": 260,
        "feature_dim": 1408,
    },
}
"""Supported model variants with their input resolution and feature dimension."""

_DEFAULT_MODEL = "enet_b0_8_best_vgaf"
"""Default EmotiEffNet variant."""


def _load_state_dict_from_pickle(path: str) -> dict[str, torch.Tensor]:
    """Load state_dict from an EmotiEffLib full-model pickle.

    The B0 checkpoints store the classifier as ``nn.Sequential([Linear])``,
    producing keys like ``classifier.0.weight``.  The fresh timm model uses a
    plain ``nn.Linear``, so keys must be remapped to ``classifier.weight``.

    Args:
        path: Path to the ``.pt`` checkpoint file.

    Returns:
        A state_dict compatible with a fresh ``timm`` EfficientNet model.

    """
    old_model = torch.load(path, map_location="cpu", weights_only=False)
    sd = old_model.state_dict()
    return {k.replace("classifier.0.", "classifier."): v for k, v in sd.items()}


class EmotiEffNetWrapper(VisualModelWrapper):
    """EmotiEffNet penultimate-layer feature extractor.

    Loads an EfficientNet-B0 or B2 model pre-trained for facial expression
    recognition and extracts features from the penultimate layer (before the
    classification head).  The output is a raw activation vector suitable for
    downstream classifiers.

    Model weights are downloaded automatically on first use from the
    EmotiEffLib GitHub repository.

    Supported variants:

    +--------------------------+-------+-----------+
    | ``model_name``           | Input | Feat dim  |
    +==========================+=======+===========+
    | ``enet_b0_8_best_vgaf``  | 224   | 1280      |
    +--------------------------+-------+-----------+
    | ``enet_b0_8_best_afew``  | 224   | 1280      |
    +--------------------------+-------+-----------+
    | ``enet_b0_8_va_mtl``     | 224   | 1280      |
    +--------------------------+-------+-----------+
    | ``enet_b2_8``            | 260   | 1408      |
    +--------------------------+-------+-----------+
    | ``enet_b2_7``            | 260   | 1408      |
    +--------------------------+-------+-----------+

    Args:
        model_name: Model variant to load.  Defaults to
            ``"enet_b0_8_best_vgaf"``.
        device_id: GPU device index.  ``None`` or negative uses CPU.

    Raises:
        ValueError: If ``model_name`` is not one of the supported variants.

    Example::

        from exordium.video.deep.emotieffnet import EmotiEffNetWrapper

        model = EmotiEffNetWrapper(model_name="enet_b2_8", device_id=0)
        features = model(face_crop)       # (1, 1408)
        result = model.video_to_feature("clip.mp4")
        # result["features"]: (T, 1408) — one embedding per frame

    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device_id: int | None = None,
    ) -> None:
        if model_name not in _MODELS:
            raise ValueError(f"Invalid model_name: {model_name!r}. Choose from {sorted(_MODELS)}.")
        super().__init__(device_id)

        cfg = _MODELS[model_name]
        self.img_size: int = cfg["img_size"]
        self.feature_dim: int = cfg["feature_dim"]

        # Download weights from EmotiEffLib GitHub repo.
        weight_dir = WEIGHT_DIR / "emotieffnet"
        weight_file = f"{model_name}.pt"
        local_path = weight_dir / weight_file
        download_file(
            _WEIGHT_URL.format(name=model_name),
            local_path,
        )

        # Create a fresh model from the current timm version and load only the
        # state_dict extracted from the legacy pickle.
        state_dict = _load_state_dict_from_pickle(str(local_path))
        model = timm.create_model(
            cfg["timm_name"],
            pretrained=False,
            num_classes=cfg["num_classes"],
        )
        model.load_state_dict(state_dict)

        # Replace the classification head with Identity to expose penultimate
        # features.
        model.classifier = nn.Identity()

        model.to(self.device)
        model.eval()
        self.model: nn.Module = model

        # Normalisation constants (ImageNet, same as used during training).
        self._mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )
        self._std = torch.tensor(_IMAGENET_STD, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )

        logger.info(
            "EmotiEffNet (%s) loaded to %s  —  img_size=%d, feature_dim=%d",
            model_name,
            self.device,
            self.img_size,
            self.feature_dim,
        )

    def preprocess(
        self,
        frames: torch.Tensor | Sequence,
    ) -> torch.Tensor:
        """Resize and normalise frames for EmotiEffNet.

        Applies bilinear resize to the model's expected input resolution and
        normalises with ImageNet mean/std.

        Args:
            frames: Any input supported by
                :meth:`~exordium.video.deep.base.VisualModelWrapper._to_uint8_tensor`.

        Returns:
            Float tensor of shape ``(B, 3, H, H)`` on ``self.device``, where
            ``H`` is :attr:`img_size` (224 or 260).

        """
        x = self._to_uint8_tensor(frames).to(self.device)
        x = TF.resize(x, [self.img_size, self.img_size], antialias=True)
        x = x.float().div(255)
        return (x - self._mean) / self._std

    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """EmotiEffNet forward pass returning penultimate-layer features.

        The classification head has been replaced with an identity layer, so
        the model returns raw penultimate activations.

        Args:
            tensor: Preprocessed face tensor of shape
                ``(B, 3, H, H)`` on ``self.device``.

        Returns:
            Feature tensor of shape ``(B, D)`` where ``D`` is
            :attr:`feature_dim` (1280 for B0 variants, 1408 for B2).

        """
        return self.model(tensor)  # (B, D)
