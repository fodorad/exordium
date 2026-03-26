"""DINOv2 vision encoder wrapper using HuggingFace Transformers."""

import torch
import torchvision.transforms.functional as TF
from transformers import Dinov2Model

from exordium.video.deep.base import _IMAGENET_MEAN, _IMAGENET_STD, VisualModelWrapper

_MODEL_IDS: dict[str, str] = {
    "small": "facebook/dinov2-small",
    "base":  "facebook/dinov2-base",
    "large": "facebook/dinov2-large",
    "giant": "facebook/dinov2-giant",
}
"""Mapping of DINOv2 variant names to HuggingFace model IDs."""

_FEATURE_DIMS: dict[str, int] = {
    "small": 384,
    "base":  768,
    "large": 1024,
    "giant": 1536,
}
"""Output embedding dimension per DINOv2 variant."""

_DEFAULT_DINOV2_MODEL = "base"
"""Default DINOv2 variant (ViT-B/14, 768-d)."""


class DINOv2Wrapper(VisualModelWrapper):
    """Wrapper for DINOv2 frame-wise feature extraction via HuggingFace Transformers.

    Loads a DINOv2 vision encoder and extracts L2-normalised CLS-token embeddings.
    DINOv2 is a self-supervised ViT trained without language supervision, which
    makes it particularly effective for dense visual tasks such as emotion
    recognition from faces and body pose.

    The default variant is ``"base"`` (ViT-B/14, 768-d, ~86M parameters),
    offering approximately 5–7× faster inference than CLIP ViT-H/14 at
    competitive representation quality.

    Args:
        model_name: DINOv2 variant — ``"small"`` (384-d), ``"base"`` (768-d),
            ``"large"`` (1024-d), or ``"giant"`` (1536-d).
            Defaults to ``"base"``.
        device_id: GPU device index. ``None`` or negative uses CPU.

    Raises:
        ValueError: If ``model_name`` is not one of the supported variants.

    Example::

        from exordium.video.deep.dinov2 import DINOv2Wrapper

        model = DINOv2Wrapper(model_name="base", device_id=0)
        result = model.video_to_feature("clip.mp4")
        # result["features"]: (T, 768) — one embedding per frame

    """

    def __init__(
        self,
        model_name: str = _DEFAULT_DINOV2_MODEL,
        device_id: int | None = None,
    ) -> None:
        if model_name not in _MODEL_IDS:
            raise ValueError(
                f"Invalid model_name: {model_name!r}. Choose from {list(_MODEL_IDS)}."
            )
        super().__init__(device_id)
        self.feature_dim = _FEATURE_DIMS[model_name]
        self.model = Dinov2Model.from_pretrained(_MODEL_IDS[model_name])
        assert isinstance(self.model, Dinov2Model)
        self.model.eval()
        self.model.to(self.device)  # ty: ignore[invalid-argument-type]
        self._mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )
        self._std = torch.tensor(_IMAGENET_STD, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )

    def preprocess(self, frames) -> torch.Tensor:
        """Resize and normalise frames to DINOv2 input convention.

        Applies bicubic resize to 224×224 and normalises with ImageNet
        mean/std as used during DINOv2 pre-training.

        Args:
            frames: Any input supported by
                :meth:`~exordium.video.deep.base.VisualModelWrapper._to_uint8_tensor`.

        Returns:
            Float tensor of shape ``(B, 3, 224, 224)`` on ``self.device``.

        """
        x = self._to_uint8_tensor(frames).to(self.device)
        x = TF.resize(x, [224, 224], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        x = x.float().div(255)
        return (x - self._mean) / self._std

    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """DINOv2 encoder forward pass.

        Runs the vision transformer and extracts the CLS token from
        ``last_hidden_state[:, 0, :]``, then L2-normalises the result.

        Args:
            tensor: Preprocessed image tensor of shape ``(B, 3, 224, 224)``
                on ``self.device``.

        Returns:
            L2-normalised CLS-token tensor of shape ``(B, D)`` where ``D``
            is the feature dimension of the selected variant (e.g. 768 for
            ``"base"``).

        """
        outputs = self.model(pixel_values=tensor)
        cls = outputs.last_hidden_state[:, 0, :]  # (B, D) — CLS token
        return cls / cls.norm(dim=-1, keepdim=True)
