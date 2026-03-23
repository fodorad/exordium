"""CLIP vision-language model wrapper using HuggingFace Transformers."""

import torch
import torchvision.transforms.functional as TF
from transformers import CLIPVisionModelWithProjection

from exordium.video.deep.base import VisualModelWrapper

_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
"""CLIP RGB channel means for OpenAI-style normalisation."""
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
"""CLIP RGB channel standard deviations for OpenAI-style normalisation."""

_DEFAULT_CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
"""Default CLIP checkpoint (ViT-H/14, 1024-d) from HuggingFace."""


class ClipWrapper(VisualModelWrapper):
    """Wrapper for CLIP image feature extraction via HuggingFace Transformers.

    Loads a CLIP vision encoder and extracts L2-normalised image embeddings.
    The default checkpoint is ``laion/CLIP-ViT-H-14-laion2B-s32B-b79K``
    (ViT-H/14, 1024-d).

    Args:
        model_name: HuggingFace model ID for a CLIP vision encoder.
            Defaults to ``"laion/CLIP-ViT-H-14-laion2B-s32B-b79K"``.
        device_id: GPU device index. ``None`` or negative uses CPU.

    """

    def __init__(
        self,
        model_name: str = _DEFAULT_CLIP_MODEL,
        device_id: int | None = None,
    ):
        super().__init__(device_id)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        self._mean = torch.tensor(_CLIP_MEAN, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )
        self._std = torch.tensor(_CLIP_STD, dtype=torch.float32, device=self.device).view(
            1, 3, 1, 1
        )

    def preprocess(self, frames) -> torch.Tensor:
        """Resize and normalise frames to CLIP input convention.

        Applies bicubic resize to 224×224 and normalises with CLIP's
        mean/std (different from ImageNet).

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
        """CLIP vision encoder forward pass.

        Runs the vision transformer, projects to the shared embedding space,
        and L2-normalises the result.

        Args:
            tensor: Preprocessed image tensor of shape ``(B, 3, 224, 224)``
                on ``self.device``.

        Returns:
            L2-normalised feature tensor of shape ``(B, D)`` where ``D``
            depends on the model (1024 for ViT-H-14).

        """
        embeds = self.model(pixel_values=tensor).image_embeds
        return embeds / embeds.norm(dim=-1, keepdim=True)
