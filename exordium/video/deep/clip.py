from collections.abc import Sequence

import numpy as np
import open_clip
import torch
from PIL import Image

from exordium.video.deep.base import VisualModelWrapper


class ClipWrapper(VisualModelWrapper):
    """Wrapper for OpenCLIP image feature extraction models.

    Args:
        model_name: OpenCLIP model architecture name.
            Defaults to ``"ViT-H-14-quickgelu"``.
        pretrained: Pretrained weights tag. Defaults to ``"dfn5b"``.
        device_id: GPU device index. ``None`` or negative uses CPU.

    """

    def __init__(
        self,
        model_name: str = "ViT-H-14-quickgelu",
        pretrained: str = "dfn5b",
        device_id: int | None = None,
    ):
        super().__init__(device_id)
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model.eval()
        model.to(self.device)
        self.model = model
        self.preprocess = preprocess

    def _preprocess(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Convert RGB numpy frames to a normalised OpenCLIP input tensor.

        Args:
            frames: RGB uint8 arrays each of shape (H, W, 3).

        Returns:
            Tensor of shape (B, C, H, W) on self.device.

        """
        return torch.stack([self.preprocess(Image.fromarray(f)) for f in frames]).to(self.device)

    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """OpenCLIP image encoder forward pass.

        Args:
            tensor: Preprocessed image tensor of shape (B, C, H, W)
                on self.device.

        Returns:
            Normalised feature tensor of shape (B, D) where D depends on
            the model (1024 for ViT-H-14).

        """
        return self.model.encode_image(tensor)
