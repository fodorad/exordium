"""UniGaze gaze estimation model wrapper."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import unigaze
from PIL import Image
from torchvision import transforms

from exordium.utils.device import get_torch_device
from exordium.video.core.io import images_to_np
from exordium.video.face.gaze.base import GazeWrapper
from exordium.video.face.transform import rotate_face


class UnigazeWrapper(GazeWrapper):
    """UniGaze gaze estimation wrapper.

    Predicts gaze direction (pitch and yaw) from face crops using a
    ViT-based model from the UniGaze family.

    Args:
        model_name: UniGaze model variant.  Available:
            ``unigaze_b16_joint`` (ViT-B/16, smallest),
            ``unigaze_l16_joint`` (ViT-L/16),
            ``unigaze_h14_joint`` (ViT-H/14),
            ``unigaze_h14_cross_X`` (ViT-H/14 cross-dataset).
        device_id: Device index.  ``None`` or negative for CPU.

    """

    def __init__(self, model_name: str = "unigaze_b16_joint", device_id: int | None = None) -> None:
        self.device = get_torch_device(device_id)
        self.model = unigaze.load(model_name, device=str(self.device))

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.inference_mode()
    def __call__(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict gaze angles from a preprocessed face tensor.

        Args:
            samples: Preprocessed face tensor of shape ``(B, 3, 224, 224)``
                on the model device.

        Returns:
            Tuple of ``(yaw, pitch)`` tensors each of shape ``(B,)`` in
            radians.

        """
        output = self.model(samples)
        pred_gaze = output["pred_gaze"]  # (B, 2) → (pitch, yaw)
        return pred_gaze[:, 1], pred_gaze[:, 0]  # yaw, pitch

    def predict_pipeline(
        self,
        faces: Sequence[str | Path | Image.Image | np.ndarray],
        roll_angles: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict gaze from face images with optional head-roll correction.

        Args:
            faces: Face images (paths, PIL images, or RGB numpy arrays).
            roll_angles: Per-face roll angles in degrees.  If provided each
                face is rotated to align upright before inference.

        Returns:
            Tuple of ``(yaw, pitch)`` numpy arrays each of shape ``(B,)``
            in radians.

        """
        faces_rgb = images_to_np(faces, "RGB", resize=(224, 224))

        if roll_angles is not None:
            faces_rgb = [rotate_face(face, roll)[0] for face, roll in zip(faces_rgb, roll_angles)]

        samples = torch.stack([self.transform(Image.fromarray(face)) for face in faces_rgb]).to(
            self.device
        )

        yaw, pitch = self(samples)
        return yaw.detach().cpu().numpy(), pitch.detach().cpu().numpy()
