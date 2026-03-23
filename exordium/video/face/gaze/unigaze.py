"""UniGaze gaze estimation model wrapper."""

import torch
import torchvision.transforms.functional as TF

from exordium.utils.device import get_torch_device
from exordium.video.deep.base import _IMAGENET_MEAN, _IMAGENET_STD
from exordium.video.face.gaze.base import GazeWrapper


class UnigazeWrapper(GazeWrapper):
    """UniGaze gaze estimation wrapper.

    Predicts gaze direction (pitch and yaw) from face crops using a
    ViT-based model from the UniGaze family.  Weights are downloaded
    automatically by the ``unigaze`` package on first use.

    Args:
        model_name: UniGaze model variant.  Available options:

            * ``"unigaze_b16_joint"`` — ViT-B/16, smallest and fastest
            * ``"unigaze_l16_joint"`` — ViT-L/16
            * ``"unigaze_h14_joint"`` — ViT-H/14
            * ``"unigaze_h14_cross_X"`` — ViT-H/14, cross-dataset variant

        device_id: Device index.  ``None`` or negative for CPU.

    """

    def __init__(
        self,
        model_name: str = "unigaze_b16_joint",
        device_id: int | None = None,
    ) -> None:
        import unigaze as _unigaze

        self.device = get_torch_device(device_id)
        self.model = _unigaze.load(model_name, device=str(self.device))
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def preprocess(self, frames) -> torch.Tensor:
        """Resize and normalise face crops to UniGaze input convention.

        UniGaze expects 224×224 inputs normalised to ImageNet mean/std.

        Args:
            frames: Any input accepted by :meth:`~GazeWrapper._to_uint8_tensor`.

        Returns:
            Float tensor of shape ``(B, 3, 224, 224)`` on ``self.device``.

        """
        x = self._to_uint8_tensor(frames).to(self.device)
        x = TF.resize(x, [224, 224], antialias=True)
        x = x.float().div(255)
        return (x - self._mean) / self._std

    def inference(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run UniGaze and return ``(yaw, pitch)`` angles in radians.

        Performs the full forward pass and extracts gaze angles from the
        model output dict.  UniGaze returns ``pred_gaze`` of shape ``(B, 2)``
        where column 0 is pitch and column 1 is yaw, both in radians.

        Args:
            tensor: Float tensor of shape ``(B, 3, 224, 224)`` on
                ``self.device``, normalised to ImageNet mean/std.

        Returns:
            Tuple of ``(yaw, pitch)`` tensors each of shape ``(B,)``
            in radians on ``self.device``.

        """
        pred_gaze = self.model(tensor)["pred_gaze"]  # (B, 2): col 0=pitch, col 1=yaw
        return pred_gaze[:, 1], pred_gaze[:, 0]
