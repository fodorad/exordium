from pathlib import Path
import cv2
import numpy as np
import torch
from exordium import PathType


def auto_downsample_ratio(h, w):
    """Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


def visualize_image(image: torch.Tensor, output_path: PathType) -> None:
    image = image.squeeze(0)
    print(image.shape)
    if image.ndim == 3:
        image = image.permute((1,2,0))

    image = image.cpu().detach().numpy() * 255.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def visualize_video(video: torch.Tensor, output_dir: PathType) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, image in enumerate(video):
        visualize_image(image, output_dir / f'{index:06d}.png')


class RVMWrapper(torch.nn.Module):

    def __init__(self, model_name: str = "mobilenetv3", device: str = 'cuda0', downsample_ratio: float | None = None):
        super().__init__()

        if model_name not in ["mobilenetv3", "resnet50"]:
            raise ValueError("Invalid model value, choose from \{'mobilenetv3', 'resnet50'\}.")

        self.model_name = model_name
        self.device = device
        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", model_name)
        self.model.to(device)
        self.background_color = torch.tensor([.47, 1, .6]).view(3, 1, 1).to(device) # green
        self.background_color = torch.tensor([0., 0., 0.]).view(3, 1, 1).to(device) # black
        self.downsample_ratio = downsample_ratio # HD: 0.25, 4k: 0.125

    def __call__(self, input_tensor: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """RGB inputs should be normalized to [0..1], (C,H,W) or (T,C,H,W) order.
        """
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.Tensor(input_tensor)

        if self.downsample_ratio is None:
            self.downsample_ratio = auto_downsample_ratio(*input_tensor.shape[-2:])

        if input_tensor.ndim == 3:
            # image
            output, alpha = self.image_matting(input_tensor.unsqueeze(0)) # input_tensor.shape == (1,3,H,W), image.shape == (1,3,H,W), alpha.shape == (1,1,H,W)

        else: # (T,C,H,W) -> (T,C,H,W)
            # video
            output, alpha = self.video_matting(input_tensor) # input_tensor.shape == (T,3,H,W), image.shape == (T,3,H,W), alpha.shape == (T,1,H,W)

        return output, alpha

    def image_matting(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # image.shape == (1,C,H,W)
        recurrent_states = [None] * 4

        with torch.no_grad():
            foreground, alpha, *recurrent_states = self.model(image.to(self.device, non_blocking=True), *recurrent_states, self.downsample_ratio)
            image = foreground * alpha + self.background_color * (1 - alpha)

        return image, alpha

    def video_matting(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # video.shape == (T,3,H,W)
        recurrent_states = [None] * 4

        video_out = torch.clone(video)
        alpha_out = torch.zeros((video.shape[0], 1, video.shape[2], video.shape[3]))
        with torch.no_grad():
            for index, image in enumerate(video):
                foreground, alpha, *recurrent_states = self.model(image.unsqueeze(0).to(self.device, non_blocking=True), *recurrent_states, self.downsample_ratio)
                video_out[index,...] = foreground * alpha + self.background_color * (1 - alpha)
                alpha_out[index,...] = alpha

        return video_out, alpha_out