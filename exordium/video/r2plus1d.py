import logging
import numpy as np
import torch
from torchvision.transforms import Compose
from einops.layers.torch import Rearrange, Reduce
from exordium.video.transform import ToTensor, Resize, CenterCrop, Normalize


class R2plus1DWrapper(torch.nn.Module):
    """R2+1D wrapper class.

    paper: https://arxiv.org/pdf/1711.11248.pdf
    code: https://github.com/moabitcoin/ig65m-pytorch
    """
    def __init__(self, pool_spatial: str = "mean", pool_temporal: str = "mean"):
        super().__init__()
        self.model = torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_kinetics", num_classes=400, pretrained=True)
        self.pool_spatial = Reduce("n c t h w -> n c t", reduction=pool_spatial)
        self.pool_temporal = Reduce("n c t -> n c", reduction=pool_temporal)
        self.transform = Compose([
            ToTensor(),
            Rearrange("t h w c -> c t h w"),
            Resize(112), # shorter side to 112, keeping scale
            CenterCrop(112),
            Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])

    def __call__(self, video: np.ndarray) -> np.ndarray:
        if video.ndim == 4 and video.shape[0] != 32 and video.shape[3] == 3:
            raise ValueError(f'Invalid input video. Expected np.ndarray of shape (T, H, W, C) == (32, H, W, 3) got instead {video.shape}.')

        sample = self.transform(video) # (T, H, W, C) -> (C, T, H, W)
        sample = sample.unsqueeze(0) # (B, C, T, H, W)
        output = self.forward(sample) # (B, 512)
        return output.detach().cpu().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.pool_spatial(x)
        x = self.pool_temporal(x)
        return x

    def freeze(self, layer_names: list[str] | None = None) -> None:

        # freeze feature extractors, finetune last block (5th)
        if layer_names is None:
            layer_names = ['stem', 'layer1', 'layer2', 'layer3']

        layer_counter = 0
        for (name, module) in self.model.named_children():
            if name in layer_names:
                for layer in module.children():
                    for param in layer.parameters():
                        param.requires_grad = False

                    logging.info(f'Layer {layer_counter} in module {name} is frozen!')
                    layer_counter += 1

    def unfreeze(self) -> None:
        layer_counter = 0
        for (name, module) in self.model.named_children():
            for layer in module.children():
                for param in layer.parameters():
                    param.requires_grad = True

                logging.info(f'Layer {layer_counter} in module {name} is unfrozen!')
                layer_counter += 1

    def remove_head(self) -> None:
        self.model.fc = torch.nn.Identity()

    def init_new_head(self, num_outputs: int) -> None:
        self.model.fc = torch.nn.Linear(in_features=512, out_features=num_outputs)