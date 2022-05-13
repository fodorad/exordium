import numpy as np
import torch
from torchvision.transforms import Compose 
from einops.layers.torch import Rearrange, Reduce


class ToTensor:
    def __call__(self, x):
        return torch.from_numpy(np.array(x)).float() / 255.


class Resize:
    def __init__(self, size, mode="bilinear"):
        self.size = size
        self.mode = mode

    def __call__(self, video):
        size = self.size
        scale = None

        if isinstance(size, int):
            scale = float(size) / min(video.shape[-2:])
            size = None

        return torch.nn.functional.interpolate(video, size=size, scale_factor=scale,
                                               mode=self.mode, align_corners=False, recompute_scale_factor=True)


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        size = self.size

        if isinstance(size, int):
            size = size, size

        th, tw = size
        h, w = video.shape[-2:]

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return video[..., i:(i + th), j:(j + tw)]


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        shape = (-1,) + (1,) * (video.dim() - 1)

        mean = torch.as_tensor(self.mean, device=video.device).reshape(shape)
        std = torch.as_tensor(self.std, device=video.device).reshape(shape)

        return (video - mean) / std


class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        shape = (-1,) + (1,) * (video.dim() - 1)

        mean = torch.as_tensor(self.mean, device=video.device).reshape(shape)
        std = torch.as_tensor(self.std, device=video.device).reshape(shape)

        return (video * std) + mean


def get_model():
    return torch.hub.load("moabitcoin/ig65m-pytorch", "r2plus1d_34_32_kinetics", num_classes=400, pretrained=True)


def freeze(model, layer_names: list = None, verbose: bool = True):

    # freeze feature extractors, finetune last block (5th)
    if layer_names is None:
        layer_names = ['stem', 'layer1', 'layer2', 'layer3']

    layer_counter = 0
    for (name, module) in model.named_children():
        if name in layer_names:
            for layer in module.children():
                for param in layer.parameters():
                    param.requires_grad = False
                
                if verbose: print('Layer "{}" in module "{}" is frozen!'.format(layer_counter, name))
                layer_counter+=1


def unfreeze(model, verbose: bool = True):
    layer_counter = 0
    for (name, module) in model.named_children():
        for layer in module.children():
            for param in layer.parameters():
                param.requires_grad = True
            
            if verbose: print('Layer "{}" in module "{}" is unfrozen!'.format(layer_counter, name))
            layer_counter+=1


def get_parameters(model):
    return filter(lambda p: p.requires_grad, model.parameters())


def init_new_head(model, num_outputs: int):
    model.fc = torch.nn.Linear(in_features=512, out_features=num_outputs)


def remove_head(model):
    model.fc = torch.nn.Identity()


class VideoModel(torch.nn.Module):
    def __init__(self, pool_spatial="mean", pool_temporal="mean"):
        super().__init__()
        self.model = get_model()
        self.pool_spatial = Reduce("n c t h w -> n c t", reduction=pool_spatial)
        self.pool_temporal = Reduce("n c t -> n c", reduction=pool_temporal)

    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.pool_spatial(x)
        x = self.pool_temporal(x)
        return x


if __name__ == '__main__':

    transform = Compose([
        ToTensor(),
        Rearrange("t h w c -> c t h w"),
        Resize(112), # shorter side to 112, keeping scale
        CenterCrop(112),
        Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    ])

    from torchvision.io import read_video
    video_path = 'data/videos/9KAqOrdiZ4I.001.mp4'
    frames, _, _ = read_video(video_path, pts_unit='sec') # (T, H, W, C)
    # frames = frames.permute(0, 3, 1, 2) # (T, H, W, C) -> (T, C, H, W)
    sample = transform(frames) # (T, H, W, C) -> (C, T, H, W)
    sample = sample.unsqueeze(0) # (B, C, T, H, W)

    model = get_model()
    freeze(model)
    unfreeze(model)
    remove_head(model)
    output = model(sample) # (B, 512)
    print(output.shape)
    model2 = VideoModel()
    output2 = model2(sample) # (B, 512)
    print(output2.shape)