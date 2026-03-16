"""FabNet facial feature extractor wrapper."""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from exordium import WEIGHT_DIR
from exordium.utils.ckpt import download_file
from exordium.video.deep.base import VisualModelWrapper


class FabNetWrapper(VisualModelWrapper):
    """FAb-Net face encoder wrapper.

    Produces a 256-dimensional identity-preserving feature vector from a face
    crop.  Model weights are downloaded automatically on first use.

    Args:
        device_id: GPU device index. ``None`` or negative uses CPU.

    """

    def __init__(self, device_id: int = -1):
        super().__init__(device_id)
        self.remote_path = (
            "https://github.com/fodorad/exordium/releases/download/v1.0.0/fabnet_weights.pth"
        )
        self.local_path = WEIGHT_DIR / "fabnet" / Path(self.remote_path).name
        download_file(self.remote_path, self.local_path)
        state_dict = torch.load(
            str(self.local_path), weights_only=False, map_location=torch.device("cpu")
        )
        self.model = FrontaliseModelMasks_wider()
        self.model.load_state_dict(state_dict["state_dict_model"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([T.ToTensor(), T.Resize((256, 256))])

        logging.info(f"FAb-Net is loaded to {self.device}.")

    def _preprocess(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        """Stack RGB numpy frames into a (B, 3, 256, 256) tensor on self.device.

        Args:
            frames: RGB uint8 arrays each of shape (H, W, 3).

        Returns:
            Tensor of shape (B, 3, 256, 256) with values in [0, 1].

        """
        return torch.stack([self.transform(Image.fromarray(f)) for f in frames]).to(self.device)

    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        """FAb-Net encoder forward pass.

        Args:
            tensor: Preprocessed face tensor of shape (B, 3, 256, 256)
                on self.device with values in [0, 1].

        Returns:
            Identity feature tensor of shape (B, 256).

        """
        feature = self.model.encoder(tensor)
        return torch.reshape(feature, (tensor.shape[0], 256))


class FrontaliseModelMasks_wider(nn.Module):  # pragma: no cover
    """FAb-Net model architecture with mask prediction branch.

    Code: https://github.com/oawiles/FAb-Net
    Authors: Olivia Wiles, A. Sophia Koepke, Andrew Zisserman
    """

    def __init__(self, inner_nc=256, num_additional_ids=32):
        super().__init__()
        self.encoder = self.generate_encoder_layers(
            output_size=inner_nc, num_filters=num_additional_ids
        )
        self.decoder = self.generate_decoder_layers(inner_nc * 2, num_filters=num_additional_ids)
        self.mask = self.generate_decoder_layers(
            inner_nc * 2, num_output_channels=1, num_filters=num_additional_ids
        )

    def generate_encoder_layers(self, output_size=128, num_filters=64) -> nn.Sequential:
        """Generate encoder layers for the FAb-Net model.

        Args:
            output_size: Size of the bottleneck output. Defaults to 128.
            num_filters: Number of base filters. Defaults to 64.

        Returns:
            Sequential module containing the encoder layers.
        """
        conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
        conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
        conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
        conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
        conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

        batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
        batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
        batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
        batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
        batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
        batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

        leaky_relu = nn.LeakyReLU(0.2, True)
        return nn.Sequential(
            conv1,
            leaky_relu,
            conv2,
            batch_norm2_0,
            leaky_relu,
            conv3,
            batch_norm4_0,
            leaky_relu,
            conv4,
            batch_norm8_0,
            leaky_relu,
            conv5,
            batch_norm8_1,
            leaky_relu,
            conv6,
            batch_norm8_2,
            leaky_relu,
            conv7,
            batch_norm8_3,
            leaky_relu,
            conv8,
        )

    def generate_decoder_layers(
        self, num_input_channels, num_output_channels=2, num_filters=32
    ) -> nn.Sequential:
        """Generate decoder layers for the FAb-Net model.

        Args:
            num_input_channels: Number of input channels.
            num_output_channels: Number of output channels. Defaults to 2.
            num_filters: Number of base filters. Defaults to 32.

        Returns:
            Sequential module containing the decoder layers.
        """
        dconv1 = nn.Conv2d(num_input_channels, num_filters * 8, 3, 1, 1)
        dconv2 = nn.Conv2d(num_filters * 8, num_filters * 8, 3, 1, 1)
        dconv3 = nn.Conv2d(num_filters * 8, num_filters * 8, 3, 1, 1)
        dconv4 = nn.Conv2d(num_filters * 8, num_filters * 8, 3, 1, 1)
        dconv5 = nn.Conv2d(num_filters * 8, num_filters * 4, 3, 1, 1)
        dconv6 = nn.Conv2d(num_filters * 4, num_filters * 2, 3, 1, 1)
        dconv7 = nn.Conv2d(num_filters * 2, num_filters, 3, 1, 1)
        dconv8 = nn.Conv2d(num_filters, num_output_channels, 3, 1, 1)

        batch_norm = nn.BatchNorm2d(num_filters)
        batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
        batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
        batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
        batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
        batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
        batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

        relu = nn.ReLU()
        tanh = nn.Tanh()

        return nn.Sequential(
            relu,
            nn.Upsample(scale_factor=2, mode="bilinear"),
            dconv1,
            batch_norm8_4,
            relu,
            nn.Upsample(scale_factor=2, mode="bilinear"),
            dconv2,
            batch_norm8_5,
            relu,
            nn.Upsample(scale_factor=2, mode="bilinear"),
            dconv3,
            batch_norm8_6,
            relu,
            nn.Upsample(scale_factor=2, mode="bilinear"),
            dconv4,
            batch_norm8_7,
            relu,
            nn.Upsample(scale_factor=2, mode="bilinear"),
            dconv5,
            batch_norm4_1,
            relu,
            nn.Upsample(scale_factor=2, mode="bilinear"),
            dconv6,
            batch_norm2_1,
            relu,
            nn.Upsample(scale_factor=2, mode="bilinear"),
            dconv7,
            batch_norm,
            relu,
            nn.Upsample(scale_factor=2, mode="bilinear"),
            dconv8,
            tanh,
        )
