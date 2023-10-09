import logging
from pathlib import Path
from typing import Sequence
import cv2
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as T
from exordium import WEIGHT_DIR
from exordium.utils.ckpt import download_file
from exordium.video.io import images2np, batch_iterator
from exordium.video.detection import Track
from exordium.utils.decorator import load_or_create
# from exordium.utils.ckpt import download_file


class FabNetWrapper:
    """FAb-Net wrapper class."""

    def __init__(self, gpu_id: int = 0):
        # Weights are already prepared in RESOURCE_DIR
        #   remote_path = 'https://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/release_bmvc_fabnet.zip'
        #   local_path = RESOURCE_DIR / 'fabnet' / Path(remote_path).name
        #   download_file(remote_path, local_path)
        #   os.system(f'unzip {local_path} -d {local_path.parent}')
        self.remote_path = 'https://github.com/fodorad/exordium/releases/download/v1.0.0/fabnet_weights.pth'
        self.local_path = WEIGHT_DIR / 'fabnet' / Path(self.remote_path).name
        download_file(self.remote_path, self.local_path)
        self.device = f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu'
        state_dict = torch.load(str(self.local_path))
        self.model = FrontaliseModelMasks_wider()
        self.model.load_state_dict(state_dict['state_dict_model'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((256, 256))
        ])

        logging.info(f'FAb-Net is loaded to {self.device}.')

    def __call__(self, faces: Sequence[np.ndarray]) -> np.ndarray:
        """FAb-Net inference.

        Args:
            faces (list[np.ndarray]): list of face images of shape (H, W, C) and BGR channel order.

        Returns:
            np.ndarray: FAb-Net features with shape (B, 256)
        """
        samples = images2np(faces, 'BGR', resize=(256, 256)) # (B, H, W, C) == (B, 256, 256, 3)
        samples = torch.stack([self.transform(sample) for sample in samples]).to(self.device) # (B, C, H, W) == (B, 3, 256, 256)

        if samples.ndim != 4:
            raise Exception(f'Invalid input shape. Expected sample shape is (B, C, H, W) got instead {samples.shape}.')

        with torch.no_grad():
            feature = self.model.encoder(samples)

        feature = feature.detach().cpu()
        feature = torch.reshape(feature, shape=(samples.shape[0], 256))
        feature = feature.numpy()

        if not feature.shape == (samples.shape[0], 256):
            raise Exception(f'Invalid output shape. Expected feature shape is {(samples.shape[0], 256)} got instead {feature.shape}.')

        return feature

    @load_or_create('npy')
    def dir_to_feature(self, frame_dir: list[str], batch_size: int = 30, verbose: bool = False, **kwargs) -> np.ndarray:
        img_paths = sorted(list(Path(frame_dir).glob('*.png')))
        ids, features = [], []
        preprocess = lambda image_path: cv2.resize(cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED), (256, 256), interpolation=cv2.INTER_AREA)
        for index in tqdm(range(0, len(img_paths), batch_size), total=np.ceil(len(img_paths)/batch_size).astype(int), desc='FAb-Net extraction', disable=not verbose):
            batch_paths = img_paths[index:index+batch_size]
            ids += [int(p.stem) for p in batch_paths]
            samples = np.stack([preprocess(image_path) for image_path in batch_paths])
            feature = self(samples)
            features.append(feature)
        features = np.concatenate(features, axis=0)
        return ids, features

    @load_or_create('pkl')
    def track_to_feature(self, track: Track, batch_size: int = 30, **kwargs) -> np.ndarray:
        ids, features = [], []
        for subset in batch_iterator(track, batch_size):
            ids += [detection.frame_id for detection in subset if not detection.is_interpolated]
            samples = [detection.bb_crop() for detection in subset if not detection.is_interpolated] # (B, H, W, C)
            feature = self(samples)
            features.append(feature)
        features = np.concatenate(features, axis=0)
        return ids, features


#################################################################
#                                                               #
#   Code: https://github.com/oawiles/FAb-Net                    #
#   Authors: Olivia Wiles, A. Sophia Koepke, Andrew Zisserman   #
#                                                               #
#################################################################

class FrontaliseModelMasks_wider(nn.Module):

	def __init__(self, inner_nc=256, num_additional_ids=32):
		super().__init__()
		self.encoder = self.generate_encoder_layers(output_size=inner_nc, num_filters=num_additional_ids)
		self.decoder = self.generate_decoder_layers(inner_nc*2, num_filters=num_additional_ids)
		self.mask = self.generate_decoder_layers(inner_nc*2, num_output_channels=1, num_filters=num_additional_ids)

	def generate_encoder_layers(self, output_size=128, num_filters=64):
		conv1 = nn.Conv2d(3, num_filters, 4, 2, 1)
		conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
		conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
		conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
		conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
		conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
		conv7 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
		conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

		batch_norm = nn.BatchNorm2d(num_filters)
		batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
		batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
		batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

		leaky_relu = nn.LeakyReLU(0.2, True)
		return nn.Sequential(conv1, leaky_relu,
                             conv2, batch_norm2_0, leaky_relu,
                             conv3, batch_norm4_0, leaky_relu,
							 conv4, batch_norm8_0, leaky_relu,
                             conv5, batch_norm8_1, leaky_relu,
                             conv6, batch_norm8_2, leaky_relu,
                             conv7, batch_norm8_3, leaky_relu,
                             conv8)

	def generate_decoder_layers(self, num_input_channels, num_output_channels=2, num_filters=32):
		up = nn.Upsample(scale_factor=2, mode='bilinear')

		dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
		dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
		dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
		dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 8, 3, 1, 1)
		dconv5 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
		dconv6 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
		dconv7 = nn.Conv2d(num_filters * 2 , num_filters, 3, 1, 1)
		dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

		batch_norm = nn.BatchNorm2d(num_filters)
		batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
		batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
		batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
		batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
		batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
		batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

		leaky_relu = nn.LeakyReLU(0.2)
		relu = nn.ReLU()
		tanh = nn.Tanh()

		return nn.Sequential(relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv1, batch_norm8_4,
							 relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv2, batch_norm8_5,
                             relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv3, batch_norm8_6,
                             relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv4, batch_norm8_7,
                             relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv5, batch_norm4_1,
							 relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv6, batch_norm2_1,
                             relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv7, batch_norm,
							 relu, nn.Upsample(scale_factor=2, mode='bilinear'), dconv8, tanh)