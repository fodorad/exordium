from PIL import Image
from pathlib import Path
from typing import Sequence
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from exordium import WEIGHT_DIR, PathType
from exordium.utils.ckpt import download_file
from exordium.video.io import images2np
from exordium.video.gaze import pitchyaw_to_pixel
from exordium.video.transform import rotate_face, rotate_vector


class L2csNetWrapper():
    """L2CS-Net wrapper class."""

    def __init__(self, gpu_id: int = 0):
        self.device = f'cuda:{gpu_id}' if gpu_id >= 0 else 'cpu'
        self.remote_path = 'https://github.com/fodorad/exordium/releases/download/v1.0.0/l2csnet_weights.pkl'
        self.local_path = WEIGHT_DIR / 'l2csnet' / Path(self.remote_path).name
        download_file(self.remote_path, self.local_path)
        saved_state_dict = torch.load(self.local_path)
        del saved_state_dict['fc_finetune.weight']
        del saved_state_dict['fc_finetune.bias']

        self.model = L2CS_Builder(arch='ResNet50', bins=90)
        self.model.load_state_dict(saved_state_dict)
        self.model.cuda(self.device)
        self.model.eval()

        self.softmax = nn.Softmax(dim=1)
        self.idx_tensor = torch.FloatTensor([idx for idx in range(90)]).cuda(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


    def __call__(self, faces: Sequence[PathType | Image.Image | np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Predicts gaze vector for a single face image.

        The images are loaded and converted to RGB channel order, resized, then normalized.
        The outputs of the model are the gaze direction in spherical angles (pitch and yaw).
        The prediction is returned as continues values in degrees.

        Note:
            The X axis of the Head Coordinate System should be aligned to the X axis of the Camera Coordinate System.
            The target's head roll angle in degrees can be determined with e.g. 3DDFA_V2 head pose estimation.
            Then the face with the rotate_face function from exordium.video.transform.

        Args:
            faces (list[PathType | Image.Image | np.ndarray]): sequence of face images of shape (H, W, C) and RGB channel order.

        Returns:
            tuple[float, float]: yaw and pitch angles in degrees.
        """
        faces_rgb = images2np(faces, 'RGB', resize=(448, 448)) # (B, H, W, C) == (B, 448, 448, 3)
        samples = torch.stack([self.transform(Image.fromarray(sample)) for sample in faces_rgb]).to(self.device) # (B, C, H, W) == (B, 3, 448, 448)

        # gaze prediction
        gaze_pitch, gaze_yaw = self.model(samples)
        pitch_predicted = self.softmax(gaze_pitch)
        yaw_predicted = self.softmax(gaze_yaw)

        # get continuous predictions in degrees
        pitch_predicted = torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        pitch_normed = pitch_predicted.detach().cpu().numpy() * np.pi / 180.0
        yaw_normed = yaw_predicted.detach().cpu().numpy() * np.pi / 180.0
        return yaw_normed, pitch_normed


    def predict_pipeline(self, faces: Sequence[PathType | Image.Image | np.ndarray],
                               roll_angles: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predicts gaze vector using face images and their corresponding head pose roll angles.

        Args:
            faces (Sequence[PathType | Image.Image | np.ndarray]): sequence of face images of shape (H, W, C) and RGB channel order.
            roll_angles (Sequence[float]): head pose roll angles, one per face image in "faces" argument.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                  rotated faces,
                  yaw angles,
                  pitch angles,
                  pixel coords of the vectors in the normed space,
                  pixel coords of the vectors in the original space.
        """
        # load images
        faces_rgb = images2np(faces, 'RGB', resize=(448, 448)) # (B, H, W, C) == (B, 448, 448, 3)
        # rotate faces to the normed space
        rotated_faces_rgb = [rotate_face(face, -roll_angle)[0] for face, roll_angle in zip(faces_rgb, roll_angles)]
        # predict gaze shperical angles in normed space
        yaw_normed, pitch_normed = self(rotated_faces_rgb) # (B,) and (B,)
        # convert to XY pixel coordinates
        gaze_vectors_normed = [pitchyaw_to_pixel(pitch=p_normed, yaw=y_normed) for p_normed, y_normed in zip(pitch_normed, yaw_normed)]
        # rotate back to the original space
        gaze_vectors = [rotate_vector(gaze_vector_normed, -roll_angle) for gaze_vector_normed, roll_angle in zip(gaze_vectors_normed, roll_angles)]
        return (np.stack(rotated_faces_rgb, axis=0), # (B, H, W, C)
                yaw_normed, # (B,)
                pitch_normed, # (B,)
                np.stack(gaze_vectors_normed, axis=0), # (B, 2)
                np.stack(gaze_vectors, axis=0)) # (B, 2)


class L2CS(nn.Module):

    def __init__(self, block, layers, num_bins):
        super(L2CS, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_yaw_gaze = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch_gaze = nn.Linear(512 * block.expansion, num_bins)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw_gaze =  self.fc_yaw_gaze(x)
        pre_pitch_gaze = self.fc_pitch_gaze(x)
        return pre_yaw_gaze, pre_pitch_gaze


def L2CS_Builder(arch: str = 'ResNet50', bins: int = 90):
    match arch:
        case 'ResNet18':
            model = L2CS(models.resnet.BasicBlock, [2, 2,  2, 2], bins)
        case 'ResNet34':
            model = L2CS(models.resnet.BasicBlock, [3, 4,  6, 3], bins)
        case 'ResNet50':
            model = L2CS(models.resnet.Bottleneck, [3, 4,  6, 3], bins)
        case 'ResNet101':
            model = L2CS(models.resnet.Bottleneck, [3, 4, 23, 3], bins)
        case 'ResNet152':
            model = L2CS(models.resnet.Bottleneck, [3, 8, 36, 3], bins)
        case _:
            raise ValueError('Invalid architecture')
    return model