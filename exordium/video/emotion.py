import sys
from pathlib import Path

import cv2
import torch
import numpy as np
from torchvision import transforms 
from exordium.utils.shared import get_project_root
from exordium.video.face import crop_and_align_face

sys.path.append(str((get_project_root() / 'tools' / 'emonet').resolve()))
from emonet.models.emonet import EmoNet

# For 8 emotions
CLASS_NAMES = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
    7: 'Contempt',
}


class EmoNetWrapper():

    def __init__(self, n_classes: int = 8, device: str = 'cuda:0') -> None:
        assert n_classes in {5, 8}, 'Only 5 and 8 emotion classes are supported.'
        state_dict_path = get_project_root() / 'tools' / 'emonet' / 'pretrained' / f'emonet_{n_classes}.pth'
        state_dict = torch.load(str(state_dict_path), map_location='cpu')
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
        net = EmoNet(n_expression=n_classes)
        net.load_state_dict(state_dict, strict=False)
        net = net.to(device)
        net.eval()
        
        self.device = device
        self.model = net
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def load_face(self, path: str):
        face_bgr = crop_and_align_face([path], size=256, verbose=True)[0]['img']
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        assert face_rgb.shape == (256, 256, 3), f'Invalid shape: {face_rgb.shape}'
        return face_rgb

    def heatmap_to_image(self, heatmap):
        s = np.sum(heatmap.squeeze(), axis=0, keepdims=True)
        s_n = s / np.max(s)
        img = (np.transpose(s_n, (1,2,0)) * 255).astype(np.uint8)
        return img

    def extract_emonet(self, face_rgb: np.ndarray):
        sample = self.transform(face_rgb)
        sample = sample.to(self.device)
        sample = torch.unsqueeze(sample, dim=0)
        assert sample.ndim == 4, \
            'Image is expected with shape (batch_size, channel_size, height, width)'
        out_dict = self.model(sample)
        expression = out_dict['expression'].cpu().detach().numpy()
        valence = out_dict['valence'].cpu().detach().numpy()
        arousal = out_dict['arousal'].cpu().detach().numpy()
        # heatmap = out_dict['heatmap'].cpu().detach().numpy()
        expression = int(np.argmax(np.squeeze(expression))) # discrete classes
        valence = np.round(np.clip(np.squeeze(valence), -1.0, 1.0), decimals=3) # negative-positive
        arousal = np.round(np.clip(np.squeeze(arousal), -1.0, 1.0), decimals=3) # calm-excited
        return expression, valence, arousal

    def visualize(self, valence: float, arousal: float, output_path: str | Path) -> None:
        va_wheel_path = get_project_root() / 'exordium' / 'video' / 'resource' / 'va_wheel.jpg'
        assert va_wheel_path.exists(), f'Invalid path to va_wheel.jpg: {va_wheel_path}'
        va_wheel = cv2.imread(str(va_wheel_path))
        height, width, _ = va_wheel.shape
        mapped_x = int(valence * height // 2 + height // 2)
        mapped_y = int(-arousal * width // 2 + width // 2)
        radius = 10
        color = (0, 0, 255)
        thickness = -1
        image = cv2.circle(va_wheel, (mapped_y, mapped_x), radius, color, thickness)
        cv2.imwrite(str(output_path), image)


if __name__ == '__main__':
    
    paths = [
        'data/processed/frame/00000.png',
        'data/processed/frame/00001.png',
        'data/processed/frame/00002.png',
        'data/processed/frame/00003.jpg',
        'data/processed/frame/00004.jpg',
        'data/processed/frame/00005.jpg',
        'data/processed/frame/00006.jpg',
        'data/processed/frame/00007.jpg',
        'data/processed/frame/00008.jpg',
        'data/processed/frame/00009.jpg',
    ]
    m = EmoNetWrapper()

    for i, path in enumerate(paths):
        f = m.load_face(path)
        o = m.extract_emonet(f)
        print(f'path: {path} emotion: {CLASS_NAMES[o[0]]} valence: {o[1]} arousal: {o[2]}')
        m.visualize(o[1], o[2], f'test_{i}.png')
    