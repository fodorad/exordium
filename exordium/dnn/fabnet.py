import os
import sys
import threading
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor

from exordium.shared import get_project_root, get_weight_location, threads_eval
sys.path.append(str(Path(f'{get_project_root()}/tools/FAb-Net/FAb-Net/code').resolve()))
from models_multiview import FrontaliseModelMasks_wider


def get_weights():
    '''Downloads and FAb-Net model weights if required
    '''
    cache_dir = get_weight_location()
    weights_dir = cache_dir / 'fabnet'
    if not (weights_dir / 'release').exists():
        pretrained_weights = 'https://www.robots.ox.ac.uk/~vgg/research/unsup_learn_watch_faces/release_bmvc_fabnet.zip'
        weights_dir.mkdir(parents=True, exist_ok=True)
        if not Path(weights_dir / 'release_bmvc_fabnet.zip').exists():
            os.system(f'wget {pretrained_weights} -P {weights_dir}')
        os.system(f'unzip {weights_dir}/release_bmvc_fabnet.zip -d {weights_dir}')
    # general feature extractor and emotion classifier weights (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt)
    weights_path = weights_dir / 'release' / 'affectnet_4views.pth'
    assert weights_path.exists()
    return weights_path


def build_fabnet(include_top: bool = False):
    """Builds pretrained FAb-Net model

    Args:
        include_top (bool, optional): also return the classifier on top. Defaults to False.

    Returns:
        FrontaliseModelMasks_wider: FAb-Net feature extractor if include_top is False
        or
        Tuple[FrontaliseModelMasks_wider, nn.Sequential]: FAb-Net feature extractor and classifier if include_top is True
    """
    weights_path = get_weights()
    inner_nc = 256
    extractor = FrontaliseModelMasks_wider(3, inner_nc=inner_nc, num_additional_ids=32)
    extractor.load_state_dict(torch.load(weights_path)['state_dict_model'])
    num_classes = 8
    classifier = torch.nn.Sequential(torch.nn.BatchNorm1d(inner_nc), torch.nn.Linear(inner_nc, num_classes, bias=False))
    classifier.load_state_dict(torch.load(weights_path)['state_dict'])
    print('FAb-Net model is loaded')
    if include_top:
        return extractor, classifier
    else:
        return extractor


#from exordium.shared import timer
#@timer
def job_fabnet(device: str, frame_dirs: list, batch_size: int, output_dir: str) -> torch.Tensor:
    '''parallelize videos over multiple gpus
    '''
    print(f'[FAb-Net] Job started on {device}.')
    extractor = build_fabnet(include_top=False)
    extractor.eval()
    extractor.to(device)
    transform = Compose([Resize((256, 256)), ToTensor()])
    for frame_dir in frame_dirs:
        #img_paths = [str(Path(frame_dir) / elem.decode("utf-8")) for elem in sorted(os.listdir(frame_dir))]
        img_paths = [str(Path(frame_dir) / elem) for elem in sorted(os.listdir(frame_dir))]
        features = []
        for i in range(0, len(img_paths), batch_size):
            samples = torch.stack([transform(Image.open(img_path).convert('RGB')) for img_path in img_paths[i:i+batch_size]]) # (L, C, H, W)
            samples = samples.to(device)
            feature = extractor.encoder(samples)
            # IDEA : add support for the classifier
            # probabilities = nn.Sigmoid()(classifier(xc.squeeze()))
            # probabilities = probabilities / probabilities.sum(axis=1)[:,None]
            # probabilities = probabilities.detach().cpu().numpy().squeeze()
            feature = feature.detach().cpu().numpy().squeeze()
            assert feature.shape == (samples.shape[0], 256), f'[FAb-Net] Got shape {feature.shape} instead of the expected shape: {(samples.shape[0], 256)}.'
            features.append(feature)
        features = np.concatenate(features, axis=0)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        outfile = str(Path(output_dir) / f'{Path(frame_dir).name}.npy')
        assert features.shape == (len(img_paths), 256)
        np.save(outfile, features)
    print(f'[FAb-Net] Job done on {device}.')


def get_fabnet(video_paths: list, device_ids: str = 'all', batch_size=32, output_dir: str = 'tmp'):
    threads_eval(job_fabnet, video_paths, device_ids=device_ids, batch_size=batch_size, output_dir=output_dir)


if __name__ == '__main__':

    video_paths = [
        'data/processed/frames/9KAqOrdiZ4I.001',
        'data/processed/frames/h-jMFLm6U_Y.000',
        'data/processed/frames/nEm44UpCKmA.002'
    ]

    get_fabnet(video_paths, output_dir='data/processed/fabnet', batch_size=64, device_ids='0')
    get_fabnet(video_paths, output_dir='data/processed/fabnet', batch_size=64, device_ids='all')

    extractor, classifier = build_fabnet(include_top=True)
    print(f'type of extractor:', type(extractor))
    print(f'type of classifier:', type(classifier))