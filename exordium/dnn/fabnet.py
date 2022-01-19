import os
import sys
import threading
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor

from exordium.shared import get_project_root, get_weight_location
sys.path.append(str(Path(f'{get_project_root()}/tools/FAb-Net/FAb-Net/code').resolve()))
from models_multiview import FrontaliseModelMasks_wider


def get_weights():
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


def get_model(include_top: bool = False):
    weights_path = get_weights()
    inner_nc = 256
    extractor = FrontaliseModelMasks_wider(3, inner_nc=inner_nc, num_additional_ids=32)
    extractor.load_state_dict(torch.load(weights_path)['state_dict_model'])
    extractor.eval()
    num_classes = 8
    classifier = torch.nn.Sequential(torch.nn.BatchNorm1d(inner_nc), torch.nn.Linear(inner_nc, num_classes, bias=False))
    classifier.load_state_dict(torch.load(weights_path)['state_dict'])
    classifier.eval()
    print('FAb-Net model is loaded')
    if include_top:
        return extractor, classifier
    else:
        return extractor


#from exordium.shared import timer
#@timer
def get_fabnet(video_paths: list, output_dir: str, batch_size: int = 32, device_ids: str = 'all'):
    # get gpu devices
    if device_ids == 'all':
        devices = [f'cuda:{id}' for id in range(torch.cuda.device_count())]
    else:
        devices = [f'cuda:{int(id)}' for id in device_ids.split(',')]
    print(f'[FAb-Net] Available devices: {devices}.')

    # parallelize videos over multiple gpus
    def job(device: str, frame_dirs: list, batch_size: int, output_dir: str) -> torch.Tensor:
        print(f'[FAb-Net] Job started on {device}.')
        extractor = get_model(include_top=False)
        extractor.to(device)
        transform = Compose([Resize((256, 256)), ToTensor()])
        for frame_dir in frame_dirs:
            img_paths = [str(Path(frame_dir) / elem.decode("utf-8")) for elem in sorted(os.listdir(frame_dir))]
            imgs = torch.stack([transform(Image.open(img_path).convert('RGB')) for img_path in img_paths]) # (L, C, H, W)
            features = []
            for i in range(0, imgs.shape[0], batch_size):
                samples = imgs[i:i+batch_size]
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
            print(output_dir)
            print(Path(frame_dir).name)
            outfile = str(Path(output_dir) / f'{Path(frame_dir).name}.npy')
            print(outfile)
            np.save(outfile, features)
        print(f'[FAb-Net] Job done on {device}.')

    splits = np.array_split(video_paths, len(devices))
    threads = []
    for i in range(len(devices)):
        threads.append(threading.Thread(target=job, args=(devices[i], splits[i], batch_size, output_dir)))
    for thread in threads: thread.start()
    for thread in threads: thread.join()


if __name__ == '__main__':

    video_paths = [
        'data/processed/frames/9KAqOrdiZ4I.001',
        'data/processed/frames/h-jMFLm6U_Y.000',
        'data/processed/frames/nEm44UpCKmA.002'
    ]

    get_fabnet(video_paths, output_dir='data/processed/fabnet', batch_size=64, device_ids='0')
    get_fabnet(video_paths, output_dir='data/processed/fabnet', batch_size=64, device_ids='all')

    extractor, classifier = get_model(include_top=True)
    print(f'type of extractor:', type(extractor))
    print(f'type of classifier:', type(classifier))