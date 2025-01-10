from pathlib import Path
from typing import Sequence
import torch
from PIL import Image
import numpy as np
import open_clip
from decord import VideoReader, cpu
from tqdm import tqdm
from exordium.utils.decorator import load_or_create


class ClipWrapper:

    def __init__(self, model_name: str = 'ViT-H-14-quickgelu', pretrained: str = 'dfn5b', gpu_id: int = 0):
        self.device = torch.device(gpu_id) if gpu_id >= 0 else torch.device('cpu')
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model.eval()
        model.to(self.device)
        self.model = model
        self.preprocess = preprocess
    
    def load_image(self, image_path: str):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0)
        return image # (B, C, H, W)
    
    @load_or_create('npy')
    def extract_from_video(self, video_path: str, batch_size: int = 15, verbose: bool = False, **kwargs):
        vr = VideoReader(str(video_path), ctx=cpu(0))
        features = []
        for batch_ind in tqdm(range(0, len(vr), batch_size), desc='clip', disable=not verbose):
            frame_indices = [ind for ind in range(batch_ind, batch_ind + batch_size) if ind < len(vr)]
            images: np.ndarray = vr.get_batch(frame_indices).asnumpy() # (T, H, W, C)
            frame_list = [images[i] for i in range(images.shape[0])]
            batch_features = self(frame_list)
            batch_features = batch_features.detach().cpu().numpy() # (B, 1024)
            features.append(batch_features)
            del frame_list
        features = np.concatenate(features, axis=0) # (T, 1024)
        assert features.ndim == 2
        assert features.shape[-1] == 1024
        del vr
        return features


    @load_or_create('pkl')
    def dir_to_feature(self, img_paths: list[str], batch_size: int = 30, verbose: bool = False, **kwargs) -> np.ndarray:
        ids, features = [], []
        for index in tqdm(range(0, len(img_paths), batch_size), total=np.ceil(len(img_paths)/batch_size).astype(int), desc='CLIP extraction', disable=not verbose):
            batch_paths = img_paths[index:index+batch_size]
            ids += [int(p.stem) for p in batch_paths]
            feature = self(batch_paths)
            feature = feature.detach().cpu().numpy()
            features.append(feature)
        features = np.concatenate(features, axis=0)
        return ids, features


    def __call__(self, frames: Sequence[str | Path] | Sequence[np.ndarray]) -> np.ndarray:

        if isinstance(frames[0], str | Path):
            samples = [Image.open(str(frame_path)) for frame_path in frames]
        else:
            samples = [Image.fromarray(array) for array in frames] # [0..255]

        samples = torch.stack([self.preprocess(sample) for sample in samples]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(samples)
        
        del samples
        
        return image_features


if __name__ == '__main__':
    image_path = 'data/tmp/10025.jpg'
    model = ClipWrapper()
    import time
    start_time = time.time()
    output = model([image_path, image_path])
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print("output_shape:",output.shape)