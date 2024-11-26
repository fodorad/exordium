from pathlib import Path
from typing import Sequence
import torch
from PIL import Image
import numpy as np
import open_clip


class ClipWrapper:

    def __init__(self, model_name: str = 'ViT-H-14-quickgelu', pretrained: str = 'dfn5b', gpu_id: int | None = 0):
        self.device = torch.device(gpu_id) if gpu_id is not None else torch.device('cpu')
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model.eval()
        model.to(self.device)
        self.model = model
        self.preprocess = preprocess
    
    def load_image(self, image_path: str):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0)
        return image # (B, C, H, W)

    def __call__(self, frames: Sequence[str] | Sequence[np.ndarray]) -> np.ndarray:

        if isinstance(frames[0], str):
            samples = [Image.open(frame_path) for frame_path in frames]
        else:
            samples = [Image.fromarray(array) for array in frames] # [0..255]
        samples = torch.stack([self.preprocess(sample) for sample in samples]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(samples)
        
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