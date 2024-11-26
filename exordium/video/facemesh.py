from typing import Sequence
import numpy as np
import cv2
import mediapipe as mp
from exordium.video.io import batch_iterator
from exordium.video.detection import Track
from exordium.utils.decorator import load_or_create


class FaceMeshWrapper():

    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.model = mp_face_mesh.FaceMesh()

    def __call__(self, rgb_images: Sequence[np.ndarray]) -> np.ndarray:
        batch = []
        for rgb_image in rgb_images:
            height, width, _ = rgb_image.shape
            result = self.model.process(rgb_image)
            if not result.multi_face_landmarks:
                continue
            face = result.multi_face_landmarks[0] # first face
            landmarks = []
            for i in range(0, 468):
                pt = face.landmark[i]
                x = int(pt.x * width)
                y = int(pt.y * height)
                landmarks.append((x,y))
            batch.append(np.array(landmarks))
        return batch

    @load_or_create('pkl')
    def track_to_feature(self, track: Track, batch_size: int = 30, **kwargs) -> tuple[list, np.ndarray]:
        ids, features = [], []
        for subset in batch_iterator(track, batch_size):
            ids += [detection.frame_id for detection in subset if not detection.is_interpolated]
            samples = [detection.bb_crop() for detection in subset if not detection.is_interpolated] # (B, H, W, C)
            #for sample_index, (sample_id, sample) in enumerate(zip([detection.frame_id for detection in subset if not detection.is_interpolated], samples)):
            #    cv2.imwrite(f'{sample_id}.png', sample)
            #    assert sample.ndim == 3, f'invalid image with shape {sample.shape}'
            feature = self(samples)
            features.append(feature)
        features = np.concatenate(features, axis=0)
        return ids, features


if __name__ == "__main__":
    img = cv2.imread('data/tmp/0_Anger2.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model = FaceMeshWrapper()
    face_landmarks = model(img_rgb)
    print('shape:', face_landmarks.shape)