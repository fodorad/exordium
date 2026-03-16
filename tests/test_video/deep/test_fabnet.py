import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from exordium.video.core.detection import DetectionFactory, Track
from exordium.video.deep.fabnet import FabNetWrapper
from tests.fixtures import IMAGE_FACE


def _build_face_track(image_path: Path, num_frames: int = 10) -> Track:
    """Build a synthetic Track from a single face image."""
    frame = np.array(Image.open(image_path).convert("RGB"))
    h, w = frame.shape[:2]
    bb = np.array([0, 0, w, h])
    lmks = np.array(
        [
            [w // 4, h // 4],
            [3 * w // 4, h // 4],
            [w // 2, h // 2],
            [w // 4, 3 * h // 4],
            [3 * w // 4, 3 * h // 4],
        ]
    )
    track = Track(track_id=0)
    for fid in range(num_frames):
        det = DetectionFactory.create_detection(
            frame_id=fid,
            source=frame,
            score=0.9,
            bb_xywh=bb.copy(),
            landmarks=lmks.copy(),
        )
        track.add(det)
    return track


class FabNetTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.TMP_DIR = Path(tempfile.mkdtemp())
        cls.model = FabNetWrapper()
        cls.face_track = _build_face_track(IMAGE_FACE, num_frames=10)

    @classmethod
    def tearDownClass(cls):
        if cls.TMP_DIR.exists():
            shutil.rmtree(cls.TMP_DIR)

    def test_model_loaded(self):
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.device)

    def test_call_face_image(self):
        feature = self.model.predict([IMAGE_FACE])
        self.assertIsInstance(feature, np.ndarray)
        self.assertEqual(feature.shape, (1, 256))

    def test_call_face_image_inference(self):
        img = torchvision.io.read_image(str(IMAGE_FACE)).float() / 255.0  # (C, H, W) in [0, 1]
        img = torch.unsqueeze(img, dim=0).repeat(5, 1, 1, 1)  # (1, C, H, W)
        transform = torchvision.transforms.Resize((256, 256))
        img = transform(img)  # (B, C, H, W)
        feature = self.model.inference(img)
        feature = feature.detach().cpu().numpy()
        self.assertIsInstance(feature, np.ndarray)
        self.assertEqual(feature.shape, (5, 256))

    def test_call_single_face(self):
        face = self.face_track[0].bb_crop()
        feature = self.model.predict([face])
        self.assertIsInstance(feature, np.ndarray)
        self.assertEqual(feature.shape, (1, 256))

    def test_call_batch(self):
        faces = [det.bb_crop() for det in self.face_track[:8]]
        feature = self.model.predict(faces)
        self.assertIsInstance(feature, np.ndarray)
        self.assertEqual(feature.shape, (len(faces), 256))

    def test_call_with_wide_crop(self):
        faces = [det.bb_crop_wide() for det in self.face_track[:4]]
        feature = self.model.predict(faces)
        self.assertEqual(feature.shape, (4, 256))

    def test_feature_dtype(self):
        faces = [self.face_track[0].bb_crop()]
        feature = self.model.predict(faces)
        self.assertTrue(np.issubdtype(feature.dtype, np.floating))

    def test_track_to_feature(self):
        ids, feature = self.model.track_to_feature(self.face_track)
        self.assertIsInstance(ids, list)
        self.assertIsInstance(feature, np.ndarray)
        self.assertEqual(feature.shape, (len(ids), 256))
        self.assertGreater(len(ids), 0)

    def test_track_to_feature_with_output(self):
        output_path = self.TMP_DIR / "fabnet_features.pkl"
        ids, feature = self.model.track_to_feature(self.face_track, output_path=output_path)
        self.assertTrue(output_path.exists())
        self.assertEqual(feature.shape, (len(ids), 256))

    def test_deterministic_output(self):
        faces = [self.face_track[0].bb_crop()]
        feature1 = self.model.predict(faces)
        feature2 = self.model.predict(faces)
        np.testing.assert_array_almost_equal(feature1, feature2)

    def test_dir_to_feature(self):
        face_image_bgr = cv2.resize(cv2.imread(str(IMAGE_FACE)), (256, 256))
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "imgs"
            img_dir.mkdir()
            for i in range(3):
                cv2.imwrite(str(img_dir / f"{i}.jpg"), face_image_bgr)
            ids, features = self.model.dir_to_feature(sorted(img_dir.glob("*.jpg")))
        self.assertEqual(len(ids), 3)
        self.assertEqual(features.shape, (3, 256))


if __name__ == "__main__":
    unittest.main()
