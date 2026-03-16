"""Tests for exordium.video.deep.base.VisualModelWrapper."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exordium.video.deep.base import VisualModelWrapper
from tests.fixtures import IMAGE_FACE, VIDEO_MULTISPEAKER_SHORT


class MockWrapper(VisualModelWrapper):
    """Minimal concrete subclass for testing VisualModelWrapper."""

    def __init__(self, feat_dim: int = 64, device_id=None):
        super().__init__(device_id)
        self.feat_dim = feat_dim

    def _preprocess(self, frames):
        tensors = []
        for f in frames:
            t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False
            ).squeeze(0)
            tensors.append(t)
        return torch.stack(tensors).to(self.device)

    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        b = tensor.shape[0]
        return torch.zeros(b, self.feat_dim, device=self.device)


class TestVisualModelWrapperPredict(unittest.TestCase):
    """Tests for predict() with various input types."""

    @classmethod
    def setUpClass(cls):
        cls.model = MockWrapper(feat_dim=64, device_id=None)

    def test_predict_single_ndarray(self):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = self.model.predict(img)
        self.assertEqual(result.shape, (1, 64))
        self.assertEqual(result.dtype, np.float32)

    def test_predict_list_of_ndarrays(self):
        imgs = [np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8) for _ in range(3)]
        result = self.model.predict(imgs)
        self.assertEqual(result.shape, (3, 64))

    def test_predict_single_path_string(self):
        result = self.model.predict(str(IMAGE_FACE))
        self.assertEqual(result.shape, (1, 64))

    def test_predict_single_path_object(self):
        result = self.model.predict(IMAGE_FACE)
        self.assertEqual(result.shape, (1, 64))

    def test_predict_list_of_paths(self):
        result = self.model.predict([IMAGE_FACE, IMAGE_FACE])
        self.assertEqual(result.shape, (2, 64))


class TestVisualModelWrapperCall(unittest.TestCase):
    """Tests for __call__() with tensor inputs."""

    @classmethod
    def setUpClass(cls):
        cls.model = MockWrapper(feat_dim=64, device_id=None)

    def test_call_3d_tensor_auto_unsqueeze(self):
        tensor = torch.rand(3, 32, 32)
        result = self.model(tensor)
        self.assertEqual(result.shape, (1, 64))

    def test_call_4d_tensor_batched(self):
        tensor = torch.rand(4, 3, 32, 32)
        result = self.model(tensor)
        self.assertEqual(result.shape, (4, 64))

    def test_call_returns_torch_tensor(self):
        tensor = torch.rand(2, 3, 32, 32)
        result = self.model(tensor)
        self.assertIsInstance(result, torch.Tensor)


class TestVisualModelWrapperDirToFeature(unittest.TestCase):
    """Tests for dir_to_feature()."""

    @classmethod
    def setUpClass(cls):
        cls.model = MockWrapper(feat_dim=64, device_id=None)
        cls.tmp_dir = Path(tempfile.mkdtemp())
        import shutil

        for i in range(3):
            shutil.copy(str(IMAGE_FACE), str(cls.tmp_dir / f"{i}.jpg"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_dir_to_feature_returns_tuple(self):
        paths = sorted(self.tmp_dir.glob("*.jpg"))
        ids, features = self.model.dir_to_feature(paths)
        self.assertIsInstance(ids, list)
        self.assertIsInstance(features, np.ndarray)

    def test_dir_to_feature_shapes(self):
        paths = sorted(self.tmp_dir.glob("*.jpg"))
        ids, features = self.model.dir_to_feature(paths)
        self.assertEqual(len(ids), 3)
        self.assertEqual(features.shape, (3, 64))

    def test_dir_to_feature_ids_are_integers(self):
        paths = sorted(self.tmp_dir.glob("*.jpg"))
        ids, _ = self.model.dir_to_feature(paths)
        for i in ids:
            self.assertIsInstance(i, int)


class TestVisualModelWrapperVideoToFeature(unittest.TestCase):
    """Tests for video_to_feature()."""

    @classmethod
    def setUpClass(cls):
        cls.model = MockWrapper(feat_dim=64, device_id=None)

    def test_video_to_feature_returns_tuple(self):
        ids, features = self.model.video_to_feature(VIDEO_MULTISPEAKER_SHORT)
        self.assertIsInstance(ids, list)
        self.assertIsInstance(features, np.ndarray)

    def test_video_to_feature_shapes(self):
        ids, features = self.model.video_to_feature(VIDEO_MULTISPEAKER_SHORT)
        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[1], 64)
        self.assertEqual(len(ids), features.shape[0])

    def test_video_to_feature_ids_are_sequential(self):
        ids, _ = self.model.video_to_feature(VIDEO_MULTISPEAKER_SHORT)
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids, list(range(len(ids))))


if __name__ == "__main__":
    unittest.main()
