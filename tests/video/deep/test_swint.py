"""Tests for exordium.video.deep.swint.SwinTWrapper."""

import unittest

import numpy as np
import torch

from exordium.video.deep.swint import SwinTWrapper
from tests.fixtures import IMAGE_FACE


class TestSwinTWrapperTiny(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SwinTWrapper(arch="tiny", pretrained=True, device_id=None)

    def test_preprocess_shape(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        preprocessed = self.model.preprocess(img)
        self.assertEqual(preprocessed.shape, (1, 3, 224, 224))

    def test_call_from_numpy(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = self.model(img)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[0], 1)

    def test_batch_tensor(self):
        imgs = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
        out = self.model(imgs)
        self.assertEqual(out.shape[0], 4)

    def test_feature_dim_positive(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = self.model(img)
        self.assertGreater(out.shape[1], 0)

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 2)

    def test_batch_consistent_feature_dim(self):
        img1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        out1 = self.model(img1)
        out2 = self.model(img2)
        self.assertEqual(out1.shape[1], out2.shape[1])


class TestSwinTWrapperSmall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.model = SwinTWrapper(arch="small", pretrained=True, device_id=None)
        except Exception as e:
            raise unittest.SkipTest(f"Swin-small weights not available: {e}") from e

    def test_call_from_image(self):
        out = self.model(IMAGE_FACE)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape[0], 1)
        self.assertGreater(out.shape[1], 0)


class TestSwinTWrapperInvalidArch(unittest.TestCase):
    def test_invalid_arch_raises(self):
        with self.assertRaises(ValueError):
            SwinTWrapper(arch="xlarge", device_id=None)


if __name__ == "__main__":
    unittest.main()
