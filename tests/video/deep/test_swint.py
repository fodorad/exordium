"""Tests for SwinTWrapper visual feature extractor."""

import unittest

import numpy as np
import torch

from exordium.video.deep.swint import SwinTWrapper
from tests.fixtures import IMAGE_FACE, hf_file_exists


class TestSwinTWrapperInvalidArch(unittest.TestCase):
    def test_invalid_arch_raises(self):
        with self.assertRaises(ValueError):
            SwinTWrapper(arch="xlarge", device_id=None)


class TestSwinTWrapperTiny(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SwinTWrapper(arch="tiny", pretrained=True, device_id=None)

    def test_preprocess_shape(self):
        preprocessed = self.model.preprocess(
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        )
        self.assertEqual(preprocessed.shape, (1, 3, 224, 224))

    def test_call_from_numpy(self):
        out = self.model(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[0], 1)

    def test_batch_tensor(self):
        out = self.model(torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8))
        self.assertEqual(out.shape[0], 4)

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertEqual(out.ndim, 2)

    def test_feature_dim_positive(self):
        out = self.model(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        self.assertGreater(out.shape[1], 0)


class TestSwinTWeightAvailability(unittest.TestCase):
    def test_swint_tiny_weights_file(self):
        self.assertTrue(
            hf_file_exists("fodorad/exordium-weights", "swin_tiny_patch4_window7_224.pth"),
            "swin_tiny_patch4_window7_224.pth not found in fodorad/exordium-weights",
        )


if __name__ == "__main__":
    unittest.main()
