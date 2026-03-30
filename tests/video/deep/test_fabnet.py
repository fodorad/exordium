"""Tests for FabNetWrapper visual feature extractor."""

import unittest

import numpy as np
import torch

from exordium.video.deep.fabnet import FabNetWrapper
from tests.fixtures import IMAGE_FACE, hf_file_exists


class TestFabNetWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FabNetWrapper(device_id=None)

    def test_preprocess_shape(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        preprocessed = self.model.preprocess(img)
        self.assertEqual(preprocessed.shape, (1, 3, 256, 256))

    def test_call_from_numpy(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        out = self.model(img)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (1, 256))

    def test_batch_tensor(self):
        out = self.model(torch.randint(0, 255, (4, 3, 256, 256), dtype=torch.uint8))
        self.assertEqual(out.shape, (4, 256))

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[1], 256)

    def test_output_is_float(self):
        out = self.model(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        self.assertEqual(out.dtype, torch.float32)


class TestFabNetWeightAvailability(unittest.TestCase):
    def test_fabnet_weights_file(self):
        self.assertTrue(
            hf_file_exists("fodorad/exordium-weights", "fabnet_weights.pth"),
            "fabnet_weights.pth not found in fodorad/exordium-weights",
        )


if __name__ == "__main__":
    unittest.main()
