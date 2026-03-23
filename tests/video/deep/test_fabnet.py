"""Tests for exordium.video.deep.fabnet.FabNetWrapper."""

import unittest

import numpy as np
import torch

from exordium.video.deep.fabnet import FabNetWrapper
from tests.fixtures import IMAGE_FACE


class TestFabNetWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FabNetWrapper(device_id=None)

    def test_preprocess_shape(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        preprocessed = self.model.preprocess(img)
        self.assertEqual(preprocessed.shape, (1, 3, 256, 256))

    def test_inference_single(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        preprocessed = self.model.preprocess(img)
        out = self.model(preprocessed)
        self.assertEqual(out.shape, (1, 256))

    def test_call_from_numpy(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        out = self.model(img)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (1, 256))

    def test_batch_tensor(self):
        imgs = torch.randint(0, 255, (4, 3, 256, 256), dtype=torch.uint8)
        out = self.model(imgs)
        self.assertEqual(out.shape, (4, 256))

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[1], 256)

    def test_output_is_float(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out = self.model(img)
        self.assertEqual(out.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
