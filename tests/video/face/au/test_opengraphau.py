"""Tests for OpenGraphAU action unit extractor."""

import unittest

import numpy as np
import torch

from exordium.video.face.au.opengraphau import OpenGraphAuWrapper
from tests.fixtures import IMAGE_FACE


class TestOpenGraphAuWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.model = OpenGraphAuWrapper(device_id=None)
        except Exception as e:
            raise unittest.SkipTest(f"OpenGraphAU weights not available: {e}") from e

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 2)

    def test_from_numpy(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = self.model(img)
        self.assertIsInstance(out, torch.Tensor)

    def test_output_shape(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = self.model(img)
        self.assertEqual(out.shape[0], 1)
        self.assertGreater(out.shape[1], 0)

    def test_batch_tensor(self):
        imgs = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
        out = self.model(imgs)
        self.assertEqual(out.shape[0], 4)


if __name__ == "__main__":
    unittest.main()
