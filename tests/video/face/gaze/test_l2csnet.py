"""Tests for L2csNetWrapper gaze estimator."""

import unittest

import numpy as np
import torch

from exordium.video.face.gaze.l2csnet import L2csNetWrapper
from tests.fixtures import IMAGE_FACE, hf_file_exists


class TestL2csNetWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = L2csNetWrapper(device_id=None)

    def test_returns_yaw_pitch_tensors(self):
        yaw, pitch = self.model(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        self.assertIsInstance(yaw, torch.Tensor)
        self.assertEqual(yaw.shape, (1,))
        self.assertEqual(pitch.shape, (1,))

    def test_batch_input(self):
        yaw, pitch = self.model(torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8))
        self.assertEqual(yaw.shape, (4,))
        self.assertEqual(pitch.shape, (4,))

    def test_from_image_path(self):
        yaw, pitch = self.model(IMAGE_FACE)
        self.assertEqual(yaw.shape[0], 1)


class TestL2csNetWeightAvailability(unittest.TestCase):
    def test_l2csnet_weights_file(self):
        self.assertTrue(
            hf_file_exists("fodorad/exordium-weights", "l2csnet_weights.pkl"),
            "l2csnet_weights.pkl not found in fodorad/exordium-weights",
        )


if __name__ == "__main__":
    unittest.main()
