"""Tests for UnigazeWrapper gaze estimator."""

import unittest

import numpy as np
import torch

from exordium.video.face.gaze.unigaze import UnigazeWrapper


class TestUnigazeWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = UnigazeWrapper(device_id=None)

    def test_returns_yaw_pitch_tensors(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        yaw, pitch = self.model(img)
        self.assertIsInstance(yaw, torch.Tensor)
        self.assertEqual(yaw.shape, (1,))

    def test_batch_tensor_input(self):
        imgs = torch.randint(0, 255, (3, 3, 224, 224), dtype=torch.uint8)
        yaw, pitch = self.model(imgs)
        self.assertEqual(yaw.shape, (3,))
        self.assertEqual(pitch.shape, (3,))


if __name__ == "__main__":
    unittest.main()
