"""Tests for L2csNetWrapper gaze estimator."""

import unittest

import numpy as np
import torch

from exordium.video.face.gaze.base import GazeWrapper
from exordium.video.face.gaze.l2csnet import L2csNetWrapper
from tests.fixtures import IMAGE_FACE


class TestL2csNetWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = L2csNetWrapper(device_id=None)

    def test_returns_yaw_pitch_tensors(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        yaw, pitch = self.model(img)
        self.assertIsInstance(yaw, torch.Tensor)
        self.assertIsInstance(pitch, torch.Tensor)
        self.assertEqual(yaw.shape, (1,))
        self.assertEqual(pitch.shape, (1,))

    def test_batch_input(self):
        imgs = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
        yaw, pitch = self.model(imgs)
        self.assertEqual(yaw.shape, (4,))
        self.assertEqual(pitch.shape, (4,))

    def test_from_image_path(self):
        yaw, pitch = self.model(IMAGE_FACE)
        self.assertEqual(yaw.shape[0], 1)

    def test_looking_at_camera_output_type_tensor(self):
        yaw = torch.tensor([0.0, 0.1, 5.0])
        pitch = torch.tensor([0.0, 0.05, 3.0])
        lac = GazeWrapper.looking_at_camera(yaw, pitch, thr=0.3)
        self.assertIsInstance(lac, torch.Tensor)
        self.assertEqual(lac.shape, (3,))
        self.assertTrue(lac[0].item())
        self.assertFalse(lac[2].item())

    def test_looking_at_camera_output_type_numpy_input(self):
        """numpy input is converted to tensor; result is always torch.Tensor."""
        yaw = np.array([0.0, 5.0])
        pitch = np.array([0.0, 3.0])
        lac = GazeWrapper.looking_at_camera(yaw, pitch, thr=0.3)
        self.assertIsInstance(lac, torch.Tensor)
        self.assertTrue(lac[0].item())
        self.assertFalse(lac[1].item())

    def test_visualize_returns_list(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        yaw = torch.tensor([0.1])
        pitch = torch.tensor([0.05])
        out = GazeWrapper.visualize([img], yaw, pitch)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)

    def test_visualize_tensor_in_tensor_out(self):
        imgs = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
        yaw = torch.tensor([0.1, 0.2])
        pitch = torch.tensor([0.0, 0.0])
        out = GazeWrapper.visualize(imgs, yaw, pitch)
        self.assertEqual(len(out), 2)
        self.assertIsInstance(out[0], torch.Tensor)
        self.assertEqual(out[0].shape, (3, 64, 64))

    def test_visualize_numpy_in_numpy_out(self):
        imgs = [np.zeros((64, 64, 3), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.uint8)]
        yaw = np.array([0.1, 0.2])
        pitch = np.array([0.0, 0.0])
        out = GazeWrapper.visualize(imgs, yaw, pitch)
        self.assertIsInstance(out[0], np.ndarray)


if __name__ == "__main__":
    unittest.main()
