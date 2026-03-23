"""Tests for SixDRepNetWrapper and headpose visualization helpers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exordium.video.face.headpose import draw_headpose_axis, draw_headpose_cube
from tests.fixtures import IMAGE_FACE


class TestDrawHeadposeAxis(unittest.TestCase):
    """Type-preservation and branch coverage for draw_headpose_axis."""

    def setUp(self):
        self.hp = np.array([10.0, 5.0, -3.0])
        self.np_img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.t_img = torch.zeros(3, 100, 100, dtype=torch.uint8)

    def test_numpy_in_numpy_out(self):
        out = draw_headpose_axis(self.np_img, self.hp)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (100, 100, 3))

    def test_tensor_in_tensor_out(self):
        out = draw_headpose_axis(self.t_img, self.hp)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (3, 100, 100))

    def test_tensor_headpose(self):
        hp_t = torch.tensor([10.0, 5.0, -3.0])
        out = draw_headpose_axis(self.np_img, hp_t)
        self.assertIsInstance(out, np.ndarray)

    def test_tuple_headpose(self):
        out = draw_headpose_axis(self.np_img, (10.0, 5.0, -3.0))
        self.assertIsInstance(out, np.ndarray)

    def test_explicit_origin(self):
        out = draw_headpose_axis(self.np_img, self.hp, tdx=50, tdy=50)
        self.assertEqual(out.shape, (100, 100, 3))

    def test_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "axis.png"
            draw_headpose_axis(self.np_img, self.hp, output_path=p)
            self.assertTrue(p.exists())


class TestDrawHeadposeCube(unittest.TestCase):
    """Type-preservation and branch coverage for draw_headpose_cube."""

    def setUp(self):
        self.hp = np.array([10.0, 5.0, -3.0])
        self.np_img = np.zeros((100, 100, 3), dtype=np.uint8)
        self.t_img = torch.zeros(3, 100, 100, dtype=torch.uint8)

    def test_numpy_in_numpy_out(self):
        out = draw_headpose_cube(self.np_img, self.hp)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (100, 100, 3))

    def test_tensor_in_tensor_out(self):
        out = draw_headpose_cube(self.t_img, self.hp)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (3, 100, 100))

    def test_tensor_headpose(self):
        hp_t = torch.tensor([10.0, 5.0, -3.0])
        out = draw_headpose_cube(self.np_img, hp_t)
        self.assertIsInstance(out, np.ndarray)

    def test_tuple_headpose(self):
        out = draw_headpose_cube(self.np_img, (10.0, 5.0, -3.0))
        self.assertIsInstance(out, np.ndarray)

    def test_explicit_origin(self):
        out = draw_headpose_cube(self.np_img, self.hp, tdx=50, tdy=50)
        self.assertEqual(out.shape, (100, 100, 3))

    def test_output_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "cube.png"
            draw_headpose_cube(self.np_img, self.hp, output_path=p)
            self.assertTrue(p.exists())


class TestSixDRepNetWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.video.face.headpose import SixDRepNetWrapper

        cls.model = SixDRepNetWrapper(device_id=None)

    def test_single_face_numpy(self):
        face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = self.model(face)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (1, 3))

    def test_single_face_tensor(self):
        face = torch.randint(0, 255, (3, 112, 112), dtype=torch.uint8)
        out = self.model(face)
        self.assertEqual(out.shape, (1, 3))

    def test_batch_tensor(self):
        faces = torch.randint(0, 255, (4, 3, 112, 112), dtype=torch.uint8)
        out = self.model(faces)
        self.assertEqual(out.shape, (4, 3))

    def test_list_of_variable_size_crops(self):
        crops = [
            np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8),
            np.random.randint(0, 255, (120, 100, 3), dtype=np.uint8),
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
        ]
        out = self.model(crops)
        self.assertEqual(out.shape, (3, 3))

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertEqual(out.shape[1], 3)
        self.assertTrue(out.abs().max().item() < 180.0)

    def test_output_in_degree_range(self):
        face = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        out = self.model(face)
        yaw, pitch, roll = out[0]
        for angle in (yaw, pitch, roll):
            self.assertLess(abs(angle.item()), 180.0)


if __name__ == "__main__":
    unittest.main()
