"""Tests for SixDRepNet headpose estimator and visualization helpers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exordium.video.face.headpose import draw_headpose_axis, draw_headpose_cube
from tests.fixtures import IMAGE_FACE, head_ok


class TestDrawHeadposeAxis(unittest.TestCase):
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
        out = draw_headpose_axis(self.np_img, torch.tensor([10.0, 5.0, -3.0]))
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
        out = draw_headpose_cube(self.np_img, torch.tensor([10.0, 5.0, -3.0]))
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
        out = self.model(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (1, 3))

    def test_batch_tensor(self):
        out = self.model(torch.randint(0, 255, (4, 3, 112, 112), dtype=torch.uint8))
        self.assertEqual(out.shape, (4, 3))

    def test_list_of_variable_size_crops(self):
        crops = [
            np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8),
            np.random.randint(0, 255, (120, 100, 3), dtype=np.uint8),
        ]
        out = self.model(crops)
        self.assertEqual(out.shape, (2, 3))

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertEqual(out.shape[1], 3)

    def test_output_in_degree_range(self):
        out = self.model(torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8))
        for angle in out[0]:
            self.assertLess(abs(angle.item()), 180.0)


class TestSixDRepNetWeightAvailability(unittest.TestCase):
    def test_sixdrepnet_url(self):
        url = "https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth"
        self.assertTrue(head_ok(url), f"Not reachable: {url}")


if __name__ == "__main__":
    unittest.main()
