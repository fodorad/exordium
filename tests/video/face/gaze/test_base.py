"""Tests for gaze geometry utilities, GazeWrapper, and gaze visualization functions."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exordium.video.face.gaze.base import (
    GazeWrapper,
    compute_angular_error,
    convert_draw_vector,
    convert_rotate_draw_vector,
    draw_vector,
    gazeto3d,
    looking_at_camera_xy,
    looking_at_camera_yaw_pitch,
    pitchyaw_to_pixel,
    rotate_vector,
    softmax_temperature,
    vector_to_pitchyaw,
)


class TestGazeGeometryUtils(unittest.TestCase):
    def test_pitchyaw_to_pixel_zero(self):
        dx, dy = pitchyaw_to_pixel(0.0, 0.0)
        self.assertAlmostEqual(dx, 0.0, places=5)
        self.assertAlmostEqual(dy, 0.0, places=5)

    def test_rotate_vector_zero(self):
        x, y = rotate_vector((1.0, 0.0), 0.0)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 0.0)

    def test_rotate_vector_90(self):
        x, y = rotate_vector((1.0, 0.0), 90.0)
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 1.0, places=5)

    def test_vector_to_pitchyaw_shape(self):
        vecs = torch.randn(4, 3)
        out = vector_to_pitchyaw(vecs)
        self.assertEqual(out.shape, (4, 2))

    def test_looking_at_camera_xy_center(self):
        self.assertTrue(looking_at_camera_xy((0.0, 0.0), thr=0.5))

    def test_looking_at_camera_xy_far(self):
        self.assertFalse(looking_at_camera_xy((10.0, 0.0), thr=0.5))

    def test_looking_at_camera_yaw_pitch(self):
        self.assertTrue(looking_at_camera_yaw_pitch(0.0, 0.0, thr=0.5))

    def test_compute_angular_error_zero(self):
        angles = torch.zeros(4, 2)
        err = compute_angular_error(angles, angles)
        self.assertAlmostEqual(err.item(), 0.0, places=4)


class _TypePreservationMixin:
    def make_np(self, h=100, w=100):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def make_tensor(self, h=100, w=100):
        return torch.zeros(3, h, w, dtype=torch.uint8)

    def assertNumpyOut(self, out, h=100, w=100):
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (h, w, 3))

    def assertTensorOut(self, out, h=100, w=100):
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (3, h, w))


class TestDrawVector(_TypePreservationMixin, unittest.TestCase):
    def test_numpy_in_numpy_out(self):
        img = self.make_np()
        out = draw_vector(img, (50, 50), (10, 10))
        self.assertNumpyOut(out)

    def test_tensor_in_tensor_out(self):
        img = self.make_tensor()
        out = draw_vector(img, (50, 50), (10, 10))
        self.assertTensorOut(out)

    def test_does_not_modify_input(self):
        img = self.make_np()
        original = img.copy()
        draw_vector(img, (50, 50), (10, 10))
        np.testing.assert_array_equal(img, original)


class TestConvertDrawVector(_TypePreservationMixin, unittest.TestCase):
    def test_numpy_in_numpy_out(self):
        img = self.make_np()
        out = convert_draw_vector(img, yaw=0.1, pitch=0.05, origin=(50, 50))
        self.assertNumpyOut(out)

    def test_tensor_in_tensor_out(self):
        img = self.make_tensor()
        out = convert_draw_vector(img, yaw=0.1, pitch=0.05, origin=(50, 50))
        self.assertTensorOut(out)

    def test_convert_rotate_draw_vector_numpy(self):
        img = self.make_np()
        out = convert_rotate_draw_vector(
            img, yaw=0.1, pitch=0.0, rotation_degree=15.0, origin=(50, 50), length=30
        )
        self.assertNumpyOut(out)

    def test_convert_rotate_draw_vector_tensor(self):
        img = self.make_tensor()
        out = convert_rotate_draw_vector(
            img, yaw=0.1, pitch=0.0, rotation_degree=15.0, origin=(50, 50), length=30
        )
        self.assertTensorOut(out)


class TestVectorToPitchyaw(unittest.TestCase):
    def test_numpy_input_converts_to_tensor(self):
        """numpy input → line 66: vectors = torch.as_tensor(vectors, dtype=torch.float32)."""
        vecs = np.random.randn(4, 3).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        result = vector_to_pitchyaw(vecs)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (4, 2))

    def test_tensor_input_skips_conversion(self):
        """torch.Tensor input: should skip line 66."""
        t = torch.randn(3, 3)
        t = t / (t.norm(dim=1, keepdim=True) + 1e-8)
        result = vector_to_pitchyaw(t)
        self.assertEqual(result.shape, (3, 2))


class TestGazeto3d(unittest.TestCase):
    def test_wrong_shape_raises_value_error(self):
        """gaze.shape != (2,) → lines 87-88: raise ValueError."""
        bad = np.array([0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            gazeto3d(bad)

    def test_correct_shape_returns_3d_vector(self):
        """gaze.shape == (2,) → lines 89-90: pitch, yaw → 3D vector."""
        gaze = np.array([0.1, 0.2])
        result = gazeto3d(gaze)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (3,))
        norm = np.linalg.norm(result)
        self.assertAlmostEqual(norm, 1.0, places=6)


class TestSoftmaxTemperature(unittest.TestCase):
    def test_output_shape_matches_input(self):
        x = torch.randn(4, 10)
        result = softmax_temperature(x, 0.5)
        self.assertEqual(result.shape, (4, 10))

    def test_probabilities_sum_to_one(self):
        x = torch.randn(3, 8)
        result = softmax_temperature(x, 1.0)
        sums = result.sum(dim=1)
        for s in sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)


class TestGazeWrapperLookingAtCamera(unittest.TestCase):
    def test_tensor_input_returns_tensor(self):
        yaw = torch.tensor([0.0, 0.1, 5.0])
        pitch = torch.tensor([0.0, 0.05, 3.0])
        lac = GazeWrapper.looking_at_camera(yaw, pitch, thr=0.3)
        self.assertIsInstance(lac, torch.Tensor)
        self.assertEqual(lac.shape, (3,))
        self.assertTrue(lac[0].item())
        self.assertFalse(lac[2].item())

    def test_numpy_input_converts_and_returns_tensor(self):
        yaw = np.array([0.0, 5.0])
        pitch = np.array([0.0, 3.0])
        lac = GazeWrapper.looking_at_camera(yaw, pitch, thr=0.3)
        self.assertIsInstance(lac, torch.Tensor)
        self.assertTrue(lac[0].item())
        self.assertFalse(lac[1].item())


class TestGazeWrapperVisualize(unittest.TestCase):
    def _make_face(self, size=64):
        return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)

    def test_visualize_with_output_path_saves_file(self):
        """output_path is not None → lines 532-534: mkdir + imwrite."""
        face = self._make_face()
        yaw = np.array([0.1])
        pitch = np.array([0.05])
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "sub" / "gaze_vis.jpg"
            results = GazeWrapper.visualize([face], yaw, pitch, output_path=out)
            self.assertTrue(out.exists())
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)

    def test_visualize_tensor_input_returns_tensors(self):
        """Tensor (B, 3, H, W) input → output list of CHW tensors."""
        face_t = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
        yaw = torch.tensor([0.0, 0.1])
        pitch = torch.tensor([0.0, 0.05])
        results = GazeWrapper.visualize(face_t, yaw, pitch)
        self.assertIsInstance(results[0], torch.Tensor)
        self.assertEqual(results[0].shape[0], 3)

    def test_visualize_with_roll_angles(self):
        """roll_angles provided → non-zero rotation applied to gaze arrow."""
        face = self._make_face()
        yaw = np.array([0.2])
        pitch = np.array([0.1])
        results = GazeWrapper.visualize([face], yaw, pitch, roll_angles=[15.0])
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
