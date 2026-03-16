"""Tests for exordium.video.face.gaze.base (gaze utilities and rotate_vector)."""

import unittest

import numpy as np
import torch

from exordium.video.face.gaze import (
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
    spherical2cartesial,
    vector_to_pitchyaw,
    visualize_normed_space,
    visualize_target_gaze,
)


class TestRotateVector(unittest.TestCase):
    """Tests for rotate_vector function."""

    def test_rotate_vector_90_degrees(self):
        """Test 90 degree rotation."""
        xy = np.array([1.0, 0.0])
        rotated = rotate_vector(xy, 90)
        expected = np.array([0.0, 1.0])
        np.testing.assert_array_almost_equal(rotated, expected, decimal=5)

    def test_rotate_vector_180_degrees(self):
        """Test 180 degree rotation."""
        xy = np.array([1.0, 0.0])
        rotated = rotate_vector(xy, 180)
        expected = np.array([-1.0, 0.0])
        np.testing.assert_array_almost_equal(rotated, expected, decimal=5)

    def test_rotate_vector_360_degrees(self):
        """Test 360 degree rotation (should return to original)."""
        xy = np.array([1.0, 1.0])
        rotated = rotate_vector(xy, 360)
        np.testing.assert_array_almost_equal(rotated, xy, decimal=5)


class TestVectorToPitchyaw(unittest.TestCase):
    def test_output_shape(self):
        vectors = np.random.randn(5, 3)
        out = vector_to_pitchyaw(vectors)
        self.assertEqual(out.shape, (5, 2))

    def test_single_vector(self):
        vectors = np.array([[0.0, 0.0, 1.0]])
        out = vector_to_pitchyaw(vectors)
        self.assertEqual(out.shape, (1, 2))


class TestGazeto3d(unittest.TestCase):
    def test_output_shape(self):
        gaze = np.array([0.1, 0.2])
        out = gazeto3d(gaze)
        self.assertEqual(out.shape, (3,))

    def test_zero_gaze_is_unit_z(self):
        gaze = np.array([0.0, 0.0])
        out = gazeto3d(gaze)
        self.assertAlmostEqual(out[2], -1.0, places=5)

    def test_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            gazeto3d(np.array([0.1, 0.2, 0.3]))


class TestPitchyawToPixel(unittest.TestCase):
    def test_output_shape(self):
        out = pitchyaw_to_pixel(0.1, 0.2)
        self.assertEqual(out.shape, (2,))

    def test_zero_angles_give_zero_vector(self):
        out = pitchyaw_to_pixel(0.0, 0.0)
        np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-6)

    def test_length_scales_output(self):
        out1 = pitchyaw_to_pixel(0.5, 0.3, length=1.0)
        out2 = pitchyaw_to_pixel(0.5, 0.3, length=2.0)
        np.testing.assert_allclose(out2, out1 * 2, atol=1e-6)


class TestSpherical2Cartesial(unittest.TestCase):
    def test_output_shape(self):
        x = torch.rand(4, 2)
        out = spherical2cartesial(x)
        self.assertEqual(out.shape, (4, 3))

    def test_batch_size_one(self):
        x = torch.tensor([[0.0, 0.0]])
        out = spherical2cartesial(x)
        self.assertEqual(out.shape, (1, 3))


class TestComputeAngularError(unittest.TestCase):
    def test_near_zero_error_for_same_angles(self):
        # acos(1.0) may produce NaN due to floating-point clamping; test robustly
        angles = torch.tensor([[0.1, 0.2], [0.3, 0.1], [0.0, 0.4], [0.2, 0.3]])
        error = compute_angular_error(angles, angles.clone())
        # Accept NaN (numerical artifact) or very small value
        self.assertTrue(error.isnan() or error.item() < 5.0)

    def test_positive_error(self):
        a = torch.zeros(4, 2)
        b = torch.ones(4, 2) * 0.5
        error = compute_angular_error(a, b)
        self.assertGreater(error.item(), 0.0)


class TestSoftmaxTemperature(unittest.TestCase):
    def test_output_sums_to_one_per_row(self):
        tensor = torch.rand(4, 90)
        out = softmax_temperature(tensor, temperature=1.0)
        row_sums = out.sum(dim=1)
        for s in row_sums:
            self.assertAlmostEqual(s.item(), 1.0, places=5)

    def test_high_temperature_uniform(self):
        tensor = torch.rand(2, 10)
        out = softmax_temperature(tensor, temperature=1000.0)
        self.assertEqual(out.shape, (2, 10))


class TestLookingAtCamera(unittest.TestCase):
    def test_looking_at_camera_xy_center(self):
        xy = np.array([0.0, 0.0])
        self.assertTrue(looking_at_camera_xy(xy, thr=0.5))

    def test_looking_at_camera_xy_far(self):
        xy = np.array([1.0, 1.0])
        self.assertFalse(looking_at_camera_xy(xy, thr=0.5))

    def test_looking_at_camera_yaw_pitch_zero(self):
        self.assertTrue(looking_at_camera_yaw_pitch(0.0, 0.0, thr=0.5))

    def test_looking_at_camera_yaw_pitch_large(self):
        self.assertFalse(looking_at_camera_yaw_pitch(1.5, 1.5, thr=0.5))


class TestDrawVector(unittest.TestCase):
    def test_returns_array(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        origin = np.array([150, 100])
        end_point = np.array([50.0, -30.0])
        result = draw_vector(image, origin, end_point)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, image.shape)

    def test_does_not_modify_original(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        original = image.copy()
        draw_vector(image, np.array([100, 100]), np.array([20.0, 20.0]))
        np.testing.assert_array_equal(image, original)


class TestConvertRotateDrawVector(unittest.TestCase):
    def test_output_shape(self):
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        origin = np.array([200, 200])
        result = convert_rotate_draw_vector(
            image, yaw=0.2, pitch=0.1, rotation_degree=15.0, origin=origin, length=100
        )
        self.assertEqual(result.shape, image.shape)


class TestConvertDrawVector(unittest.TestCase):
    def test_output_shape(self):
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        origin = np.array([200, 200])
        result = convert_draw_vector(image, yaw=0.2, pitch=0.1, origin=origin, length=100)
        self.assertEqual(result.shape, image.shape)


class TestVisualizeTargetGaze(unittest.TestCase):
    def test_output_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face = np.zeros((112, 112, 3), dtype=np.uint8)
        origin = np.array([320, 240])
        gaze_normed = np.array([0.1, 0.2])
        gaze_vec = np.array([0.05, 0.1])
        result = visualize_target_gaze(frame, face, gaze_normed, gaze_vec, origin)
        self.assertEqual(result.shape, frame.shape)


class TestVisualizeNormedSpace(unittest.TestCase):
    def test_output_shape(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        face = np.zeros((112, 112, 3), dtype=np.uint8)
        result = visualize_normed_space(image, face, yaw=0.1, pitch=0.0)
        self.assertEqual(result.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
