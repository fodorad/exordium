"""Tests for exordium.video.face.headpose module.

Tests merged from test_video/test_headpose.py and test_utils/test_headpose.py.
"""

import unittest

import numpy as np

from exordium.video.face.headpose import SixDRepNetWrapper, draw_headpose_axis
from tests.fixtures import IMAGE_FACE


class TestSixDRepNetWrapper(unittest.TestCase):
    """Tests for SixDRepNetWrapper."""

    @classmethod
    def setUpClass(cls):
        cls.model = SixDRepNetWrapper(device_id=None)  # CPU
        cls.face_rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    def test_predict_single_returns_dict(self):
        result = self.model.predict_single(self.face_rgb)
        self.assertIsInstance(result, dict)
        self.assertIn("headpose", result)

    def test_predict_single_shape(self):
        result = self.model.predict_single(self.face_rgb)
        self.assertEqual(result["headpose"].shape, (3,))

    def test_predict_single_dtype(self):
        result = self.model.predict_single(self.face_rgb)
        self.assertEqual(result["headpose"].dtype, np.float32)

    def test_predict_single_finite(self):
        result = self.model.predict_single(self.face_rgb)
        self.assertTrue(np.all(np.isfinite(result["headpose"])))

    def test_call_batch_shape(self):
        faces = [self.face_rgb, self.face_rgb]
        result = self.model(faces)
        self.assertEqual(result.shape, (2, 3))

    def test_call_batch_dtype(self):
        result = self.model([self.face_rgb])
        self.assertEqual(result.dtype, np.float32)

    def test_predict_single_real_face(self):
        from exordium.video.core.io import image_to_np

        face_rgb = image_to_np(IMAGE_FACE, "RGB")
        result = self.model.predict_single(face_rgb)
        self.assertEqual(result["headpose"].shape, (3,))
        self.assertTrue(np.all(np.isfinite(result["headpose"])))

    def test_predict_single_deterministic(self):
        result1 = self.model.predict_single(self.face_rgb)
        result2 = self.model.predict_single(self.face_rgb)
        np.testing.assert_array_equal(result1["headpose"], result2["headpose"])


class TestDrawHeadposeAxis(unittest.TestCase):
    """Tests for draw_headpose_axis."""

    def setUp(self):
        self.image = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_returns_numpy_array(self):
        result = draw_headpose_axis(self.image, (0.0, 0.0, 0.0))
        self.assertIsInstance(result, np.ndarray)

    def test_output_shape_unchanged(self):
        result = draw_headpose_axis(self.image, (0.0, 0.0, 0.0))
        self.assertEqual(result.shape, self.image.shape)

    def test_zero_angles(self):
        """Zero yaw/pitch/roll should draw axes along axes of image."""
        result = draw_headpose_axis(self.image, (0.0, 0.0, 0.0))
        self.assertIsInstance(result, np.ndarray)

    def test_nonzero_angles(self):
        result = draw_headpose_axis(self.image, (30.0, -15.0, 10.0))
        self.assertEqual(result.shape, self.image.shape)

    def test_custom_origin(self):
        result = draw_headpose_axis(self.image, (10.0, 5.0, 0.0), tdx=300, tdy=200)
        self.assertEqual(result.shape, self.image.shape)

    def test_custom_size(self):
        result = draw_headpose_axis(self.image, (0.0, 0.0, 0.0), size=50)
        self.assertEqual(result.shape, self.image.shape)

    def test_ndarray_headpose_input(self):
        headpose = np.array([20.0, -10.0, 5.0])
        result = draw_headpose_axis(self.image, headpose)
        self.assertEqual(result.shape, self.image.shape)

    def test_extreme_angles(self):
        result = draw_headpose_axis(self.image, (90.0, 90.0, 90.0))
        self.assertEqual(result.shape, self.image.shape)

    def test_negative_angles(self):
        result = draw_headpose_axis(self.image, (-45.0, -30.0, -20.0))
        self.assertEqual(result.shape, self.image.shape)

    def test_default_origin_uses_image_center(self):
        """When tdx/tdy are None, the origin should default to image center."""
        result = draw_headpose_axis(self.image, (0.0, 0.0, 0.0), tdx=None, tdy=None)
        self.assertEqual(result.shape, self.image.shape)


if __name__ == "__main__":
    unittest.main()
