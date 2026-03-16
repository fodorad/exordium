"""Tests for exordium.video.face.transform (face-specific transform functions)."""

import unittest

import cv2
import numpy as np

from exordium.video.face.transform import align_face, crop_eye_keep_ratio, rotate_face
from tests.fixtures import IMAGE_FACE


class TestRotateFace(unittest.TestCase):
    """Tests for rotate_face function."""

    def test_rotate_face_basic(self):
        """Test basic face rotation."""
        face = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        rotated_face, rotation_matrix = rotate_face(face, 45)

        self.assertEqual(rotated_face.shape, face.shape, "Shape should be preserved")
        self.assertEqual(rotation_matrix.shape, (2, 3), "Rotation matrix should be 2x3")

    def test_rotate_face_zero_rotation(self):
        """Test face rotation with 0 degrees."""
        face = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        rotated_face, rotation_matrix = rotate_face(face, 0)

        self.assertEqual(rotated_face.shape, face.shape, "Shape should be preserved")
        # Image should be nearly identical for 0 rotation
        np.testing.assert_array_almost_equal(rotated_face, face, decimal=0)

    def test_rotate_face_returns_tuple(self):
        """Test that rotate_face returns a (image, matrix) tuple."""
        face = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
        result = rotate_face(face, 30)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_rotate_face_output_dtype(self):
        """Test that output image preserves dtype."""
        face = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        rotated_face, _ = rotate_face(face, 15)
        self.assertEqual(rotated_face.dtype, np.uint8)


class TestAlignFace(unittest.TestCase):
    """Tests for align_face function."""

    def test_align_face_invalid_landmarks_raises(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        bb_xyxy = np.array([0, 0, 100, 100])
        bad_landmarks = np.array([[10, 10], [20, 10]])  # wrong shape
        with self.assertRaises(Exception):
            align_face(image, bb_xyxy, bad_landmarks)


class TestCropEyeKeepRatio(unittest.TestCase):
    """Tests for crop_eye_keep_ratio function."""

    def test_crop_eye_keep_ratio(self):
        img = cv2.imread(str(IMAGE_FACE))
        landmarks = np.array(
            [
                [100, 150],
                [110, 145],
                [120, 145],
                [130, 150],
                [120, 158],
                [110, 158],
            ]
        )
        result = crop_eye_keep_ratio(img, landmarks)
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[0], 36)
        self.assertEqual(result.shape[1], 60)


if __name__ == "__main__":
    unittest.main()
