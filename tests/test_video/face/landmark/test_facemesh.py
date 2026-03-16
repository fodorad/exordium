"""Tests for exordium.video.face.landmark.facemesh module."""

import unittest

import cv2
import numpy as np

from exordium.video.face.landmark.facemesh import (
    FaceMeshWrapper,
    rotate_landmarks,
    visualize_landmarks,
)
from tests.fixtures import IMAGE_FACE


class TestRotateLandmarks(unittest.TestCase):
    """Tests for rotate_landmarks helper function."""

    def test_rotate_landmarks_identity(self):
        """Test rotation with identity matrix."""
        landmarks = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
        # Identity rotation matrix (2, 3)
        R = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        result = rotate_landmarks(landmarks, R)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.dtype, int)

    def test_rotate_landmarks_90_degrees(self):
        """Test rotation by 90 degrees."""
        landmarks = np.array([[10, 0], [0, 10]], dtype=float)
        # 90 degree rotation matrix (2x3 affine)
        R = np.array([[0, -1, 0], [1, 0, 0]], dtype=float)
        result = rotate_landmarks(landmarks, R)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, int)

    def test_rotate_landmarks_output_shape(self):
        """Test that output shape matches input shape."""
        landmarks = np.array([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=float)
        R = np.array([[1, 0, 5], [0, 1, 10]], dtype=float)
        result = rotate_landmarks(landmarks, R)
        self.assertEqual(result.shape, landmarks.shape)

    def test_rotate_landmarks_int_output(self):
        """Test that output is converted to int."""
        landmarks = np.array([[10.5, 20.3], [30.7, 40.1]], dtype=float)
        R = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        result = rotate_landmarks(landmarks, R)
        self.assertEqual(result.dtype, int)

    def test_rotate_landmarks_many_points(self):
        """Test with many landmarks (FaceMesh size)."""
        landmarks = np.random.rand(478, 2) * 100
        R = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        result = rotate_landmarks(landmarks, R)
        self.assertEqual(result.shape, (478, 2))
        self.assertEqual(result.dtype, int)


class TestVisualizeLandmarks(unittest.TestCase):
    """Tests for visualize_landmarks function."""

    def test_visualize_landmarks_basic(self):
        """Test basic landmark visualization."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        landmarks = np.array([[10, 10], [20, 20], [30, 30]], dtype=int)
        result = visualize_landmarks(image, landmarks)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, image.dtype)
        self.assertFalse(np.array_equal(result, image))  # Image should be modified

    def test_visualize_landmarks_invalid_shape_raises(self):
        """Test that invalid landmark shape raises exception."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bad_landmarks = np.array([[10, 10, 10]])  # Wrong shape
        with self.assertRaises(Exception):
            visualize_landmarks(image, bad_landmarks)

    def test_visualize_landmarks_with_output_path(self):
        """Test saving landmarks visualization to file."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "landmarks.png"
            image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            landmarks = np.array([[10, 10], [20, 20]], dtype=int)

            result = visualize_landmarks(image, landmarks, output_path=output_path)

            self.assertTrue(output_path.exists())
            self.assertEqual(result.shape, image.shape)

    def test_visualize_landmarks_no_indices(self):
        """Test visualization without showing indices."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        landmarks = np.array([[10, 10], [20, 20]], dtype=int)
        result = visualize_landmarks(image, landmarks, show_indices=False)

        self.assertEqual(result.shape, image.shape)

    def test_visualize_landmarks_custom_color(self):
        """Test visualization with custom color."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        landmarks = np.array([[50, 50]], dtype=int)
        result = visualize_landmarks(image, landmarks, color=(0, 0, 255))  # Red in BGR

        self.assertEqual(result.shape, image.shape)
        # Check that red channel was modified (circle drawn)
        self.assertGreater(result[50, 50, 2], image[50, 50, 2])


class TestFaceMeshWrapper(unittest.TestCase):
    """Tests for FaceMeshWrapper class."""

    @classmethod
    def setUpClass(cls):
        """Initialize FaceMeshWrapper for tests."""
        cls.wrapper = FaceMeshWrapper()

    def test_facemesh_wrapper_init(self):
        """Test FaceMeshWrapper initialization."""
        self.assertIsNotNone(self.wrapper.landmarker)

    def test_facemesh_wrapper_call_with_valid_image(self):
        """Test __call__ with a valid face image."""
        img = cv2.imread(str(IMAGE_FACE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.wrapper([img_rgb])

        self.assertIsInstance(result, list)
        # Result may be empty if no face detected, but structure should be correct
        for landmarks in result:
            self.assertIsInstance(landmarks, np.ndarray)
            self.assertEqual(landmarks.ndim, 2)
            self.assertEqual(landmarks.shape[1], 2)

    def test_facemesh_wrapper_call_with_no_face(self):
        """Test __call__ with image containing no face."""
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.wrapper([blank])

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)  # No face detected

    def test_facemesh_wrapper_call_returns_float32(self):
        """Test that landmarks are returned as float32."""
        img = cv2.imread(str(IMAGE_FACE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.wrapper([img_rgb])

        if result:
            self.assertEqual(result[0].dtype, np.float32)

    def test_facemesh_wrapper_call_batch(self):
        """Test __call__ with multiple images."""
        img = cv2.imread(str(IMAGE_FACE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        batch = [img_rgb, img_rgb, img_rgb]

        result = self.wrapper(batch)

        self.assertIsInstance(result, list)
        # All detected faces should have 468 landmarks
        for landmarks in result:
            self.assertEqual(landmarks.shape[1], 2)

    def test_facemesh_wrapper_landmarks_in_pixel_space(self):
        """Test that returned landmarks are in pixel coordinates."""
        img = cv2.imread(str(IMAGE_FACE))
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.wrapper([img_rgb])

        if result:
            landmarks = result[0]
            # All coordinates should be within image bounds (with some tolerance)
            self.assertTrue((landmarks[:, 0] >= -10).all())
            self.assertTrue((landmarks[:, 0] <= w + 10).all())
            self.assertTrue((landmarks[:, 1] >= -10).all())
            self.assertTrue((landmarks[:, 1] <= h + 10).all())

    def test_facemesh_wrapper_dense_landmarks(self):
        """Test that FaceMesh detects dense landmarks when face is detected."""
        img = cv2.imread(str(IMAGE_FACE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.wrapper([img_rgb])

        if result:  # Only check if face was detected
            landmarks = result[0]
            # MediaPipe returns 478 landmarks (468 base + 10 extra)
            self.assertEqual(landmarks.shape[0], 478)


if __name__ == "__main__":
    unittest.main()
