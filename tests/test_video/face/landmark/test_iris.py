"""Tests for exordium.video.iris module (pure utility functions)."""

import unittest

import numpy as np

from exordium.video.face.landmark.iris import (
    FaceMeshLandmarks,
    IrisLandmarks,
    TddfaLandmarks,
    calculate_eye_aspect_ratio,
    calculate_eyelid_pupil_distances,
    calculate_iris_diameters,
)


class TestCalculateIrisDiameters(unittest.TestCase):
    def _make_iris(self, center, right, top, left, bottom):
        lm = np.zeros((5, 2))
        lm[IrisLandmarks.CENTER.value] = center
        lm[IrisLandmarks.RIGHT.value] = right
        lm[IrisLandmarks.TOP.value] = top
        lm[IrisLandmarks.LEFT.value] = left
        lm[IrisLandmarks.BOTTOM.value] = bottom
        return lm

    def test_output_shape(self):
        iris = self._make_iris([32, 32], [40, 32], [32, 25], [24, 32], [32, 39])
        out = calculate_iris_diameters(iris)
        self.assertEqual(out.shape, (2,))

    def test_horizontal_distance(self):
        # left at x=10, right at x=20 — horizontal distance == 10
        iris = self._make_iris([15, 15], [20, 15], [15, 10], [10, 15], [15, 20])
        out = calculate_iris_diameters(iris)
        self.assertAlmostEqual(out[0], 10.0, places=5)

    def test_vertical_distance(self):
        # top at y=5, bottom at y=15 — vertical distance == 10
        iris = self._make_iris([15, 10], [20, 10], [15, 5], [10, 10], [15, 15])
        out = calculate_iris_diameters(iris)
        self.assertAlmostEqual(out[1], 10.0, places=5)

    def test_zero_diameters(self):
        iris = np.zeros((5, 2))
        out = calculate_iris_diameters(iris)
        np.testing.assert_allclose(out, [0.0, 0.0])

    def test_asymmetric_iris(self):
        iris = self._make_iris([0, 0], [3, 0], [0, 4], [-3, 0], [0, -4])
        out = calculate_iris_diameters(iris)
        self.assertAlmostEqual(out[0], 6.0, places=5)  # left to right
        self.assertAlmostEqual(out[1], 8.0, places=5)  # top to bottom


class TestCalculateEyelidPupilDistances(unittest.TestCase):
    def _make_iris_centered(self, cx, cy):
        lm = np.zeros((5, 2))
        lm[IrisLandmarks.CENTER.value] = [cx, cy]
        return lm

    def _make_eye_landmarks_16(self, top_y, bottom_y, cx):
        lm = np.zeros((16, 2))
        lm[FaceMeshLandmarks.TOP] = [cx, top_y]
        lm[FaceMeshLandmarks.BOTTOM] = [cx, bottom_y]
        return lm

    def test_output_shape(self):
        iris = self._make_iris_centered(32, 32)
        eye = self._make_eye_landmarks_16(top_y=20, bottom_y=44, cx=32)
        out = calculate_eyelid_pupil_distances(iris, eye)
        self.assertEqual(out.shape, (2,))

    def test_equal_distances(self):
        iris = self._make_iris_centered(32, 32)
        eye = self._make_eye_landmarks_16(top_y=22, bottom_y=42, cx=32)
        out = calculate_eyelid_pupil_distances(iris, eye)
        self.assertAlmostEqual(out[0], out[1], places=5)

    def test_top_distance(self):
        iris = self._make_iris_centered(32, 32)
        eye = self._make_eye_landmarks_16(top_y=22, bottom_y=42, cx=32)
        out = calculate_eyelid_pupil_distances(iris, eye)
        self.assertAlmostEqual(out[0], 10.0, places=5)

    def test_bottom_distance(self):
        iris = self._make_iris_centered(32, 32)
        eye = self._make_eye_landmarks_16(top_y=22, bottom_y=42, cx=32)
        out = calculate_eyelid_pupil_distances(iris, eye)
        self.assertAlmostEqual(out[1], 10.0, places=5)

    def test_zero_center(self):
        iris = self._make_iris_centered(0, 0)
        eye = self._make_eye_landmarks_16(top_y=0, bottom_y=0, cx=0)
        out = calculate_eyelid_pupil_distances(iris, eye)
        np.testing.assert_allclose(out, [0.0, 0.0])


class TestCalculateEyeAspectRatio(unittest.TestCase):
    def _make_tddfa_landmarks(self, left, top_l, top_r, right, bot_r, bot_l):
        lm = np.zeros((6, 2))
        lm[TddfaLandmarks.LEFT.value] = left
        lm[TddfaLandmarks.TOP_LEFT.value] = top_l
        lm[TddfaLandmarks.TOP_RIGHT.value] = top_r
        lm[TddfaLandmarks.RIGHT.value] = right
        lm[TddfaLandmarks.BOTTOM_RIGHT.value] = bot_r
        lm[TddfaLandmarks.BOTTOM_LEFT.value] = bot_l
        return lm

    def test_tddfa_output_is_float(self):
        lm = self._make_tddfa_landmarks(
            left=[0, 5], top_l=[2, 8], top_r=[4, 8], right=[6, 5], bot_r=[4, 2], bot_l=[2, 2]
        )
        ear = calculate_eye_aspect_ratio(lm)
        self.assertIsInstance(ear, float)

    def test_tddfa_open_eye(self):
        # Wide open eye: lr = 6, each tb = 6, EAR = (6+6)/(2*6) = 1.0
        lm = self._make_tddfa_landmarks(
            left=[0, 5], top_l=[2, 8], top_r=[4, 8], right=[6, 5], bot_r=[4, 2], bot_l=[2, 2]
        )
        ear = calculate_eye_aspect_ratio(lm)
        self.assertGreater(ear, 0.0)

    def test_tddfa_closed_eye(self):
        # Top and bottom at same y: EAR approaches 0
        lm = self._make_tddfa_landmarks(
            left=[0, 5], top_l=[2, 5], top_r=[4, 5], right=[6, 5], bot_r=[4, 5], bot_l=[2, 5]
        )
        ear = calculate_eye_aspect_ratio(lm)
        self.assertAlmostEqual(ear, 0.0, places=5)

    def test_facemesh_16_output_is_float(self):
        lm = np.zeros((16, 2))
        lm[FaceMeshLandmarks.LEFT] = [0, 5]
        lm[FaceMeshLandmarks.RIGHT] = [10, 5]
        lm[FaceMeshLandmarks.TOP] = [5, 8]
        lm[FaceMeshLandmarks.BOTTOM] = [5, 2]
        lm[FaceMeshLandmarks.TOP_LEFT] = [3, 7]
        lm[FaceMeshLandmarks.TOP_RIGHT] = [7, 7]
        lm[FaceMeshLandmarks.BOTTOM_LEFT] = [3, 3]
        lm[FaceMeshLandmarks.BOTTOM_RIGHT] = [7, 3]
        ear = calculate_eye_aspect_ratio(lm)
        self.assertIsInstance(ear, float)
        self.assertGreater(ear, 0.0)

    def test_facemesh_71_treated_as_facemesh(self):
        lm = np.zeros((71, 2))
        lm[FaceMeshLandmarks.LEFT] = [0, 5]
        lm[FaceMeshLandmarks.RIGHT] = [10, 5]
        lm[FaceMeshLandmarks.TOP] = [5, 8]
        lm[FaceMeshLandmarks.BOTTOM] = [5, 2]
        lm[FaceMeshLandmarks.TOP_LEFT] = [3, 7]
        lm[FaceMeshLandmarks.TOP_RIGHT] = [7, 7]
        lm[FaceMeshLandmarks.BOTTOM_LEFT] = [3, 3]
        lm[FaceMeshLandmarks.BOTTOM_RIGHT] = [7, 3]
        ear = calculate_eye_aspect_ratio(lm)
        self.assertIsInstance(ear, float)
        self.assertGreater(ear, 0.0)

    def test_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            calculate_eye_aspect_ratio(np.zeros((4, 2)))

    def test_invalid_shape_3d_raises(self):
        with self.assertRaises(ValueError):
            calculate_eye_aspect_ratio(np.zeros((6, 3)))

    def test_facemesh_closed_eye(self):
        lm = np.zeros((16, 2))
        lm[FaceMeshLandmarks.LEFT] = [0, 5]
        lm[FaceMeshLandmarks.RIGHT] = [10, 5]
        # All vertical pairs at same y => EAR = 0
        lm[FaceMeshLandmarks.TOP] = [5, 5]
        lm[FaceMeshLandmarks.BOTTOM] = [5, 5]
        lm[FaceMeshLandmarks.TOP_LEFT] = [3, 5]
        lm[FaceMeshLandmarks.TOP_RIGHT] = [7, 5]
        lm[FaceMeshLandmarks.BOTTOM_LEFT] = [3, 5]
        lm[FaceMeshLandmarks.BOTTOM_RIGHT] = [7, 5]
        ear = calculate_eye_aspect_ratio(lm)
        self.assertAlmostEqual(ear, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()


class TestIrisWrapper(unittest.TestCase):
    """Tests for IrisWrapper (requires cached weights)."""

    @classmethod
    def setUpClass(cls):
        try:
            from exordium.video.face.landmark.iris import IrisWrapper

            cls.model = IrisWrapper(device_id=-1)
        except Exception as e:
            raise unittest.SkipTest(f"IrisWrapper unavailable: {e}")

    def test_call_returns_eye_and_iris_landmarks(self):
        eye = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        eye_lm, iris_lm = self.model([eye])
        self.assertEqual(eye_lm.shape, (71, 2))
        self.assertEqual(iris_lm.shape, (5, 2))

    def test_call_batch(self):
        eyes = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        eye_lm, iris_lm = self.model(eyes)
        self.assertEqual(eye_lm.shape, (3, 71, 2))
        self.assertEqual(iris_lm.shape, (3, 5, 2))

    def test_eye_to_features_returns_dict(self):
        eye = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        features = self.model.eye_to_features(eye)
        self.assertIn("eye", features)
        self.assertIn("landmarks", features)
        self.assertIn("iris_landmarks", features)
        self.assertIn("iris_diameters", features)
        self.assertIn("eyelid_pupil_distances", features)
        self.assertIn("ear", features)
        self.assertEqual(features["eye"].shape, (64, 64, 3))
        self.assertEqual(features["landmarks"].shape, (71, 2))
        self.assertEqual(features["iris_landmarks"].shape, (5, 2))
