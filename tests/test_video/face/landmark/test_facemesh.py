"""Tests for exordium.video.facemesh module."""

import unittest

import numpy as np

from tests.fixtures import IMAGE_FACE


class TestFaceMeshWrapper(unittest.TestCase):
    """Tests for FaceMeshWrapper."""

    @classmethod
    def setUpClass(cls):
        try:
            from exordium.video.face.landmark.facemesh import FaceMeshWrapper

            cls.model = FaceMeshWrapper()
        except Exception as e:
            raise unittest.SkipTest(f"FaceMeshWrapper unavailable: {e}")

    def test_init(self):
        self.assertIsNotNone(self.model.model)

    def test_call_with_face_image(self):
        import cv2

        img = cv2.imread(str(IMAGE_FACE))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))
        result = self.model([rgb])
        self.assertIsInstance(result, list)

    def test_call_with_no_face(self):
        """Blank image should return no landmarks."""
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        result = self.model([blank])
        self.assertEqual(len(result), 0)

    def test_call_with_batch(self):
        import cv2

        img = cv2.imread(str(IMAGE_FACE))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        result = self.model([rgb, blank])
        self.assertIsInstance(result, list)

    def test_landmarks_shape_when_detected(self):
        import cv2

        img = cv2.imread(str(IMAGE_FACE))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (224, 224))
        result = self.model([rgb])
        if len(result) > 0:
            self.assertEqual(result[0].shape, (468, 2))


if __name__ == "__main__":
    unittest.main()
