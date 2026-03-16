"""Tests for exordium.video.face.au.opengraphau module."""

import unittest

import numpy as np


class TestOpenGraphAuWrapper(unittest.TestCase):
    """Tests for OpenGraphAuWrapper (requires model weights)."""

    @classmethod
    def setUpClass(cls):
        try:
            from exordium.video.face.au.opengraphau import AU_REGISTRY, OpenGraphAuWrapper

            cls.AU_REGISTRY = AU_REGISTRY
            cls.model = OpenGraphAuWrapper(device_id=None)
        except Exception as e:
            raise unittest.SkipTest(f"OpenGraphAuWrapper unavailable: {e}")

    def _make_face(self, h=224, w=224):
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    def test_au_registry_not_empty(self):
        self.assertGreater(len(self.AU_REGISTRY), 0)

    def test_au_registry_entries_are_tuples(self):
        for entry in self.AU_REGISTRY:
            self.assertIsInstance(entry, tuple)
            self.assertEqual(len(entry), 2)

    def test_model_loaded(self):
        self.assertIsNotNone(self.model)

    def test_predict_single_face_shape(self):
        face = self._make_face()
        result = self.model.predict([face])
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[0], 1)

    def test_predict_batch_shape(self):
        faces = [self._make_face() for _ in range(3)]
        result = self.model.predict(faces)
        self.assertEqual(result.shape[0], 3)

    def test_predict_output_range(self):
        face = self._make_face()
        result = self.model.predict([face])
        self.assertTrue(np.all(result >= 0.0), "AU intensities should be non-negative")

    def test_predict_dtype(self):
        face = self._make_face()
        result = self.model.predict([face])
        self.assertTrue(np.issubdtype(result.dtype, np.floating))


if __name__ == "__main__":
    unittest.main()
