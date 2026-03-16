"""Tests for exordium.video.clip module."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.fixtures import IMAGE_CAT_TIE, IMAGE_FACE, VIDEO_MULTISPEAKER_SHORT


class TestClipWrapper(unittest.TestCase):
    """Tests for ClipWrapper."""

    @classmethod
    def setUpClass(cls):
        from exordium.video.deep.clip import ClipWrapper

        cls.model = ClipWrapper(device_id=None)

    def test_init(self):
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.preprocess)

    def test_call_with_file_paths(self):
        features = self.model.predict([str(IMAGE_FACE), str(IMAGE_CAT_TIE)])
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[1], 1024)

    def test_call_with_numpy_arrays(self):
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(3)]
        features = self.model.predict(frames)
        self.assertEqual(features.shape[0], 3)
        self.assertEqual(features.shape[1], 1024)

    def test_call_single_image(self):
        features = self.model.predict([str(IMAGE_FACE)])
        self.assertEqual(features.shape, (1, 1024))

    def test_video_to_feature(self):
        ids, features = self.model.video_to_feature(str(VIDEO_MULTISPEAKER_SHORT))
        self.assertIsInstance(ids, list)
        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[1], 1024)
        self.assertEqual(len(ids), features.shape[0])

    def test_dir_to_feature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # dir_to_feature expects stems to be integer frame IDs
            p0 = Path(tmpdir) / "0.jpg"
            p1 = Path(tmpdir) / "1.jpg"
            shutil.copy(str(IMAGE_FACE), str(p0))
            shutil.copy(str(IMAGE_CAT_TIE), str(p1))
            ids, features = self.model.dir_to_feature([p0, p1])
        self.assertEqual(len(ids), 2)
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[1], 1024)


if __name__ == "__main__":
    unittest.main()
