"""Tests for VisualModelWrapper base class via ClipWrapper."""

import pathlib
import shutil
import tempfile
import unittest

import torch

from exordium.video.deep.clip import ClipWrapper
from tests.fixtures import IMAGE_FACE


class TestVisualModelWrapperBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ClipWrapper(device_id=None)

    def test_dir_to_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                dest = pathlib.Path(tmp) / f"{i:06d}.jpg"
                shutil.copy(IMAGE_FACE, dest)
            img_paths = sorted(pathlib.Path(tmp).glob("*.jpg"))
            result = self.model.dir_to_feature(img_paths, batch_size=2)
            self.assertIsInstance(result, dict)
            self.assertIn("frame_ids", result)
            self.assertIn("features", result)
            feats = result["features"]
            ids = result["frame_ids"]
            self.assertIsInstance(feats, torch.Tensor)
            self.assertEqual(feats.ndim, 2)
            self.assertEqual(len(ids), feats.shape[0])


if __name__ == "__main__":
    unittest.main()
