"""Tests for ClipWrapper visual feature extractor."""

import unittest

import numpy as np
import torch

from exordium.video.deep.clip import ClipWrapper
from tests.fixtures import IMAGE_EMMA, IMAGE_FACE, VIDEO_MULTISPEAKER_SHORT


class TestClipWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ClipWrapper(device_id=None)

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[0], 1)
        self.assertGreater(out.shape[1], 0)

    def test_from_numpy(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = self.model(img)
        self.assertEqual(out.shape[0], 1)

    def test_from_tensor(self):
        t = torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8)
        out = self.model(t)
        self.assertEqual(out.shape[0], 1)

    def test_batch_tensor(self):
        imgs = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
        out = self.model(imgs)
        self.assertEqual(out.shape[0], 4)

    def test_batch_paths(self):
        out = self.model([IMAGE_FACE, IMAGE_FACE])
        self.assertEqual(out.shape[0], 2)

    def test_feature_dim_consistent(self):
        out1 = self.model(IMAGE_FACE)
        out2 = self.model(IMAGE_EMMA)
        self.assertEqual(out1.shape[1], out2.shape[1])

    def test_video_to_feature(self):
        import pathlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = pathlib.Path(tmp_dir) / "clip_features.st"
            result = self.model.video_to_feature(
                VIDEO_MULTISPEAKER_SHORT,
                batch_size=8,
                output_path=out_path,
            )
            self.assertIsInstance(result, dict)
            feats = result["features"]
            ids = result["frame_ids"]
            self.assertIsInstance(feats, torch.Tensor)
            self.assertEqual(feats.ndim, 2)
            self.assertGreater(feats.shape[0], 0)
            self.assertEqual(len(ids), feats.shape[0])


if __name__ == "__main__":
    unittest.main()
