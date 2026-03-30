"""Tests for ClipWrapper visual feature extractor."""

import pathlib
import shutil
import tempfile
import unittest

import numpy as np
import torch

from exordium.video.deep.clip import _CLIP_MEAN, _CLIP_STD, _DEFAULT_CLIP_MODEL, ClipWrapper
from tests.fixtures import IMAGE_EMMA, IMAGE_FACE, VIDEO_MULTISPEAKER_SHORT, hf_repo_exists


class TestClipConstants(unittest.TestCase):
    def test_default_model_is_string(self):
        self.assertIsInstance(_DEFAULT_CLIP_MODEL, str)
        self.assertGreater(len(_DEFAULT_CLIP_MODEL), 0)

    def test_clip_mean_has_three_channels(self):
        self.assertEqual(len(_CLIP_MEAN), 3)

    def test_clip_std_has_three_channels(self):
        self.assertEqual(len(_CLIP_STD), 3)

    def test_clip_mean_values_in_range(self):
        for v in _CLIP_MEAN:
            self.assertGreater(v, 0.0)
            self.assertLess(v, 1.0)

    def test_clip_std_values_in_range(self):
        for v in _CLIP_STD:
            self.assertGreater(v, 0.0)
            self.assertLess(v, 1.0)


class TestClipWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ClipWrapper(device_id=None)

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[0], 1)

    def test_from_numpy(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = self.model(img)
        self.assertEqual(out.shape[0], 1)

    def test_from_tensor(self):
        out = self.model(torch.randint(0, 255, (3, 224, 224), dtype=torch.uint8))
        self.assertEqual(out.shape[0], 1)

    def test_batch_tensor(self):
        out = self.model(torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8))
        self.assertEqual(out.shape[0], 4)

    def test_feature_dim_consistent(self):
        self.assertEqual(self.model(IMAGE_FACE).shape[1], self.model(IMAGE_EMMA).shape[1])

    def test_dir_to_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                shutil.copy(IMAGE_FACE, pathlib.Path(tmp) / f"{i:06d}.jpg")
            result = self.model.dir_to_feature(
                sorted(pathlib.Path(tmp).glob("*.jpg")), batch_size=2
            )
            self.assertIn("features", result)
            self.assertEqual(result["features"].ndim, 2)

    def test_video_to_feature(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self.model.video_to_feature(
                VIDEO_MULTISPEAKER_SHORT,
                batch_size=8,
                output_path=pathlib.Path(tmp_dir) / "clip.st",
            )
            self.assertIsInstance(result["features"], torch.Tensor)
            self.assertEqual(result["features"].ndim, 2)


class TestClipWeightAvailability(unittest.TestCase):
    def test_clip_repo(self):
        self.assertTrue(hf_repo_exists("laion/CLIP-ViT-H-14-laion2B-s32B-b79K"))


if __name__ == "__main__":
    unittest.main()
