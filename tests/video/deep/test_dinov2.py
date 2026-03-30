"""Tests for DINOv2Wrapper visual feature extractor."""

import pathlib
import tempfile
import unittest

import numpy as np
import torch

from exordium.video.deep.dinov2 import (
    _DEFAULT_DINOV2_MODEL,
    _FEATURE_DIMS,
    _MODEL_IDS,
    DINOv2Wrapper,
)
from tests.fixtures import IMAGE_EMMA, IMAGE_FACE, VIDEO_MULTISPEAKER_SHORT, hf_repo_exists


class TestDINOv2WrapperInit(unittest.TestCase):
    def test_invalid_model_name_raises(self):
        with self.assertRaises(ValueError):
            DINOv2Wrapper(model_name="xlarge")

    def test_module_constants_consistent(self):
        self.assertEqual(set(_MODEL_IDS.keys()), set(_FEATURE_DIMS.keys()))
        self.assertIn(_DEFAULT_DINOV2_MODEL, _MODEL_IDS)


class TestDINOv2Wrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DINOv2Wrapper(model_name="base", device_id=None)

    def test_feature_dim_attribute(self):
        self.assertEqual(self.model.feature_dim, _FEATURE_DIMS["base"])

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertEqual(out.shape, (1, _FEATURE_DIMS["base"]))

    def test_from_numpy_hwc(self):
        out = self.model(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        self.assertEqual(out.shape, (1, _FEATURE_DIMS["base"]))

    def test_from_batch_tensor(self):
        out = self.model(torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8))
        self.assertEqual(out.shape, (4, _FEATURE_DIMS["base"]))

    def test_from_list_of_paths(self):
        out = self.model([IMAGE_FACE, IMAGE_FACE])
        self.assertEqual(out.shape, (2, _FEATURE_DIMS["base"]))

    def test_preprocess_output_shape(self):
        preprocessed = self.model.preprocess(
            torch.randint(0, 255, (3, 3, 64, 64), dtype=torch.uint8)
        )
        self.assertEqual(preprocessed.shape, (3, 3, 224, 224))

    def test_output_is_l2_normalised(self):
        out = self.model(torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8))
        norms = out.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_feature_dim_consistent(self):
        self.assertEqual(self.model(IMAGE_FACE).shape[1], self.model(IMAGE_EMMA).shape[1])

    def test_video_to_feature(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self.model.video_to_feature(
                VIDEO_MULTISPEAKER_SHORT,
                batch_size=8,
                output_path=pathlib.Path(tmp_dir) / "dino.st",
            )
            self.assertEqual(result["features"].shape[1], _FEATURE_DIMS["base"])


class TestDINOv2WeightAvailability(unittest.TestCase):
    def test_dinov2_base_repo(self):
        self.assertTrue(hf_repo_exists("facebook/dinov2-base"))


if __name__ == "__main__":
    unittest.main()
