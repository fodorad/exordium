"""Tests for EmotiEffNetWrapper visual feature extractor."""

import pathlib
import tempfile
import unittest

import numpy as np
import torch

from exordium.video.deep.emotieffnet import (
    _DEFAULT_MODEL,
    _MODELS,
    EmotiEffNetWrapper,
)
from tests.fixtures import IMAGE_EMMA, IMAGE_FACE, VIDEO_MULTISPEAKER_SHORT, head_ok


class TestEmotiEffNetWrapperInit(unittest.TestCase):
    def test_invalid_model_name_raises(self):
        with self.assertRaises(ValueError):
            EmotiEffNetWrapper(model_name="nonexistent_model")

    def test_module_constants_consistent(self):
        self.assertIn(_DEFAULT_MODEL, _MODELS)
        for cfg in _MODELS.values():
            self.assertIn("img_size", cfg)
            self.assertIn("feature_dim", cfg)

    def test_b0_models_have_224_input(self):
        for name, cfg in _MODELS.items():
            if "_b0_" in name:
                self.assertEqual(cfg["img_size"], 224)

    def test_b2_models_have_260_input(self):
        for name, cfg in _MODELS.items():
            if "_b2_" in name:
                self.assertEqual(cfg["img_size"], 260)

    def test_b0_feature_dim_is_1280(self):
        for name, cfg in _MODELS.items():
            if "_b0_" in name:
                self.assertEqual(cfg["feature_dim"], 1280)

    def test_b2_feature_dim_is_1408(self):
        for name, cfg in _MODELS.items():
            if "_b2_" in name:
                self.assertEqual(cfg["feature_dim"], 1408)


class TestEmotiEffNetWrapperB0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = EmotiEffNetWrapper(model_name="enet_b0_8_best_vgaf", device_id=None)

    def test_feature_dim_attribute(self):
        self.assertEqual(self.model.feature_dim, 1280)

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertEqual(out.shape, (1, 1280))

    def test_from_numpy_hwc(self):
        out = self.model(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        self.assertEqual(out.shape, (1, 1280))

    def test_from_batch_tensor(self):
        out = self.model(torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8))
        self.assertEqual(out.shape, (4, 1280))

    def test_preprocess_output_shape(self):
        preprocessed = self.model.preprocess(
            torch.randint(0, 255, (3, 3, 64, 64), dtype=torch.uint8)
        )
        self.assertEqual(preprocessed.shape, (3, 3, 224, 224))

    def test_feature_dim_consistent(self):
        self.assertEqual(self.model(IMAGE_FACE).shape[1], self.model(IMAGE_EMMA).shape[1])

    def test_video_to_feature(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self.model.video_to_feature(
                VIDEO_MULTISPEAKER_SHORT,
                batch_size=8,
                output_path=pathlib.Path(tmp_dir) / "emotieffnet.st",
            )
            self.assertEqual(result["features"].shape[1], 1280)


class TestEmotiEffNetWeightAvailability(unittest.TestCase):
    def test_emotieffnet_b0_vgaf_url(self):
        url = "https://github.com/sb-ai-lab/EmotiEffLib/blob/main/models/affectnet_emotions/enet_b0_8_best_vgaf.pt?raw=true"
        self.assertTrue(head_ok(url), f"Not reachable: {url}")


if __name__ == "__main__":
    unittest.main()
