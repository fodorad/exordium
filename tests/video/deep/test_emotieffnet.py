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
from tests.fixtures import IMAGE_EMMA, IMAGE_FACE, VIDEO_MULTISPEAKER_SHORT


class TestEmotiEffNetWrapperInit(unittest.TestCase):
    """Initialisation and configuration tests (no model download required)."""

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
    """Inference tests for the default B0 variant."""

    @classmethod
    def setUpClass(cls):
        cls.model = EmotiEffNetWrapper(model_name="enet_b0_8_best_vgaf", device_id=None)

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    def test_feature_dim_attribute(self):
        self.assertEqual(self.model.feature_dim, 1280)

    def test_img_size_attribute(self):
        self.assertEqual(self.model.img_size, 224)

    # ------------------------------------------------------------------
    # Input types
    # ------------------------------------------------------------------

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 1280)

    def test_from_numpy_hwc(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        out = self.model(img)
        self.assertEqual(out.shape, (1, 1280))

    def test_from_tensor_chw(self):
        t = torch.randint(0, 255, (3, 112, 112), dtype=torch.uint8)
        out = self.model(t)
        self.assertEqual(out.shape, (1, 1280))

    def test_from_batch_tensor(self):
        imgs = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
        out = self.model(imgs)
        self.assertEqual(out.shape, (4, 1280))

    def test_from_list_of_paths(self):
        out = self.model([IMAGE_FACE, IMAGE_FACE])
        self.assertEqual(out.shape, (2, 1280))

    # ------------------------------------------------------------------
    # Preprocess output
    # ------------------------------------------------------------------

    def test_preprocess_output_shape(self):
        imgs = torch.randint(0, 255, (3, 3, 64, 64), dtype=torch.uint8)
        preprocessed = self.model.preprocess(imgs)
        self.assertEqual(preprocessed.shape, (3, 3, 224, 224))

    def test_preprocess_output_dtype(self):
        imgs = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
        preprocessed = self.model.preprocess(imgs)
        self.assertEqual(preprocessed.dtype, torch.float32)

    def test_preprocess_output_range(self):
        imgs = torch.randint(0, 255, (8, 3, 64, 64), dtype=torch.uint8)
        preprocessed = self.model.preprocess(imgs)
        self.assertLess(preprocessed.mean().abs().item(), 2.0)

    # ------------------------------------------------------------------
    # L2 normalisation
    # ------------------------------------------------------------------

    def test_output_is_not_all_zeros(self):
        imgs = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
        out = self.model(imgs)
        self.assertTrue((out.abs().sum(dim=-1) > 0).all())

    # ------------------------------------------------------------------
    # Consistency
    # ------------------------------------------------------------------

    def test_feature_dim_consistent_across_inputs(self):
        out1 = self.model(IMAGE_FACE)
        out2 = self.model(IMAGE_EMMA)
        self.assertEqual(out1.shape[1], out2.shape[1])

    def test_deterministic_inference(self):
        imgs = torch.randint(0, 255, (2, 3, 224, 224), dtype=torch.uint8)
        out1 = self.model(imgs)
        out2 = self.model(imgs)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    # ------------------------------------------------------------------
    # video_to_feature helper
    # ------------------------------------------------------------------

    def test_video_to_feature(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = pathlib.Path(tmp_dir) / "emotieffnet_features.st"
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
            self.assertEqual(feats.shape[1], 1280)
            self.assertGreater(feats.shape[0], 0)
            self.assertEqual(len(ids), feats.shape[0])

    def test_video_to_feature_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = pathlib.Path(tmp_dir) / "emotieffnet_features.st"
            result1 = self.model.video_to_feature(
                VIDEO_MULTISPEAKER_SHORT,
                batch_size=8,
                output_path=out_path,
            )
            result2 = self.model.video_to_feature(
                VIDEO_MULTISPEAKER_SHORT,
                batch_size=8,
                output_path=out_path,
            )
            self.assertTrue(torch.allclose(result1["features"], result2["features"]))


class TestEmotiEffNetWrapperB2(unittest.TestCase):
    """Inference tests for the B2 variant."""

    @classmethod
    def setUpClass(cls):
        cls.model = EmotiEffNetWrapper(model_name="enet_b2_8", device_id=None)

    def test_feature_dim_attribute(self):
        self.assertEqual(self.model.feature_dim, 1408)

    def test_img_size_attribute(self):
        self.assertEqual(self.model.img_size, 260)

    def test_from_numpy_hwc(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        out = self.model(img)
        self.assertEqual(out.shape, (1, 1408))

    def test_from_batch_tensor(self):
        imgs = torch.randint(0, 255, (4, 3, 260, 260), dtype=torch.uint8)
        out = self.model(imgs)
        self.assertEqual(out.shape, (4, 1408))

    def test_preprocess_output_shape(self):
        imgs = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
        preprocessed = self.model.preprocess(imgs)
        self.assertEqual(preprocessed.shape, (2, 3, 260, 260))

    def test_output_is_not_all_zeros(self):
        imgs = torch.randint(0, 255, (4, 3, 260, 260), dtype=torch.uint8)
        out = self.model(imgs)
        self.assertTrue((out.abs().sum(dim=-1) > 0).all())

    def test_deterministic_inference(self):
        imgs = torch.randint(0, 255, (2, 3, 260, 260), dtype=torch.uint8)
        out1 = self.model(imgs)
        out2 = self.model(imgs)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
