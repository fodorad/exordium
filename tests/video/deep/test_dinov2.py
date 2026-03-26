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
from tests.fixtures import IMAGE_EMMA, IMAGE_FACE, VIDEO_MULTISPEAKER_SHORT


class TestDINOv2WrapperInit(unittest.TestCase):
    """Initialisation and configuration tests (no model download required)."""

    def test_invalid_model_name_raises(self):
        with self.assertRaises(ValueError):
            DINOv2Wrapper(model_name="xlarge")

    def test_module_constants_consistent(self):
        self.assertEqual(set(_MODEL_IDS.keys()), set(_FEATURE_DIMS.keys()))
        self.assertIn(_DEFAULT_DINOV2_MODEL, _MODEL_IDS)


class TestDINOv2Wrapper(unittest.TestCase):
    """Inference tests — loads the default (base) variant once for the suite."""

    @classmethod
    def setUpClass(cls):
        cls.model = DINOv2Wrapper(model_name="base", device_id=None)

    # ------------------------------------------------------------------
    # Feature dimension
    # ------------------------------------------------------------------

    def test_feature_dim_attribute(self):
        self.assertEqual(self.model.feature_dim, _FEATURE_DIMS["base"])

    # ------------------------------------------------------------------
    # Input types
    # ------------------------------------------------------------------

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], _FEATURE_DIMS["base"])

    def test_from_numpy_hwc(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        out = self.model(img)
        self.assertEqual(out.shape, (1, _FEATURE_DIMS["base"]))

    def test_from_tensor_chw(self):
        t = torch.randint(0, 255, (3, 112, 112), dtype=torch.uint8)
        out = self.model(t)
        self.assertEqual(out.shape, (1, _FEATURE_DIMS["base"]))

    def test_from_batch_tensor(self):
        imgs = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
        out = self.model(imgs)
        self.assertEqual(out.shape, (4, _FEATURE_DIMS["base"]))

    def test_from_list_of_paths(self):
        out = self.model([IMAGE_FACE, IMAGE_FACE])
        self.assertEqual(out.shape, (2, _FEATURE_DIMS["base"]))

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
        # After ImageNet normalisation values should be roughly centred near 0.
        imgs = torch.randint(0, 255, (8, 3, 64, 64), dtype=torch.uint8)
        preprocessed = self.model.preprocess(imgs)
        self.assertLess(preprocessed.mean().abs().item(), 2.0)

    # ------------------------------------------------------------------
    # L2 normalisation
    # ------------------------------------------------------------------

    def test_output_is_l2_normalised(self):
        imgs = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
        out = self.model(imgs)
        norms = out.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

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
            out_path = pathlib.Path(tmp_dir) / "dinov2_features.st"
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
            self.assertEqual(feats.shape[1], _FEATURE_DIMS["base"])
            self.assertGreater(feats.shape[0], 0)
            self.assertEqual(len(ids), feats.shape[0])

    def test_video_to_feature_cache(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = pathlib.Path(tmp_dir) / "dinov2_features.st"
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


if __name__ == "__main__":
    unittest.main()
