"""Tests for AdaFaceWrapper face recognition model."""

import unittest

import numpy as np
import torch

from exordium.video.deep.adaface import (
    _FEATURE_DIM,
    _HF_REPO_IDS,
    _INPUT_SIZE,
    AdaFaceWrapper,
)
from tests.fixtures import IMAGE_FACE, hf_repo_exists


class TestAdaFaceConstants(unittest.TestCase):
    def test_feature_dim(self):
        self.assertEqual(_FEATURE_DIM, 512)

    def test_input_size(self):
        self.assertEqual(_INPUT_SIZE, 112)

    def test_supported_backbones(self):
        self.assertIn("ir_18", _HF_REPO_IDS)
        self.assertIn("ir_50", _HF_REPO_IDS)
        self.assertIn("ir_101", _HF_REPO_IDS)


class TestAdaFaceWrapperInit(unittest.TestCase):
    def test_invalid_backbone_raises(self):
        with self.assertRaises(ValueError):
            AdaFaceWrapper(backbone="ir_999")


class TestAdaFaceWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AdaFaceWrapper(backbone="ir_50", device_id=None)

    def test_feature_dim_attribute(self):
        self.assertEqual(self.model.feature_dim, _FEATURE_DIM)

    def test_preprocess_shape(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        preprocessed = self.model.preprocess(img)
        self.assertEqual(preprocessed.shape, (1, 3, _INPUT_SIZE, _INPUT_SIZE))

    def test_preprocess_value_range(self):
        img = torch.randint(0, 255, (1, 3, 224, 224), dtype=torch.uint8)
        preprocessed = self.model.preprocess(img)
        self.assertGreaterEqual(preprocessed.min().item(), -1.0)
        self.assertLessEqual(preprocessed.max().item(), 1.0)

    def test_preprocess_dtype(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        preprocessed = self.model.preprocess(img)
        self.assertEqual(preprocessed.dtype, torch.float32)

    def test_call_from_numpy(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        out = self.model(img)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (1, _FEATURE_DIM))

    def test_call_from_tensor(self):
        out = self.model(torch.randint(0, 255, (1, 3, 112, 112), dtype=torch.uint8))
        self.assertEqual(out.shape, (1, _FEATURE_DIM))

    def test_batch_tensor(self):
        out = self.model(torch.randint(0, 255, (4, 3, 112, 112), dtype=torch.uint8))
        self.assertEqual(out.shape, (4, _FEATURE_DIM))

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[1], _FEATURE_DIM)

    def test_output_is_float(self):
        out = self.model(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        self.assertEqual(out.dtype, torch.float32)

    def test_output_l2_normalized(self):
        out = self.model(torch.randint(0, 255, (3, 3, 112, 112), dtype=torch.uint8))
        norms = torch.norm(out, dim=1)
        for i in range(out.shape[0]):
            self.assertAlmostEqual(norms[i].item(), 1.0, places=5)

    def test_same_input_same_output(self):
        img = torch.randint(0, 255, (1, 3, 112, 112), dtype=torch.uint8)
        out1 = self.model(img)
        out2 = self.model(img)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    def test_different_inputs_different_outputs(self):
        torch.manual_seed(42)
        img1 = torch.randint(0, 255, (1, 3, 112, 112), dtype=torch.uint8)
        torch.manual_seed(99)
        img2 = torch.randint(0, 255, (1, 3, 112, 112), dtype=torch.uint8)
        out1 = self.model(img1)
        out2 = self.model(img2)
        self.assertFalse(torch.allclose(out1, out2, atol=1e-3))


class TestAdaFaceWeightAvailability(unittest.TestCase):
    def test_ir50_repo_exists(self):
        self.assertTrue(hf_repo_exists("minchul/cvlface_adaface_ir50_ms1mv2"))


if __name__ == "__main__":
    unittest.main()
