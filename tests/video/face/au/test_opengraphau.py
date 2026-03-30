"""Tests for OpenGraphAU action unit extractor."""

import unittest

import numpy as np
import torch

from exordium.video.face.au.opengraphau import (
    AU_REGISTRY,
    AU_ids,
    AU_names,
    OpenGraphAuWrapper,
    _OPENGRAPHAU_WEIGHTS,
)
from tests.fixtures import IMAGE_FACE, hf_file_exists


class TestOpenGraphAuInit(unittest.TestCase):
    def test_invalid_stage_raises(self):
        with self.assertRaises(ValueError):
            OpenGraphAuWrapper(stage=3)

    def test_invalid_stage_zero_raises(self):
        with self.assertRaises(ValueError):
            OpenGraphAuWrapper(stage=0)


class TestOpenGraphAuRegistry(unittest.TestCase):
    def test_registry_has_41_entries(self):
        self.assertEqual(len(AU_REGISTRY), 41)

    def test_au_ids_length(self):
        self.assertEqual(len(AU_ids), 41)

    def test_au_names_length(self):
        self.assertEqual(len(AU_names), 41)

    def test_registry_entries_are_tuples(self):
        for entry in AU_REGISTRY:
            self.assertIsInstance(entry, tuple)
            self.assertEqual(len(entry), 2)

    def test_weight_stages_are_1_and_2(self):
        self.assertIn(1, _OPENGRAPHAU_WEIGHTS)
        self.assertIn(2, _OPENGRAPHAU_WEIGHTS)

    def test_weight_filenames_are_strings(self):
        for stage, fname in _OPENGRAPHAU_WEIGHTS.items():
            self.assertIsInstance(fname, str)
            self.assertTrue(fname.endswith(".pth"))


class TestOpenGraphAuWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.model = OpenGraphAuWrapper(stage=1, device_id=None)
        except Exception as e:
            raise unittest.SkipTest(f"OpenGraphAU weights not available: {e}") from e

    def test_from_image_path(self):
        out = self.model(IMAGE_FACE)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 2)

    def test_from_numpy(self):
        out = self.model(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        self.assertIsInstance(out, torch.Tensor)

    def test_output_shape(self):
        out = self.model(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        self.assertEqual(out.shape[0], 1)
        self.assertGreater(out.shape[1], 0)

    def test_batch_tensor(self):
        out = self.model(torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8))
        self.assertEqual(out.shape[0], 4)


class TestOpenGraphAuWeightAvailability(unittest.TestCase):
    def test_opengraphau_stage1_weights_file(self):
        self.assertTrue(
            hf_file_exists("fodorad/exordium-weights", "opengraphau-swint-1s_weights.pth"),
            "opengraphau-swint-1s_weights.pth not found in fodorad/exordium-weights",
        )


if __name__ == "__main__":
    unittest.main()
