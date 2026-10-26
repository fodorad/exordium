"""Tests for the Gaze-LLE gaze-target wrapper."""

import unittest

import numpy as np
import torch

from exordium.video.face.gaze.gazelle import (
    DEFAULT_VARIANT,
    GAZELLE_VARIANTS,
    GazelleOutput,
    GazelleWrapper,
)
from tests.fixtures import IMAGE_MULTISPEAKER, PRETRAINED, ModelTestCase, hf_file_exists

BOX_A = (0.1, 0.1, 0.3, 0.4)
BOX_B = (0.6, 0.2, 0.8, 0.5)


def _frames(batch: int = 2, h: int = 90, w: int = 160) -> torch.Tensor:
    return torch.randint(0, 255, (batch, 3, h, w), dtype=torch.uint8)


class TestGazelleWrapper(ModelTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GazelleWrapper("vitb14_inout", device_id=None, pretrained=PRETRAINED)

    def test_output_shapes_and_range(self):
        out = self.model(_frames(2), [[BOX_A, BOX_B], [BOX_A, BOX_B]])
        self.assertEqual(out.heatmap.shape, (2, 2, 64, 64))
        self.assertEqual(out.valid.shape, (2, 2))
        self.assertEqual(out.inout.shape, (2, 2))
        self.assertGreaterEqual(float(out.heatmap.min()), 0.0)
        self.assertLessEqual(float(out.heatmap.max()), 1.0)
        self.assertGreaterEqual(float(out.inout.min()), 0.0)
        self.assertLessEqual(float(out.inout.max()), 1.0)

    def test_variable_people_are_padded(self):
        out = self.model(_frames(2), [[BOX_A, BOX_B], [BOX_A]])
        self.assertEqual(out.heatmap.shape[:2], (2, 2))
        self.assertEqual(out.valid.tolist(), [[True, True], [True, False]])
        self.assertEqual(float(out.heatmap[1, 1].abs().sum()), 0.0)
        self.assertEqual(float(out.inout[1, 1]), 0.0)

    def test_frame_without_people(self):
        out = self.model(_frames(2), [[BOX_A], []])
        self.assertEqual(out.valid.tolist(), [[True], [False]])

    def test_no_people_at_all(self):
        out = self.model(_frames(1), [[]])
        self.assertEqual(out.heatmap.shape, (1, 0, 64, 64))

    def test_none_box_allowed(self):
        out = self.model(_frames(1), [[None]])
        self.assertEqual(out.heatmap.shape, (1, 1, 64, 64))

    def test_non_square_inputs(self):
        for h, w in [(60, 200), (300, 120)]:
            out = self.model(_frames(1, h, w), [[BOX_A]])
            self.assertEqual(out.heatmap.shape, (1, 1, 64, 64))

    def test_accepts_numpy_and_path(self):
        arr = np.random.randint(0, 255, (80, 120, 3), dtype=np.uint8)
        self.assertEqual(self.model(arr, [[BOX_A]]).heatmap.shape[0], 1)
        self.assertEqual(self.model(IMAGE_MULTISPEAKER, [[BOX_A]]).heatmap.shape[0], 1)

    def test_box_count_must_match_frames(self):
        with self.assertRaises(ValueError):
            self.model(_frames(2), [[BOX_A]])

    def test_out_of_range_box_is_clipped(self):
        out = self.model(_frames(1), [[(-0.5, -0.5, 1.5, 1.5)]])
        self.assertEqual(out.heatmap.shape, (1, 1, 64, 64))

    def test_deterministic(self):
        frames = _frames(1)
        a = self.model(frames, [[BOX_A]])
        b = self.model(frames, [[BOX_A]])
        self.assertTrue(torch.equal(a.heatmap, b.heatmap))

    def test_people_in_one_frame_do_not_leak_across_frames(self):
        frames = _frames(2)
        both = self.model(frames, [[BOX_A], [BOX_B]])
        alone = self.model(frames[1:], [[BOX_B]])
        self.assertTrue(torch.allclose(both.heatmap[1], alone.heatmap[0], atol=1e-4))

    def test_head_box_changes_prediction(self):
        frames = _frames(1)
        a = self.model(frames, [[BOX_A]])
        b = self.model(frames, [[BOX_B]])
        self.assertFalse(torch.allclose(a.heatmap, b.heatmap))

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            GazelleWrapper("vitx", pretrained=False)

    def test_variants_registry(self):
        self.assertIn(DEFAULT_VARIANT, GAZELLE_VARIANTS)
        self.assertEqual(len(GAZELLE_VARIANTS), 6)


class TestGazelleNoInout(unittest.TestCase):
    def test_inout_is_none(self):
        model = GazelleWrapper("vitb14", pretrained=False)
        out = model(_frames(1), [[BOX_A]])
        self.assertIsNone(out.inout)
        self.assertEqual(out.heatmap.shape, (1, 1, 64, 64))


class TestGazelleOutput(unittest.TestCase):
    @staticmethod
    def _output() -> GazelleOutput:
        heatmap = torch.zeros(1, 2, 64, 64)
        heatmap[0, 0, 10, 40] = 1.0
        return GazelleOutput(
            heatmap=heatmap,
            valid=torch.tensor([[True, False]]),
            inout=torch.tensor([[0.9, 0.0]]),
        )

    def test_gaze_point(self):
        point = self._output().gaze_point()
        self.assertEqual(point.shape, (1, 2, 2))
        self.assertAlmostEqual(float(point[0, 0, 0]), 40.5 / 64)
        self.assertAlmostEqual(float(point[0, 0, 1]), 10.5 / 64)

    def test_mass_in_box(self):
        out = self._output()
        mass = out.mass_in_box((0.5, 0.0, 1.0, 0.5))
        self.assertEqual(mass.shape, (1, 2))
        self.assertAlmostEqual(float(mass[0, 0]), 1.0, places=5)
        self.assertEqual(float(mass[0, 1]), 0.0)


class TestGazelleWeightAvailability(unittest.TestCase):
    def test_all_checkpoints_mirrored(self):
        for spec in GAZELLE_VARIANTS.values():
            self.assertTrue(
                hf_file_exists("fodorad/exordium-weights", spec.weight_file),
                f"{spec.weight_file} not found in fodorad/exordium-weights",
            )


if __name__ == "__main__":
    unittest.main()
