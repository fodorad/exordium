"""Tests for gaze-target geometry utilities (boxes, heatmap points, box mass)."""

import unittest

import torch

from exordium.video.face.gaze.base import (
    box_to_normalized,
    box_to_pixels,
    expand_box,
    heatmap_to_point,
    mass_in_box,
)


def _peak(h: int = 64, w: int = 64, y: int = 10, x: int = 40) -> torch.Tensor:
    heatmap = torch.zeros(h, w)
    heatmap[y, x] = 1.0
    return heatmap


class TestBoxConversion(unittest.TestCase):
    def test_to_normalized(self):
        box = box_to_normalized((100, 50, 300, 150), width=400, height=200)
        self.assertEqual(box, (0.25, 0.25, 0.75, 0.75))

    def test_round_trip(self):
        box = (12.0, 30.0, 220.0, 180.0)
        back = box_to_pixels(box_to_normalized(box, 640, 360), 640, 360)
        for a, b in zip(box, back):
            self.assertAlmostEqual(a, b, places=4)

    def test_to_normalized_clips_to_unit_range(self):
        box = box_to_normalized((-10, -5, 500, 300), width=400, height=200)
        self.assertEqual(box, (0.0, 0.0, 1.0, 1.0))

    def test_invalid_image_size(self):
        with self.assertRaises(ValueError):
            box_to_normalized((0, 0, 1, 1), width=0, height=10)


class TestExpandBox(unittest.TestCase):
    def test_scale_one_is_identity(self):
        box = (0.2, 0.2, 0.4, 0.6)
        for a, b in zip(expand_box(box, 1.0), box):
            self.assertAlmostEqual(a, b, places=6)

    def test_expands_around_center(self):
        x1, y1, x2, y2 = expand_box((0.4, 0.4, 0.6, 0.6), 2.0)
        self.assertAlmostEqual(x1, 0.3)
        self.assertAlmostEqual(y2, 0.7)

    def test_clipped_to_unit_range(self):
        box = expand_box((0.0, 0.0, 0.5, 0.5), 3.0)
        self.assertEqual((box[0], box[1]), (0.0, 0.0))
        self.assertLessEqual(box[2], 1.0)

    def test_invalid_scale(self):
        with self.assertRaises(ValueError):
            expand_box((0.1, 0.1, 0.2, 0.2), 0.0)


class TestHeatmapToPoint(unittest.TestCase):
    def test_argmax_finds_peak(self):
        point = heatmap_to_point(_peak(y=10, x=40), mode="argmax")
        self.assertEqual(point.shape, (2,))
        self.assertAlmostEqual(float(point[0]), (40 + 0.5) / 64)
        self.assertAlmostEqual(float(point[1]), (10 + 0.5) / 64)

    def test_soft_matches_argmax_on_single_peak(self):
        heatmap = _peak()
        self.assertTrue(
            torch.allclose(heatmap_to_point(heatmap, "soft"), heatmap_to_point(heatmap, "argmax"))
        )

    def test_soft_is_mean_of_two_peaks(self):
        heatmap = _peak(y=10, x=10) + _peak(y=10, x=50)
        point = heatmap_to_point(heatmap, "soft")
        self.assertAlmostEqual(float(point[0]), (30 + 0.5) / 64, places=5)

    def test_batch_dims_preserved(self):
        point = heatmap_to_point(torch.rand(2, 3, 64, 64))
        self.assertEqual(point.shape, (2, 3, 2))

    def test_zero_heatmap_soft_is_center(self):
        point = heatmap_to_point(torch.zeros(64, 64), "soft")
        self.assertTrue(torch.allclose(point, torch.tensor([0.5, 0.5])))

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            heatmap_to_point(_peak(), mode="median")


class TestMassInBox(unittest.TestCase):
    def test_full_image_is_one(self):
        mass = mass_in_box(torch.rand(2, 3, 64, 64), (0.0, 0.0, 1.0, 1.0))
        self.assertEqual(mass.shape, (2, 3))
        self.assertTrue(torch.allclose(mass, torch.ones(2, 3)))

    def test_peak_inside_and_outside(self):
        heatmap = _peak(y=10, x=40)
        self.assertAlmostEqual(float(mass_in_box(heatmap, (0.5, 0.0, 1.0, 0.5))), 1.0, places=5)
        self.assertAlmostEqual(float(mass_in_box(heatmap, (0.0, 0.5, 0.5, 1.0))), 0.0, places=5)

    def test_monotone_in_box_size(self):
        heatmap = torch.rand(64, 64)
        small = mass_in_box(heatmap, (0.3, 0.3, 0.5, 0.5))
        large = mass_in_box(heatmap, (0.2, 0.2, 0.7, 0.7))
        self.assertLess(float(small), float(large))

    def test_partial_cell_overlap(self):
        # box covers half of the peak cell horizontally
        heatmap = _peak(y=0, x=0)
        mass = mass_in_box(heatmap, (0.0, 0.0, 0.5 / 64, 1.0))
        self.assertAlmostEqual(float(mass), 0.5, places=5)

    def test_zero_heatmap_gives_zero(self):
        self.assertEqual(float(mass_in_box(torch.zeros(64, 64), (0, 0, 1, 1))), 0.0)

    def test_degenerate_box_gives_zero(self):
        self.assertEqual(float(mass_in_box(torch.rand(64, 64), (0.5, 0.5, 0.5, 0.5))), 0.0)


if __name__ == "__main__":
    unittest.main()
