"""Tests for exordium.utils.padding: pad_or_crop_time_dim, pad_layered_time_dim."""

import unittest

import torch

from exordium.utils.padding import pad_layered_time_dim, pad_or_crop_time_dim


class TestPadOrCropTimeDim(unittest.TestCase):
    def test_padding_correct_size(self):
        tensor = torch.randn(5, 8)
        out, mask = pad_or_crop_time_dim(tensor, target_size=10)
        self.assertEqual(out.shape[0], 10)
        self.assertEqual(mask.shape[0], 10)

    def test_padding_mask_false_for_padded(self):
        tensor = torch.randn(5, 8)
        out, mask = pad_or_crop_time_dim(tensor, target_size=10)
        self.assertTrue(mask[:5].all())
        self.assertFalse(mask[5:].any())

    def test_cropping_correct_size(self):
        tensor = torch.randn(15, 8)
        out, mask = pad_or_crop_time_dim(tensor, target_size=10)
        self.assertEqual(out.shape[0], 10)

    def test_cropping_mask_all_true(self):
        tensor = torch.randn(15, 8)
        out, mask = pad_or_crop_time_dim(tensor, target_size=10)
        self.assertTrue(mask.all())

    def test_exact_size_no_change(self):
        tensor = torch.randn(10, 8)
        out, mask = pad_or_crop_time_dim(tensor, target_size=10)
        self.assertEqual(out.shape[0], 10)
        self.assertTrue(mask.all())

    def test_vector_input(self):
        tensor = torch.randn(5)
        out, mask = pad_or_crop_time_dim(tensor, target_size=8)
        self.assertEqual(out.shape[0], 8)
        self.assertEqual(out.ndim, 1)


class TestPadLayeredTimeDim(unittest.TestCase):
    def test_padding_shape(self):
        tensor = torch.randn(4, 5, 8)
        out, mask = pad_layered_time_dim(tensor, target_time_dim=10)
        self.assertEqual(out.shape, (4, 10, 8))
        self.assertEqual(mask.shape[0], 10)

    def test_padding_mask(self):
        tensor = torch.randn(4, 5, 8)
        out, mask = pad_layered_time_dim(tensor, target_time_dim=10)
        self.assertTrue(mask[:5].all())
        self.assertFalse(mask[5:].any())

    def test_cropping_shape(self):
        tensor = torch.randn(4, 15, 8)
        out, mask = pad_layered_time_dim(tensor, target_time_dim=10)
        self.assertEqual(out.shape, (4, 10, 8))

    def test_exact_size(self):
        tensor = torch.randn(4, 10, 8)
        out, mask = pad_layered_time_dim(tensor, target_time_dim=10)
        self.assertEqual(out.shape, (4, 10, 8))


if __name__ == "__main__":
    unittest.main()
