"""Tests for exordium.utils.padding module."""

import unittest

import numpy as np
import torch

from exordium.utils.padding import pad_layered_time_dim, pad_or_crop_time_dim


class TestPadOrCropTimeDim(unittest.TestCase):
    """Test pad_or_crop_time_dim function."""

    def test_padding_numpy_2d(self):
        """Test padding with 2D NumPy array."""
        array = np.array([[1, 2], [3, 4], [5, 6]])  # Shape (3, 2)
        padded, mask = pad_or_crop_time_dim(array, target_size=5, pad_value=0)

        self.assertEqual(padded.shape, (5, 2))
        self.assertTrue(np.array_equal(padded[:3], array))
        self.assertTrue(np.array_equal(padded[3:], np.zeros((2, 2))))
        self.assertTrue(np.array_equal(mask, np.array([True, True, True, False, False])))

    def test_padding_numpy_1d(self):
        """Test padding with 1D NumPy array (vector)."""
        array = np.array([1, 2, 3])  # Shape (3,)
        padded, mask = pad_or_crop_time_dim(array, target_size=5, pad_value=0)

        self.assertEqual(padded.shape, (5,))
        self.assertTrue(np.array_equal(padded[:3], array))
        self.assertTrue(np.array_equal(padded[3:], np.zeros(2)))
        self.assertTrue(np.array_equal(mask, np.array([True, True, True, False, False])))

    def test_padding_torch_2d(self):
        """Test padding with 2D PyTorch tensor."""
        tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])  # Shape (3, 2)
        padded, mask = pad_or_crop_time_dim(tensor, target_size=5, pad_value=0)

        self.assertEqual(padded.shape, (5, 2))
        self.assertTrue(torch.equal(padded[:3], tensor))
        self.assertTrue(torch.equal(padded[3:], torch.zeros(2, 2)))
        self.assertTrue(torch.equal(mask, torch.tensor([True, True, True, False, False])))

    def test_padding_torch_1d(self):
        """Test padding with 1D PyTorch tensor (vector)."""
        tensor = torch.tensor([1.0, 2.0, 3.0])  # Shape (3,)
        padded, mask = pad_or_crop_time_dim(tensor, target_size=5, pad_value=0)

        self.assertEqual(padded.shape, (5,))
        self.assertTrue(torch.equal(padded[:3], tensor))
        self.assertTrue(torch.equal(padded[3:], torch.zeros(2)))
        self.assertTrue(torch.equal(mask, torch.tensor([True, True, True, False, False])))

    def test_cropping_numpy(self):
        """Test cropping with NumPy array."""
        array = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])  # Shape (5, 2)
        cropped, mask = pad_or_crop_time_dim(array, target_size=3)

        self.assertEqual(cropped.shape, (3, 2))
        self.assertTrue(np.array_equal(cropped, array[:3]))
        self.assertTrue(np.array_equal(mask, np.ones(3, dtype=bool)))

    def test_cropping_torch(self):
        """Test cropping with PyTorch tensor."""
        tensor = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])  # Shape (4, 2)
        cropped, mask = pad_or_crop_time_dim(tensor, target_size=2)

        self.assertEqual(cropped.shape, (2, 2))
        self.assertTrue(torch.equal(cropped, tensor[:2]))
        self.assertTrue(torch.equal(mask, torch.ones(2, dtype=torch.bool)))

    def test_no_change_needed(self):
        """Test when array size equals target size."""
        array = np.array([[1, 2], [3, 4], [5, 6]])  # Shape (3, 2)
        result, mask = pad_or_crop_time_dim(array, target_size=3)

        self.assertEqual(result.shape, (3, 2))
        self.assertTrue(np.array_equal(result, array))
        self.assertTrue(np.array_equal(mask, np.ones(3, dtype=bool)))

    def test_custom_pad_value(self):
        """Test padding with custom pad value."""
        array = np.array([[1, 2], [3, 4]])  # Shape (2, 2)
        padded, mask = pad_or_crop_time_dim(array, target_size=4, pad_value=-1)

        self.assertEqual(padded.shape, (4, 2))
        self.assertTrue(np.array_equal(padded[:2], array))
        self.assertTrue(np.array_equal(padded[2:], np.full((2, 2), -1)))

    def test_invalid_input_type(self):
        """Test with invalid input type."""
        with self.assertRaises(ValueError):
            pad_or_crop_time_dim([1, 2, 3], target_size=5)

    def test_edge_case_single_element(self):
        """Test with single element arrays."""
        array = np.array([[1, 2]])  # Shape (1, 2)
        padded, mask = pad_or_crop_time_dim(array, target_size=3, pad_value=0)

        self.assertEqual(padded.shape, (3, 2))
        self.assertTrue(np.array_equal(mask, np.array([True, False, False])))


class TestPadLayeredTimeDim(unittest.TestCase):
    """Test pad_layered_time_dim function."""

    def test_padding_required(self):
        """Test padding when input is smaller than target."""
        tensor = torch.randn(2, 3, 5)  # (L=2, N=3, F=5)
        padded, mask = pad_layered_time_dim(tensor, target_time_dim=5, pad_value=0)

        self.assertEqual(padded.shape, (2, 5, 5))
        self.assertTrue(torch.equal(padded[:, :3, :], tensor))
        self.assertTrue(torch.equal(padded[:, 3:, :], torch.zeros(2, 2, 5)))
        self.assertTrue(torch.equal(mask, torch.tensor([True, True, True, False, False])))

    def test_no_padding_needed(self):
        """Test when input size equals target."""
        tensor = torch.randn(2, 3, 5)  # (L=2, N=3, F=5)
        padded, mask = pad_layered_time_dim(tensor, target_time_dim=3, pad_value=0)

        self.assertEqual(padded.shape, (2, 3, 5))
        self.assertTrue(torch.equal(padded, tensor))
        self.assertTrue(torch.equal(mask, torch.ones(3, dtype=torch.bool)))

    def test_cropping_required(self):
        """Test cropping when input is larger than target."""
        tensor = torch.randn(2, 5, 3)  # (L=2, N=5, F=3)
        padded, mask = pad_layered_time_dim(tensor, target_time_dim=3, pad_value=0)

        self.assertEqual(padded.shape, (2, 3, 3))
        self.assertTrue(torch.equal(padded, tensor[:, :3, :]))
        self.assertTrue(torch.equal(mask, torch.ones(3, dtype=torch.bool)))

    def test_custom_pad_value(self):
        """Test padding with custom pad value."""
        tensor = torch.ones(2, 2, 3)  # (L=2, N=2, F=3)
        padded, mask = pad_layered_time_dim(tensor, target_time_dim=4, pad_value=-1)

        self.assertEqual(padded.shape, (2, 4, 3))
        self.assertTrue(torch.equal(padded[:, :2, :], tensor))
        self.assertTrue(torch.equal(padded[:, 2:, :], torch.full((2, 2, 3), -1)))

    def test_single_layer(self):
        """Test with single layer."""
        tensor = torch.randn(1, 3, 5)  # (L=1, N=3, F=5)
        padded, mask = pad_layered_time_dim(tensor, target_time_dim=5, pad_value=0)

        self.assertEqual(padded.shape, (1, 5, 5))
        self.assertTrue(torch.equal(mask[:3], torch.ones(3, dtype=torch.bool)))
        self.assertTrue(torch.equal(mask[3:], torch.zeros(2, dtype=torch.bool)))

    def test_mask_correctness(self):
        """Test that mask correctly indicates padded regions."""
        tensor = torch.randn(3, 4, 2)  # (L=3, N=4, F=2)
        padded, mask = pad_layered_time_dim(tensor, target_time_dim=7, pad_value=0)

        self.assertEqual(mask.shape, (7,))
        self.assertTrue(torch.all(mask[:4]))  # First 4 should be True
        self.assertFalse(torch.any(mask[4:]))  # Last 3 should be False


if __name__ == "__main__":
    unittest.main()
