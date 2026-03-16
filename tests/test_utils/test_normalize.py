"""Tests for exordium.utils.normalize module."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from exordium.utils.normalize import (
    get_mean_std,
    load_params_from_json,
    save_params_to_json,
    standardization,
)


class TestGetMeanStd(unittest.TestCase):
    """Test get_mean_std function."""

    def test_2d_vectors(self):
        """Test mean/std calculation for 2D vectors (B, C)."""
        # Create simple dataset with known mean and std
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        labels = torch.zeros(4)
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=2)

        mean, std = get_mean_std(dataloader, ndim=2, verbose=False)

        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
        # Mean should be [4.0, 5.0]
        self.assertTrue(torch.allclose(mean, torch.tensor([4.0, 5.0]), atol=1e-5))

    def test_3d_time_series(self):
        """Test mean/std calculation for 3D time series (B, T, C)."""
        # Create dataset with shape (B=4, T=3, C=2)
        data = torch.randn(4, 3, 2)
        labels = torch.zeros(4)
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=2)

        mean, std = get_mean_std(dataloader, ndim=3, verbose=False)

        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))
        self.assertTrue(torch.all(std > 0))

    def test_4d_images(self):
        """Test mean/std calculation for 4D images (B, C, H, W)."""
        # Create simple image dataset
        data = torch.ones(4, 3, 8, 8)  # 4 RGB images
        labels = torch.zeros(4)
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=2)

        mean, std = get_mean_std(dataloader, ndim=4, verbose=False)

        self.assertEqual(mean.shape, (3,))
        self.assertEqual(std.shape, (3,))
        # All pixels are 1.0, so mean should be 1.0 and std should be 0.0
        self.assertTrue(torch.allclose(mean, torch.ones(3), atol=1e-5))
        self.assertTrue(torch.allclose(std, torch.zeros(3), atol=1e-5))

    def test_5d_videos(self):
        """Test mean/std calculation for 5D videos (B, T, C, H, W)."""
        # Create simple video dataset
        data = torch.randn(2, 4, 3, 8, 8)  # 2 videos, 4 frames, RGB, 8x8
        labels = torch.zeros(2)
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=1)

        mean, std = get_mean_std(dataloader, ndim=5, verbose=False)

        self.assertEqual(mean.shape, (3,))
        self.assertEqual(std.shape, (3,))
        self.assertTrue(torch.all(std > 0))

    def test_unsupported_ndim(self):
        """Test that unsupported ndim raises NotImplementedError."""
        data = torch.randn(4, 2)
        labels = torch.zeros(4)
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=2)

        with self.assertRaises(NotImplementedError):
            get_mean_std(dataloader, ndim=6, verbose=False)


class TestStandardization(unittest.TestCase):
    """Test standardization function."""

    def test_standardization_2d(self):
        """Test standardization with 2D tensor (B, F)."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mean = torch.tensor([2.0, 3.0])
        std = torch.tensor([1.0, 1.0])

        result = standardization(x, mean, std)

        expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_standardization_3d(self):
        """Test standardization with 3D tensor (B, T, F)."""
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mean = torch.tensor([2.0, 3.0])
        std = torch.tensor([1.0, 1.0])

        result = standardization(x, mean, std)

        expected = torch.tensor([[[-1.0, -1.0], [1.0, 1.0]]])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_standardization_zero_std(self):
        """Test standardization handles zero std gracefully."""
        x = torch.tensor([[1.0, 2.0]])
        mean = torch.tensor([1.0, 2.0])
        std = torch.tensor([0.0, 0.0])

        result = standardization(x, mean, std)

        # Should not raise error and should produce finite values
        self.assertTrue(torch.all(torch.isfinite(result)))

    def test_standardization_preserves_shape(self):
        """Test that standardization preserves input shape."""
        x = torch.randn(5, 10, 20)
        mean = torch.randn(20)
        std = torch.abs(torch.randn(20)) + 0.1  # Ensure positive std

        result = standardization(x, mean, std)

        self.assertEqual(result.shape, x.shape)


class TestSaveLoadParams(unittest.TestCase):
    """Test save_params_to_json and load_params_from_json functions."""

    def test_save_and_load_params(self):
        """Test saving and loading parameters."""
        mean = torch.tensor([1.0, 2.0, 3.0])
        std = torch.tensor([0.5, 1.0, 1.5])

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_params.json"
            save_params_to_json(mean, std, file_path)

            # Check file exists
            self.assertTrue(file_path.exists())

            # Load and verify
            loaded_mean, loaded_std = load_params_from_json(file_path)

            self.assertTrue(torch.equal(mean, loaded_mean))
            self.assertTrue(torch.equal(std, loaded_std))

    def test_save_creates_directory(self):
        """Test that save_params_to_json creates parent directories."""
        mean = torch.tensor([1.0, 2.0])
        std = torch.tensor([0.5, 1.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "subdir" / "params.json"
            save_params_to_json(mean, std, file_path)

            self.assertTrue(file_path.exists())
            self.assertTrue(file_path.parent.exists())

    def test_save_params_format(self):
        """Test that saved JSON has correct format."""
        mean = torch.tensor([1.5, 2.5])
        std = torch.tensor([0.5, 1.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "params.json"
            save_params_to_json(mean, std, file_path)

            # Load raw JSON
            with open(file_path) as f:
                data = json.load(f)

            self.assertIn("mean", data)
            self.assertIn("std", data)
            self.assertEqual(data["mean"], [1.5, 2.5])
            self.assertEqual(data["std"], [0.5, 1.0])

    def test_load_params_dtype(self):
        """Test that loaded parameters have correct dtype."""
        mean = torch.tensor([1.0, 2.0, 3.0])
        std = torch.tensor([0.5, 1.0, 1.5])

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "params.json"
            save_params_to_json(mean, std, file_path)

            loaded_mean, loaded_std = load_params_from_json(file_path)

            self.assertEqual(loaded_mean.dtype, torch.float32)
            self.assertEqual(loaded_std.dtype, torch.float32)

    def test_roundtrip_preserves_values(self):
        """Test that save/load roundtrip preserves exact values."""
        mean = torch.tensor([1.23456789, 2.34567890])
        std = torch.tensor([0.12345678, 0.23456789])

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "params.json"
            save_params_to_json(mean, std, file_path)
            loaded_mean, loaded_std = load_params_from_json(file_path)

            self.assertTrue(torch.allclose(mean, loaded_mean, atol=1e-6))
            self.assertTrue(torch.allclose(std, loaded_std, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
