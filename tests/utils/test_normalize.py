"""Tests for exordium.utils.normalize: standardization, JSON round-trip, get_mean_std."""

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


class TestStandardization(unittest.TestCase):
    def test_known_values(self):
        x = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        mean = torch.tensor([4.0, 6.0])
        std = torch.tensor([2.0, 2.0])
        out = standardization(x, mean, std)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, x.shape)
        self.assertAlmostEqual(out[0, 0].item(), -1.0, places=4)
        self.assertAlmostEqual(out[1, 0].item(), 1.0, places=4)

    def test_zero_mean_unit_std(self):
        x = torch.tensor([[1.0, 2.0]])
        mean = torch.tensor([0.0, 0.0])
        std = torch.tensor([1.0, 1.0])
        out = standardization(x, mean, std)
        torch.testing.assert_close(out, x, atol=1e-4, rtol=1e-4)

    def test_output_shape_preserved(self):
        x = torch.randn(8, 16)
        mean = torch.zeros(16)
        std = torch.ones(16)
        out = standardization(x, mean, std)
        self.assertEqual(out.shape, x.shape)


class TestJsonRoundTrip(unittest.TestCase):
    def test_round_trip_preserves_values(self):
        mean = torch.tensor([1.0, 2.0, 3.0])
        std = torch.tensor([0.5, 1.0, 1.5])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            save_params_to_json(mean, std, path)
            loaded_mean, loaded_std = load_params_from_json(path)
            torch.testing.assert_close(mean, loaded_mean, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(std, loaded_std, atol=1e-5, rtol=1e-5)
        finally:
            path.unlink(missing_ok=True)

    def test_saved_file_exists(self):
        mean = torch.zeros(4)
        std = torch.ones(4)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            save_params_to_json(mean, std, path)
            self.assertTrue(path.exists())
        finally:
            path.unlink(missing_ok=True)


class TestGetMeanStd(unittest.TestCase):
    def _make_loader(self, data: torch.Tensor) -> DataLoader:
        labels = torch.zeros(data.shape[0], dtype=torch.long)
        return DataLoader(TensorDataset(data, labels), batch_size=4)

    def test_vectors_ndim2(self):
        data = torch.randn(20, 8)
        loader = self._make_loader(data)
        mean, std = get_mean_std(loader, ndim=2)
        self.assertEqual(mean.shape, (8,))
        self.assertEqual(std.shape, (8,))

    def test_time_series_ndim3(self):
        data = torch.randn(16, 10, 4)
        loader = self._make_loader(data)
        mean, std = get_mean_std(loader, ndim=3)
        self.assertEqual(mean.shape, (4,))

    def test_images_ndim4(self):
        data = torch.randn(8, 3, 16, 16)
        loader = self._make_loader(data)
        mean, std = get_mean_std(loader, ndim=4)
        self.assertEqual(mean.shape, (3,))

    def test_unsupported_ndim_raises(self):
        data = torch.randn(8, 5)
        loader = self._make_loader(data)
        with self.assertRaises(NotImplementedError):
            get_mean_std(loader, ndim=6)


if __name__ == "__main__":
    unittest.main()
