"""Tests for exordium.utils.device module."""

import unittest
from unittest.mock import patch

from exordium.utils.device import get_default_device, get_torch_device


class TestGetTorchDevice(unittest.TestCase):
    """Test get_torch_device function."""

    def test_device_none_returns_cpu(self):
        """Test that device_id=None returns CPU device."""
        device = get_torch_device(None)
        self.assertEqual(device.type, "cpu")

    def test_device_negative_returns_cpu(self):
        """Test that negative device_id returns CPU device."""
        device = get_torch_device(-1)
        self.assertEqual(device.type, "cpu")
        device = get_torch_device(-5)
        self.assertEqual(device.type, "cpu")

    @patch("torch.mps.is_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_device_mps_available(self, mock_cuda, mock_mps):
        """Test that MPS device is returned when available."""
        device = get_torch_device(0)
        self.assertEqual(device.type, "mps")
        self.assertEqual(str(device), "mps:0")

    @patch("torch.mps.is_available", return_value=False)
    @patch("torch.cuda.is_available", return_value=True)
    def test_device_cuda_available(self, mock_cuda, mock_mps):
        """Test that CUDA device is returned when available."""
        device = get_torch_device(0)
        self.assertEqual(device.type, "cuda")
        self.assertEqual(str(device), "cuda:0")

    @patch("torch.mps.is_available", return_value=False)
    @patch("torch.cuda.is_available", return_value=False)
    def test_device_fallback_to_cpu(self, mock_cuda, mock_mps):
        """Test that CPU is returned when no accelerators are available."""
        device = get_torch_device(0)
        self.assertEqual(device.type, "cpu")

    @patch("torch.mps.is_available", return_value=True)
    def test_device_with_specific_id(self, mock_mps):
        """Test device selection with specific ID."""
        device = get_torch_device(2)
        self.assertEqual(str(device), "mps:2")


class TestGetDefaultDevice(unittest.TestCase):
    """Test get_default_device function."""

    @patch("torch.mps.is_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_default_device_mps(self, mock_cuda, mock_mps):
        """Test that MPS:0 is default when available."""
        device = get_default_device()
        self.assertEqual(str(device), "mps:0")

    @patch("torch.mps.is_available", return_value=False)
    @patch("torch.cuda.is_available", return_value=True)
    def test_default_device_cuda(self, mock_cuda, mock_mps):
        """Test that CUDA:0 is default when available."""
        device = get_default_device()
        self.assertEqual(str(device), "cuda:0")

    @patch("torch.mps.is_available", return_value=False)
    @patch("torch.cuda.is_available", return_value=False)
    def test_default_device_cpu(self, mock_cuda, mock_mps):
        """Test that CPU is default when no accelerators available."""
        device = get_default_device()
        self.assertEqual(device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
