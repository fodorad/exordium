"""Tests for exordium.utils.device: get_torch_device, get_default_device, get_module_device."""

import unittest
from unittest.mock import patch

import torch

from exordium.utils.device import get_default_device, get_module_device, get_torch_device


class TestGetTorchDevice(unittest.TestCase):
    def test_none_returns_cpu(self):
        device = get_torch_device(None)
        self.assertEqual(device, torch.device("cpu"))

    def test_negative_returns_cpu(self):
        device = get_torch_device(-1)
        self.assertEqual(device, torch.device("cpu"))

    def test_returns_torch_device(self):
        device = get_torch_device(None)
        self.assertIsInstance(device, torch.device)

    def test_positive_id_returns_device(self):
        # On a machine without GPU this falls back to CPU
        device = get_torch_device(0)
        self.assertIsInstance(device, torch.device)


class TestGetTorchDeviceCudaPath(unittest.TestCase):
    def test_cuda_path_when_mps_unavailable(self):
        """When MPS is unavailable but CUDA is, should return cuda device."""
        with (
            patch("torch.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            device = get_torch_device(0)
        self.assertEqual(device.type, "cuda")
        self.assertEqual(device.index, 0)

    def test_cpu_fallback_when_both_unavailable(self):
        """When both MPS and CUDA are unavailable, should return CPU."""
        with (
            patch("torch.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            device = get_torch_device(0)
        self.assertEqual(device.type, "cpu")


class TestGetDefaultDevice(unittest.TestCase):
    def test_returns_torch_device(self):
        device = get_default_device()
        self.assertIsInstance(device, torch.device)

    def test_device_type_is_valid(self):
        device = get_default_device()
        self.assertIn(device.type, ("cpu", "cuda", "mps"))

    def test_cuda_default_when_mps_unavailable(self):
        """When MPS unavailable but CUDA available, default is cuda:0."""
        with (
            patch("torch.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=True),
        ):
            device = get_default_device()
        self.assertEqual(device.type, "cuda")

    def test_cpu_default_when_neither_available(self):
        with (
            patch("torch.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            device = get_default_device()
        self.assertEqual(device.type, "cpu")


class TestGetModuleDevice(unittest.TestCase):
    def test_cpu_module(self):
        model = torch.nn.Linear(4, 4)
        device = get_module_device(model)
        self.assertIsInstance(device, torch.device)
        self.assertEqual(device.type, "cpu")

    def test_linear_on_cpu(self):
        module = torch.nn.Linear(4, 4)
        device = get_module_device(module)
        self.assertEqual(device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
