"""Tests for exordium.video.core.transform module (video transformation classes)."""

import unittest

import numpy as np
import torch

from exordium.video.core.transform import (
    CenterCrop,
    Denormalize,
    Normalize,
    Resize,
    ToTensor,
)


class TestToTensor(unittest.TestCase):
    """Tests for ToTensor class."""

    def test_to_tensor_basic(self):
        """Test basic numpy to tensor conversion."""
        transform = ToTensor()
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        tensor = transform(arr)

        self.assertIsInstance(tensor, torch.Tensor, "Output should be torch.Tensor")
        self.assertEqual(tensor.dtype, torch.float32, "Dtype should be float32")
        self.assertTrue((tensor >= 0).all() and (tensor <= 1).all(), "Values should be in [0, 1]")

    def test_to_tensor_zero_array(self):
        """Test with zero array."""
        transform = ToTensor()
        arr = np.zeros((50, 50, 3), dtype=np.uint8)
        tensor = transform(arr)

        self.assertEqual(tensor.max().item(), 0.0, "Max should be 0")

    def test_to_tensor_max_array(self):
        """Test with maximum value array."""
        transform = ToTensor()
        arr = np.full((50, 50, 3), 255, dtype=np.uint8)
        tensor = transform(arr)

        self.assertAlmostEqual(tensor.max().item(), 1.0, places=5, msg="Max should be 1.0")


class TestResize(unittest.TestCase):
    """Tests for Resize class."""

    def test_resize_basic(self):
        """Test basic video resize."""
        transform = Resize(112)
        video = torch.randn(3, 10, 224, 224)  # (C, T, H, W)
        resized = transform(video)

        self.assertEqual(resized.shape[0], 3, "Channels should be preserved")
        self.assertEqual(resized.shape[1], 10, "Time should be preserved")
        self.assertEqual(min(resized.shape[2:]), 112, "Smallest dim should be 112")

    def test_resize_large_video(self):
        """Test resize with larger video."""
        transform = Resize(256)
        video = torch.randn(3, 10, 480, 640)  # Large video
        resized = transform(video)

        self.assertEqual(resized.shape[0], 3, "Channels should be preserved")
        self.assertEqual(resized.shape[1], 10, "Time should be preserved")
        self.assertEqual(min(resized.shape[2:]), 256, "Smallest dim should be 256")


class TestCenterCrop(unittest.TestCase):
    """Tests for CenterCrop class."""

    def test_center_crop_square(self):
        """Test center crop with square size."""
        transform = CenterCrop(112)
        video = torch.randn(3, 10, 224, 224)  # (C, T, H, W)
        cropped = transform(video)

        self.assertEqual(cropped.shape, (3, 10, 112, 112), "Should crop to 112x112")

    def test_center_crop_rectangular(self):
        """Test center crop with rectangular size."""
        transform = CenterCrop((100, 150))
        video = torch.randn(3, 10, 224, 224)
        cropped = transform(video)

        self.assertEqual(cropped.shape, (3, 10, 100, 150), "Should crop to 100x150")

    def test_center_crop_preserves_content(self):
        """Test that center crop extracts center region."""
        transform = CenterCrop(2)
        video = torch.zeros(1, 1, 4, 4)
        video[0, 0, 1:3, 1:3] = 1.0  # Set center to 1
        cropped = transform(video)

        self.assertTrue((cropped == 1.0).all(), "Should extract center region")


class TestNormalize(unittest.TestCase):
    """Tests for Normalize class."""

    def test_normalize_basic(self):
        """Test basic video normalization."""
        transform = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        video = torch.ones(3, 10, 224, 224) * 0.5
        normalized = transform(video)

        expected = torch.zeros_like(video)
        torch.testing.assert_close(normalized, expected)

    def test_normalize_different_channels(self):
        """Test normalization with different mean/std per channel."""
        transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        video = torch.randn(3, 10, 224, 224)
        normalized = transform(video)

        self.assertEqual(normalized.shape, video.shape, "Shape should be preserved")


class TestDenormalize(unittest.TestCase):
    """Tests for Denormalize class."""

    def test_denormalize_basic(self):
        """Test basic video denormalization."""
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        normalize = Normalize(mean=mean, std=std)
        denormalize = Denormalize(mean=mean, std=std)

        video = torch.rand(3, 10, 224, 224)
        normalized = normalize(video)
        denormalized = denormalize(normalized)

        torch.testing.assert_close(denormalized, video, rtol=1e-5, atol=1e-5)

    def test_denormalize_different_channels(self):
        """Test denormalization with different mean/std per channel."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = Normalize(mean=mean, std=std)
        denormalize = Denormalize(mean=mean, std=std)

        video = torch.rand(3, 10, 224, 224)
        normalized = normalize(video)
        denormalized = denormalize(normalized)

        torch.testing.assert_close(denormalized, video, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
