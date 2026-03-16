"""Tests for exordium.utils.transforms module."""

import unittest

import numpy as np
import torch

from exordium.utils.transforms import (
    CenterCrop,
    Denormalize,
    Normalize,
    Resize,
    ToTensor,
    crop_and_pad_window,
    get_random_eraser,
    spec_augment,
)


class TestSpecAugment(unittest.TestCase):
    """Tests for spec_augment function."""

    def test_spec_augment_basic(self):
        """Test basic spectrogram augmentation."""
        spec = np.random.randn(128, 100, 1).astype(np.float32)
        augmented = spec_augment(spec, num_mask=2)

        self.assertEqual(augmented.shape, spec.shape, "Shape should be preserved")
        self.assertFalse(np.array_equal(augmented, spec), "Spec should be modified")

    def test_spec_augment_zero_masking(self):
        """Test with zero masking percentage."""
        spec = np.random.randn(128, 100, 1).astype(np.float32)
        augmented = spec_augment(
            spec, num_mask=1, freq_masking_max_percentage=0.0, time_masking_max_percentage=0.0
        )

        # With 0% masking, output should be very similar to input
        np.testing.assert_array_almost_equal(augmented, spec, decimal=5)

    def test_spec_augment_num_masks(self):
        """Test with different number of masks."""
        spec = np.random.randn(128, 100, 1).astype(np.float32)
        augmented = spec_augment(spec, num_mask=5)

        self.assertEqual(augmented.shape, spec.shape, "Shape should be preserved")

    def test_spec_augment_custom_mean(self):
        """Test with custom mean value."""
        spec = np.random.randn(128, 100, 1).astype(np.float32)
        augmented = spec_augment(spec, num_mask=2, mean=-1.0)

        self.assertEqual(augmented.shape, spec.shape, "Shape should be preserved")


class TestCropAndPadWindow(unittest.TestCase):
    """Tests for crop_and_pad_window function."""

    def test_crop_and_pad_window_basic(self):
        """Test basic cropping and padding."""
        x = np.random.randn(100, 10)
        result = crop_and_pad_window(x, win_size=5, m_freq=10, timestep=8)

        expected_height = int(5 * 10)  # win_size * m_freq
        self.assertEqual(result.shape, (expected_height, 10), "Output shape should match expected")

    def test_crop_and_pad_window_edge_case(self):
        """Test with timestep at the beginning."""
        x = np.random.randn(100, 10)
        result = crop_and_pad_window(x, win_size=3, m_freq=10, timestep=5)

        expected_height = int(3 * 10)  # win_size * m_freq
        self.assertEqual(result.shape, (expected_height, 10), "Should match expected shape")
        self.assertEqual(result.shape[1], x.shape[1], "Width should be preserved")


class TestGetRandomEraser(unittest.TestCase):
    """Tests for get_random_eraser function."""

    def test_get_random_eraser_basic(self):
        """Test basic random eraser."""
        eraser = get_random_eraser(p=1.0)  # Always erase
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8).copy()
        erased = eraser(img.copy())

        self.assertEqual(erased.shape, img.shape, "Shape should be preserved")

    def test_get_random_eraser_no_erase(self):
        """Test with p=0 (never erase)."""
        eraser = get_random_eraser(p=0.0)  # Never erase
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        erased = eraser(img.copy())

        np.testing.assert_array_equal(erased, img, "Image should be unchanged")

    def test_get_random_eraser_pixel_level(self):
        """Test pixel-level erasing."""
        eraser = get_random_eraser(p=1.0, pixel_level=True)
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        erased = eraser(img.copy())

        self.assertEqual(erased.shape, img.shape, "Shape should be preserved")


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
