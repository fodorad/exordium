import unittest

import numpy as np
import torch

from exordium.audio.io import load_audio
from exordium.audio.smile import OpensmileWrapper
from tests.fixtures import AUDIO_MULTISPEAKER


class TestOpensmileWrapper(unittest.TestCase):
    """Tests for OpensmileWrapper class."""

    @classmethod
    def setUpClass(cls):
        cls.smile = OpensmileWrapper()
        cls.test_audio_path = AUDIO_MULTISPEAKER

    def test_initialization_default(self):
        """Test OpensmileWrapper initializes with defaults."""
        wrapper = OpensmileWrapper()
        self.assertIsInstance(wrapper, OpensmileWrapper)
        self.assertIsNotNone(wrapper.smile)

    def test_initialization_egemaps_lld(self):
        """Test initialization with egemaps and lld."""
        wrapper = OpensmileWrapper(feature_set="egemaps", feature_level="lld")
        self.assertIsInstance(wrapper, OpensmileWrapper)

    def test_initialization_compare_functionals(self):
        """Test initialization with compare and functionals."""
        wrapper = OpensmileWrapper(feature_set="compare", feature_level="functionals")
        self.assertIsInstance(wrapper, OpensmileWrapper)

    def test_invalid_feature_set_raises_error(self):
        """Test that invalid feature_set raises ValueError."""
        with self.assertRaises(ValueError) as context:
            OpensmileWrapper(feature_set="invalid")
        self.assertIn("Unsupported feature_set", str(context.exception))

    def test_invalid_feature_level_raises_error(self):
        """Test that invalid feature_level raises ValueError."""
        with self.assertRaises(ValueError) as context:
            OpensmileWrapper(feature_level="invalid")
        self.assertIn("Unsupported feature_level", str(context.exception))

    def test_extract_features_from_file_npy(self):
        """Test feature extraction from file with numpy return."""
        feature = self.smile(self.test_audio_path)
        self.assertIsInstance(feature, np.ndarray)
        self.assertEqual(len(feature.shape), 2)

    def test_extract_features_from_file_pt(self):
        """Test feature extraction from file with torch return."""
        feature = self.smile(self.test_audio_path, return_tensors="pt")
        self.assertIsInstance(feature, torch.Tensor)
        self.assertEqual(len(feature.shape), 2)

    def test_extract_features_from_numpy_array(self):
        """Test feature extraction from numpy array."""
        waveform = np.random.rand(16000).astype(np.float32)
        feature = self.smile(waveform, sr=16000)
        self.assertIsInstance(feature, np.ndarray)
        self.assertEqual(len(feature.shape), 2)

    def test_extract_features_from_torch_tensor(self):
        """Test feature extraction from torch tensor."""
        waveform = torch.rand(16000)
        feature = self.smile(waveform, sr=16000)
        self.assertIsInstance(feature, np.ndarray)
        self.assertEqual(len(feature.shape), 2)

    def test_extract_features_return_torch(self):
        """Test feature extraction with torch tensor return."""
        waveform = np.random.rand(16000).astype(np.float32)
        feature = self.smile(waveform, sr=16000, return_tensors="pt")
        self.assertIsInstance(feature, torch.Tensor)

    def test_audio_to_feature_from_file(self):
        """Test audio_to_feature method with real file."""
        feature = self.smile.audio_to_feature(self.test_audio_path)
        self.assertIsInstance(feature, np.ndarray)
        self.assertEqual(len(feature.shape), 2)

    def test_different_sample_rates(self):
        """Test feature extraction with different sample rates."""
        for sr in [8000, 16000, 22050, 44100]:
            with self.subTest(sample_rate=sr):
                waveform = np.random.rand(sr).astype(np.float32)
                feature = self.smile(waveform, sr=sr)
                self.assertIsInstance(feature, np.ndarray)

    def test_file_vs_tensor_consistency(self):
        """Test that features from file path match features from loaded tensor."""
        feature_from_file = self.smile(self.test_audio_path)
        waveform, sr = load_audio(
            self.test_audio_path,
            target_sample_rate=None,
            clamp=False,
            mono=False,
            squeeze=False,
        )
        feature_from_tensor = self.smile(waveform, sr=sr)
        np.testing.assert_array_equal(feature_from_file, feature_from_tensor)


if __name__ == "__main__":
    unittest.main()
