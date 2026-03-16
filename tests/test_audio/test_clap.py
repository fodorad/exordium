import unittest

import numpy as np
import torch

from exordium.audio.clap import CLAP_SAMPLE_RATE, ClapWrapper
from exordium.audio.io import AudioLoader, load_audio
from tests.fixtures import AUDIO_MULTISPEAKER

DIM = 512  # laion/larger_clap_music_and_speech embedding dimension


class TestClapWrapper(unittest.TestCase):
    """Unit tests for ClapWrapper."""

    @classmethod
    def setUpClass(cls):
        cls.test_audio_path = AUDIO_MULTISPEAKER
        cls.audio_loader = AudioLoader()
        cls.clap = ClapWrapper(device_id=None)  # CPU

    # --- Initialization ---

    def test_initialization(self):
        wrapper = ClapWrapper(device_id=None)
        self.assertIsNotNone(wrapper.model)
        self.assertIsNotNone(wrapper.processor)

    # --- __call__ ---

    def test_call_with_1d_array(self):
        waveform = np.random.randn(CLAP_SAMPLE_RATE).astype(np.float32)
        result = self.clap(waveform)
        self.assertEqual(result.shape, (1, DIM))

    def test_call_with_2d_tensor_batch(self):
        waveform = torch.randn(2, CLAP_SAMPLE_RATE)
        result = self.clap(waveform)
        self.assertEqual(result.shape, (2, DIM))

    def test_call_with_torch_tensor(self):
        waveform = torch.randn(1, CLAP_SAMPLE_RATE)
        result = self.clap(waveform)
        self.assertEqual(result.shape, (1, DIM))

    def test_call_returns_tensor(self):
        waveform = np.random.randn(CLAP_SAMPLE_RATE).astype(np.float32)
        result = self.clap(waveform)
        self.assertIsInstance(result, torch.Tensor)

    def test_call_invalid_shape_raises(self):
        bad = torch.randn(2, 3, 100)
        with self.assertRaises(ValueError):
            self.clap(bad)

    # --- audio_to_feature ---

    def test_audio_to_feature_from_file(self):
        features = self.clap.audio_to_feature(self.test_audio_path)
        self.assertIsInstance(features, torch.Tensor)

    def test_audio_to_feature_shape(self):
        features = self.clap.audio_to_feature(self.test_audio_path)
        self.assertEqual(features.shape, (1, DIM))

    # --- batch_audio_to_features ---

    def test_batch_audio_to_features_from_paths(self):
        results = self.clap.batch_audio_to_features([self.test_audio_path, self.test_audio_path])
        self.assertIsInstance(results, torch.Tensor)
        self.assertEqual(results.shape, (2, DIM))

    def test_batch_audio_to_features_from_waveforms(self):
        waveforms = [torch.rand(CLAP_SAMPLE_RATE) for _ in range(3)]
        results = self.clap.batch_audio_to_features(waveforms)
        self.assertEqual(results.shape, (3, DIM))

    def test_batch_audio_to_features_single(self):
        results = self.clap.batch_audio_to_features([self.test_audio_path])
        self.assertEqual(results.shape, (1, DIM))

    # --- read_audio ---

    def test_read_audio_with_file_returns_tensor(self):
        waveform, sr = self.clap.read_audio(self.test_audio_path)
        self.assertIsInstance(waveform, torch.Tensor)
        self.assertEqual(sr, CLAP_SAMPLE_RATE)

    def test_read_audio_with_tensor_no_resample(self):
        t = torch.randn(CLAP_SAMPLE_RATE)
        waveform, sr = self.clap.read_audio(t, resample=False)
        self.assertTrue(torch.equal(waveform, t))
        self.assertEqual(sr, CLAP_SAMPLE_RATE)

    def test_read_audio_resampling_changes_sr(self):
        t = torch.randn(44100)
        _, sr = self.clap.read_audio(t, sample_rate=44100, resample=True)
        self.assertEqual(sr, CLAP_SAMPLE_RATE)

    # --- Integration ---

    def test_real_audio_pipeline(self):
        waveform, sr = self.audio_loader.load_audio(
            self.test_audio_path,
            start_time_sec=0,
            end_time_sec=3,
            target_sample_rate=CLAP_SAMPLE_RATE,
            mono=True,
            squeeze=True,
            batch_dim=True,
        )
        result = self.clap(waveform, sample_rate=CLAP_SAMPLE_RATE)
        self.assertEqual(result.shape, (1, DIM))

    def test_integration_load_and_clap(self):
        waveform, sr = load_audio(
            self.test_audio_path,
            target_sample_rate=CLAP_SAMPLE_RATE,
            mono=True,
            squeeze=False,
        )
        features = self.clap(waveform, sample_rate=CLAP_SAMPLE_RATE)
        self.assertEqual(features.shape, (1, DIM))

    def test_integration_load_split_process(self):
        from exordium.audio.io import split_audio

        waveform, sr = load_audio(
            self.test_audio_path,
            target_sample_rate=CLAP_SAMPLE_RATE,
            mono=True,
            squeeze=True,
        )
        segments = split_audio(waveform, 3.0, sr)
        for segment in segments[:3]:
            features = self.clap(segment.unsqueeze(0), sample_rate=CLAP_SAMPLE_RATE)
            self.assertEqual(features.shape, (1, DIM))


if __name__ == "__main__":
    unittest.main()
