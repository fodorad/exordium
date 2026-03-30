"""Tests for exordium.audio.base.AudioModelWrapper utility methods."""

import unittest

import numpy as np
import torch

from exordium.audio.base import AudioModelWrapper


class _MinimalWrapper(AudioModelWrapper):
    """Minimal concrete subclass to test AudioModelWrapper non-abstract methods."""

    def audio_to_feature(self, audio, **kwargs):
        return None

    def batch_audio_to_features(self, audios, **kwargs):
        return None

    def inference(self, waveform):
        return None


class TestAudioModelWrapperPadWaveforms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = _MinimalWrapper(device_id=-1)

    def test_pad_waveforms_pads_to_same_length(self):
        wv1 = torch.zeros(100)
        wv2 = torch.zeros(200)
        padded, lengths = AudioModelWrapper._pad_waveforms([wv1, wv2])
        self.assertEqual(padded.shape[1], 200)
        self.assertEqual(lengths, [100, 200])

    def test_pad_waveforms_same_length(self):
        wv1 = torch.zeros(100)
        wv2 = torch.zeros(100)
        padded, lengths = AudioModelWrapper._pad_waveforms([wv1, wv2])
        self.assertEqual(padded.shape, (2, 100))

    def test_prepare_waveform_returns_tensor(self):
        audio = torch.zeros(16000)
        result = self.wrapper._prepare_waveform(audio, sample_rate=16000)
        self.assertIsInstance(result, torch.Tensor)


class TestPrepareWaveformBranches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = _MinimalWrapper(device_id=-1)

    def test_stereo_numpy_takes_first_channel(self):
        """2D (C, T) numpy array → line 113: waveform = waveform[0, :]."""
        stereo = np.random.randn(2, 1600).astype(np.float32)
        result = self.wrapper._prepare_waveform(stereo, 16000)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(result.shape[0], 1600)

    def test_stereo_tensor_takes_first_channel(self):
        """2D (C, T) torch tensor → line 113: waveform = waveform[0, :]."""
        stereo = torch.randn(2, 1600)
        result = self.wrapper._prepare_waveform(stereo, 16000)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(result.shape[0], 1600)

    def test_invalid_ndim_raises_value_error(self):
        """3D tensor → line 115: raise ValueError."""
        bad = torch.randn(2, 3, 1600)
        with self.assertRaises(ValueError):
            self.wrapper._prepare_waveform(bad, 16000)


if __name__ == "__main__":
    unittest.main()
