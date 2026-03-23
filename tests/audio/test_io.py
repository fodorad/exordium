"""Tests for exordium.audio.io: load_audio, save_audio, split_audio, AudioLoader."""

import tempfile
import unittest
from pathlib import Path

import torch

from exordium.audio.io import AudioLoader, load_audio, save_audio, split_audio
from tests.fixtures import AUDIO_MULTISPEAKER


class TestLoadAudio(unittest.TestCase):
    def test_returns_tuple(self):
        waveform, sr = load_audio(AUDIO_MULTISPEAKER)
        self.assertIsInstance(waveform, torch.Tensor)
        self.assertIsInstance(sr, int)

    def test_mono_conversion(self):
        waveform, sr = load_audio(AUDIO_MULTISPEAKER, mono=True)
        self.assertEqual(waveform.ndim, 1)

    def test_resampling(self):
        waveform, sr = load_audio(AUDIO_MULTISPEAKER, target_sample_rate=8000)
        self.assertEqual(sr, 8000)

    def test_no_resampling(self):
        waveform, sr = load_audio(AUDIO_MULTISPEAKER, target_sample_rate=None)
        self.assertIsInstance(sr, int)

    def test_file_not_found(self):
        with self.assertRaises(Exception):
            load_audio("/nonexistent/path/audio.wav")


class TestSaveAudio(unittest.TestCase):
    def test_save_and_reload(self):
        waveform, sr = load_audio(AUDIO_MULTISPEAKER)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = Path(f.name)
        try:
            save_audio(waveform, out_path, sr=sr)
            self.assertTrue(out_path.exists())
            loaded, loaded_sr = load_audio(out_path, target_sample_rate=None)
            self.assertIsInstance(loaded, torch.Tensor)
        finally:
            out_path.unlink(missing_ok=True)

    def test_save_1d_tensor(self):
        audio = torch.zeros(16000)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = Path(f.name)
        try:
            save_audio(audio, out_path, sr=16000)
            self.assertTrue(out_path.exists())
        finally:
            out_path.unlink(missing_ok=True)


class TestSplitAudio(unittest.TestCase):
    def test_splits_into_segments(self):
        audio = torch.zeros(32000)
        segments = split_audio(audio, segment_duration=1.0, sample_rate=16000)
        self.assertEqual(len(segments), 2)

    def test_segment_shape(self):
        audio = torch.zeros(16000)
        segments = split_audio(audio, segment_duration=0.5, sample_rate=16000)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].shape[-1], 8000)

    def test_last_segment_may_be_shorter(self):
        audio = torch.zeros(17000)
        segments = split_audio(audio, segment_duration=1.0, sample_rate=16000)
        self.assertEqual(len(segments), 2)
        self.assertLess(segments[-1].shape[-1], 16000)


class TestAudioLoader(unittest.TestCase):
    def setUp(self):
        self.loader = AudioLoader()

    def test_load_returns_tuple(self):
        waveform, sr = self.loader.load_audio(AUDIO_MULTISPEAKER)
        self.assertIsInstance(waveform, torch.Tensor)
        self.assertIsInstance(sr, int)

    def test_caching(self):
        waveform1, sr1 = self.loader.load_audio(AUDIO_MULTISPEAKER)
        waveform2, sr2 = self.loader.load_audio(AUDIO_MULTISPEAKER)
        self.assertEqual(sr1, sr2)
        self.assertTrue(torch.equal(waveform1, waveform2))

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.loader.load_audio("/nonexistent/audio.wav")

    def test_start_end_slicing(self):
        waveform_full, sr = self.loader.load_audio(AUDIO_MULTISPEAKER)
        duration = waveform_full.shape[-1] / sr
        if duration >= 1.0:
            waveform_slice, _ = self.loader.load_audio(
                AUDIO_MULTISPEAKER,
                start_time_sec=0.0,
                end_time_sec=min(1.0, duration),
            )
            self.assertLessEqual(waveform_slice.shape[-1], waveform_full.shape[-1])

    def test_out_of_bounds_raises_value_error(self):
        waveform, sr = self.loader.load_audio(AUDIO_MULTISPEAKER)
        duration = waveform.shape[-1] / sr
        with self.assertRaises(ValueError):
            self.loader.load_audio(
                AUDIO_MULTISPEAKER,
                start_time_sec=0.0,
                end_time_sec=duration + 100.0,
            )


class TestAudioLoaderEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = AudioLoader()

    def test_batch_dim_adds_batch_dimension(self):
        """batch_dim=True triggers line 163: waveform = torch.unsqueeze(waveform, dim=0)."""
        waveform, sr = self.loader.load_audio(AUDIO_MULTISPEAKER, batch_dim=True)
        self.assertEqual(waveform.ndim, 2)
        self.assertEqual(waveform.shape[0], 1)

    def test_end_time_before_start_raises(self):
        """end_time_sec < start_time_sec → line 155: raise ValueError."""
        with self.assertRaises(ValueError):
            self.loader.load_audio(
                AUDIO_MULTISPEAKER,
                start_time_sec=2.0,
                end_time_sec=1.0,
            )


if __name__ == "__main__":
    unittest.main()
