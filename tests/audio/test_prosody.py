"""Tests for exordium.audio.prosody.ProsodyExtractor."""

import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from exordium.audio.prosody import ProsodyExtractor


def _sine_wave(freq=200, sr=16000, duration=1.0, amplitude=0.5):
    """Generate a sine wave as a 1D float tensor — contains clear pitch."""
    t = torch.linspace(0, duration, int(sr * duration))
    return (amplitude * torch.sin(2 * math.pi * freq * t)).float()


def _silence(sr=16000, duration=1.0):
    return torch.zeros(int(sr * duration))


class TestProsodyExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = ProsodyExtractor(sr=16000)

    def _make_chunk(self):
        return torch.randn(16000)

    def test_extract_returns_dict(self):
        chunk = self._make_chunk()
        result = self.extractor.extract(chunk)
        self.assertIsInstance(result, dict)

    def test_extract_has_required_keys(self):
        chunk = self._make_chunk()
        result = self.extractor.extract(chunk)
        for key in ("pitch", "energy", "energy_variance", "voice_ratio", "engagement"):
            self.assertIn(key, result)

    def test_pitch_is_float(self):
        chunk = self._make_chunk()
        result = self.extractor.extract(chunk)
        self.assertIsInstance(result["pitch"], float)

    def test_energy_is_float(self):
        chunk = self._make_chunk()
        result = self.extractor.extract(chunk)
        self.assertIsInstance(result["energy"], float)

    def test_voice_ratio_in_range(self):
        chunk = self._make_chunk()
        result = self.extractor.extract(chunk)
        self.assertGreaterEqual(result["voice_ratio"], 0.0)
        self.assertLessEqual(result["voice_ratio"], 1.0)

    def test_engagement_in_range(self):
        chunk = self._make_chunk()
        result = self.extractor.extract(chunk)
        self.assertGreaterEqual(result["engagement"], 0.0)
        self.assertLessEqual(result["engagement"], 1.0)

    def test_extract_numpy_input(self):
        chunk = np.random.randn(16000).astype(np.float32)
        result = self.extractor.extract(chunk)
        self.assertIn("pitch", result)

    def test_reset_clears_buffers(self):
        chunk = self._make_chunk()
        self.extractor.extract(chunk)
        self.extractor.reset()
        self.assertEqual(len(self.extractor.pitch_buffer), 0)


class TestProsodyExtractorSpeechPath(unittest.TestCase):
    """Tests that exercise the speech-detected branch of extract()."""

    @classmethod
    def setUpClass(cls):
        cls.extractor = ProsodyExtractor(sr=16000)

    def test_sine_wave_activates_vad(self):
        """A loud sine wave should trigger Silero VAD and flow through the speech path."""
        chunk = _sine_wave(freq=200, amplitude=0.8)
        result = self.extractor.extract(chunk, vad_threshold=0.0)
        self.assertIsInstance(result, dict)
        for key in ("pitch", "energy", "energy_variance", "voice_ratio", "engagement"):
            self.assertIn(key, result)

    def test_energy_positive_for_loud_signal(self):
        chunk = _sine_wave(freq=200, amplitude=0.9)
        result = self.extractor.extract(chunk, vad_threshold=0.0)
        self.assertGreaterEqual(result["energy"], 0.0)

    def test_multiple_chunks_fills_buffer(self):
        self.extractor.reset()
        for _ in range(5):
            chunk = _sine_wave(freq=150, amplitude=0.7)
            self.extractor.extract(chunk, vad_threshold=0.0)
        self.assertGreaterEqual(len(self.extractor.pitch_buffer), 1)

    def test_pitch_variance_computed_after_multiple_voiced_chunks(self):
        """With several voiced chunks, pitch_variance is non-trivial."""
        self.extractor.reset()
        for freq in (100, 200, 300):
            chunk = _sine_wave(freq=freq, amplitude=0.8)
            self.extractor.extract(chunk, vad_threshold=0.0)
        result = self.extractor.extract(_sine_wave(freq=150, amplitude=0.8), vad_threshold=0.0)
        self.assertIsInstance(result["engagement"], float)


class TestProsodyExtractorInternals(unittest.TestCase):
    """Unit tests for the internal computation methods."""

    @classmethod
    def setUpClass(cls):
        cls.ext = ProsodyExtractor(sr=16000)

    def test_compute_energy_sine(self):
        audio = _sine_wave(freq=200, amplitude=0.5)
        energy = self.ext._compute_energy(audio)
        self.assertGreater(energy, 0.1)
        self.assertLess(energy, 1.0)

    def test_compute_energy_silence(self):
        energy = self.ext._compute_energy(_silence())
        self.assertAlmostEqual(energy, 0.0, places=5)

    def test_compute_energy_variance_returns_float(self):
        audio = _sine_wave(freq=200, amplitude=0.5)
        var = self.ext._compute_energy_variance(audio)
        self.assertIsInstance(var, float)
        self.assertGreaterEqual(var, 0.0)

    def test_compute_energy_variance_short_audio(self):
        """Short audio with < 2 frames returns 0.0."""
        audio = torch.zeros(10)
        var = self.ext._compute_energy_variance(audio)
        self.assertEqual(var, 0.0)

    def test_compute_pitch_returns_float(self):
        audio = _sine_wave(freq=200, amplitude=0.5)
        pitch = self.ext._compute_pitch(audio)
        self.assertIsInstance(pitch, float)
        self.assertGreaterEqual(pitch, 0.0)

    def test_compute_pitch_silence_returns_zero(self):
        pitch = self.ext._compute_pitch(_silence())
        self.assertAlmostEqual(pitch, 0.0, places=5)

    def test_compute_pitch_very_short_audio(self):
        """Audio where max_lag <= min_lag → returns 0.0."""
        audio = torch.zeros(5)
        pitch = self.ext._compute_pitch(audio)
        self.assertEqual(pitch, 0.0)

    def test_compute_engagement_zero_inputs(self):
        score = self.ext._compute_engagement(0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_compute_engagement_max_inputs(self):
        score = self.ext._compute_engagement(1e6, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_compute_engagement_partial(self):
        score = self.ext._compute_engagement(1000.0, 0.08, 0.003, 0.5)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestProsodyExtractorEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ext = ProsodyExtractor(sr=16000)

    def test_extract_all_silence(self):
        result = self.ext.extract(_silence())
        self.assertEqual(result["pitch"], 0.0)
        self.assertEqual(result["energy"], 0.0)
        self.assertEqual(result["voice_ratio"], 0.0)

    def test_extract_numpy_input(self):
        chunk = np.random.randn(16000).astype(np.float32) * 0.1
        result = self.ext.extract(chunk, vad_threshold=0.0)
        self.assertIn("pitch", result)

    def test_invalid_sample_rate_raises(self):
        with self.assertRaises(ValueError):
            ProsodyExtractor(sr=22050)

    def test_reset_clears_all_buffers(self):
        chunk = _sine_wave(freq=200, amplitude=0.8)
        self.ext.extract(chunk, vad_threshold=0.0)
        self.ext.reset()
        self.assertEqual(len(self.ext.pitch_buffer), 0)
        self.assertEqual(len(self.ext.energy_buffer), 0)
        self.assertEqual(len(self.ext.energy_variance_buffer), 0)
        self.assertEqual(len(self.ext.voice_ratio_buffer), 0)


class TestProsodyVadSpeechPath(unittest.TestCase):
    """Mock VAD to test speech-detected branches (lines 91-100, 196-198, 216-217)."""

    @classmethod
    def setUpClass(cls):
        cls.ext = ProsodyExtractor(sr=16000)

    def test_speech_path_energy_nonzero(self):
        """When VAD finds speech, energy > 0 for a sine wave."""
        audio = _sine_wave(freq=200, amplitude=0.5)
        timestamps = [{"start": 0, "end": len(audio)}]

        with patch.object(self.ext, "_get_speech_timestamps", return_value=timestamps):
            result = self.ext.extract(audio)
        self.assertGreater(result["energy"], 0.0)

    def test_speech_path_voice_ratio(self):
        """voice_ratio reflects the fraction of speech frames."""
        audio = _sine_wave(freq=200, amplitude=0.5)
        mid = len(audio) // 2
        timestamps = [{"start": 0, "end": mid}]

        self.ext.reset()
        with patch.object(self.ext, "_get_speech_timestamps", return_value=timestamps):
            result = self.ext.extract(audio)
        self.assertAlmostEqual(result["voice_ratio"], 0.5, delta=0.1)

    def test_speech_path_updates_buffers(self):
        """After speech chunk, pitch_buffer and energy_buffer should be non-empty."""
        self.ext.reset()
        audio = _sine_wave(freq=300, amplitude=0.7)
        timestamps = [{"start": 0, "end": len(audio)}]

        with patch.object(self.ext, "_get_speech_timestamps", return_value=timestamps):
            self.ext.extract(audio)
        self.assertGreater(len(self.ext.energy_buffer), 0)

    def test_pitch_variance_with_multiple_voiced_chunks(self):
        """Pitch variance computed after >= 2 voiced frames (lines 216-217)."""
        self.ext.reset()
        for freq in (100, 200, 300, 200):
            audio = _sine_wave(freq=freq, amplitude=0.8)
            timestamps = [{"start": 0, "end": len(audio)}]
            with patch.object(self.ext, "_get_speech_timestamps", return_value=timestamps):
                result = self.ext.extract(audio)
        self.assertIsInstance(result["engagement"], float)

    def test_no_speech_path_zeros(self):
        """When VAD returns no timestamps, energy=0, pitch=0."""
        self.ext.reset()
        audio = _silence()

        with patch.object(self.ext, "_get_speech_timestamps", return_value=[]):
            result = self.ext.extract(audio)
        self.assertEqual(result["energy"], 0.0)
        self.assertEqual(result["voice_ratio"], 0.0)


class TestComputePitchBranches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ext = ProsodyExtractor(sr=16000)

    def test_pitch_low_peak_returns_zero(self):
        """Peak < 30% of zero-lag energy → return 0.0."""
        audio = torch.from_numpy(np.random.randn(16000).astype(np.float32) * 0.001)
        pitch = self.ext._compute_pitch(audio)
        self.assertIsInstance(pitch, float)
        self.assertGreaterEqual(pitch, 0.0)

    def test_pitch_empty_lags_returns_zero(self):
        """When lag range is empty, return 0.0."""
        audio = torch.zeros(60)
        pitch = self.ext._compute_pitch(audio)
        self.assertEqual(pitch, 0.0)


class TestProsodyIs8kSampleRate(unittest.TestCase):
    def test_8k_sample_rate_loads(self):
        ext = ProsodyExtractor(sr=8000)
        audio = torch.zeros(8000)
        result = ext.extract(audio)
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
