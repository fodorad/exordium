import unittest

import numpy as np
import torch

from exordium.audio.io import load_audio
from exordium.audio.prosody import ProsodyExtractor
from tests.fixtures import AUDIO_MULTISPEAKER


class TestProsodyExtractor(unittest.TestCase):
    """Tests for ProsodyExtractor with mandatory Silero VAD."""

    @classmethod
    def setUpClass(cls):
        cls.extractor = ProsodyExtractor(sr=16000)
        cls.waveform, cls.sr = load_audio(AUDIO_MULTISPEAKER, target_sample_rate=16000, clamp=True)

    def setUp(self):
        self.extractor.reset()

    # --- Initialization ---

    def test_initialization_defaults(self):
        ext = ProsodyExtractor()
        self.assertEqual(ext.sr, 16000)
        self.assertEqual(ext.pitch_buffer.maxlen, 50)
        self.assertEqual(len(ext.pitch_buffer), 0)
        self.assertIsNotNone(ext._vad_model)

    def test_initialization_custom_buffer(self):
        ext = ProsodyExtractor(sr=16000, buffer_size=10)
        self.assertEqual(ext.pitch_buffer.maxlen, 10)

    def test_invalid_sample_rate(self):
        with self.assertRaises(ValueError):
            ProsodyExtractor(sr=44100)

    def test_valid_sample_rate_8000(self):
        ext = ProsodyExtractor(sr=8000)
        self.assertEqual(ext.sr, 8000)

    # --- Output structure ---

    def test_extract_returns_expected_keys(self):
        chunk = self.waveform[16000:32000]  # 1s speech chunk
        result = self.extractor.extract(chunk)
        self.assertIsInstance(result, dict)
        self.assertEqual(
            set(result.keys()), {"pitch", "energy", "energy_variance", "voice_ratio", "engagement"}
        )

    def test_extract_values_are_float(self):
        chunk = self.waveform[16000:32000]
        result = self.extractor.extract(chunk)
        for key, value in result.items():
            self.assertIsInstance(value, float, f"{key} should be float")

    # --- Input types ---

    def test_accepts_numpy_input(self):
        chunk = self.waveform[16000:32000].numpy()
        result = self.extractor.extract(chunk)
        self.assertIn("pitch", result)

    def test_accepts_torch_tensor_input(self):
        chunk = self.waveform[16000:32000]
        result = self.extractor.extract(chunk)
        self.assertIn("pitch", result)

    # --- VAD / voice ratio ---

    def test_voice_ratio_silence_zero(self):
        """Silence should have zero voice ratio."""
        silence = torch.zeros(16000)
        result = self.extractor.extract(silence)
        self.assertEqual(result["voice_ratio"], 0.0)

    def test_voice_ratio_speech_positive(self):
        """Real speech should have positive voice ratio."""
        chunk = self.waveform[16000:32000]  # known speech region
        result = self.extractor.extract(chunk)
        self.assertGreater(result["voice_ratio"], 0.0)

    def test_voice_ratio_range(self):
        chunk = self.waveform[16000:32000]
        result = self.extractor.extract(chunk)
        self.assertGreaterEqual(result["voice_ratio"], 0.0)
        self.assertLessEqual(result["voice_ratio"], 1.0)

    # --- Prosody on silence (VAD gates everything to 0) ---

    def test_silence_pitch_zero(self):
        silence = torch.zeros(16000)
        result = self.extractor.extract(silence)
        self.assertEqual(result["pitch"], 0.0)

    def test_silence_energy_zero(self):
        silence = torch.zeros(16000)
        result = self.extractor.extract(silence)
        self.assertEqual(result["energy"], 0.0)

    def test_silence_energy_variance_zero(self):
        silence = torch.zeros(16000)
        result = self.extractor.extract(silence)
        self.assertEqual(result["energy_variance"], 0.0)

    # --- Prosody on speech ---

    def test_speech_energy_positive(self):
        chunk = self.waveform[16000:32000]
        result = self.extractor.extract(chunk)
        self.assertGreater(result["energy"], 0.0)

    def test_speech_pitch_detected(self):
        """Real speech should produce non-zero pitch on voiced segments."""
        results = []
        for start in range(16000, 96000, 16000):
            self.extractor.reset()
            chunk = self.waveform[start : start + 16000]
            results.append(self.extractor.extract(chunk))
        pitches = [r["pitch"] for r in results]
        self.assertTrue(any(p > 0 for p in pitches), "Some speech chunks should detect pitch")

    # --- Engagement ---

    def test_engagement_range(self):
        chunk = self.waveform[16000:32000]
        result = self.extractor.extract(chunk)
        self.assertGreaterEqual(result["engagement"], 0.0)
        self.assertLessEqual(result["engagement"], 1.0)

    def test_engagement_silence_zero(self):
        silence = torch.zeros(16000)
        result = self.extractor.extract(silence)
        self.assertAlmostEqual(result["engagement"], 0.0, places=3)

    def test_engagement_speech_higher_than_silence(self):
        ext1 = ProsodyExtractor(sr=16000)
        ext2 = ProsodyExtractor(sr=16000)
        silence = torch.zeros(16000)
        speech = self.waveform[16000:32000]
        eng_silence = ext1.extract(silence)["engagement"]
        eng_speech = ext2.extract(speech)["engagement"]
        self.assertGreater(eng_speech, eng_silence)

    # --- Buffer and smoothing ---

    def test_buffer_fills_over_calls(self):
        chunk = self.waveform[16000:32000]
        for _ in range(5):
            self.extractor.extract(chunk)
        self.assertEqual(len(self.extractor.pitch_buffer), 5)
        self.assertEqual(len(self.extractor.energy_buffer), 5)
        self.assertEqual(len(self.extractor.energy_variance_buffer), 5)
        self.assertEqual(len(self.extractor.voice_ratio_buffer), 5)

    def test_buffer_respects_maxlen(self):
        ext = ProsodyExtractor(buffer_size=3)
        chunk = self.waveform[16000:32000]
        for _ in range(10):
            ext.extract(chunk)
        self.assertEqual(len(ext.pitch_buffer), 3)

    def test_reset_clears_buffers(self):
        chunk = self.waveform[16000:32000]
        self.extractor.extract(chunk)
        self.assertGreater(len(self.extractor.pitch_buffer), 0)
        self.extractor.reset()
        self.assertEqual(len(self.extractor.pitch_buffer), 0)
        self.assertEqual(len(self.extractor.energy_buffer), 0)
        self.assertEqual(len(self.extractor.energy_variance_buffer), 0)
        self.assertEqual(len(self.extractor.voice_ratio_buffer), 0)

    def test_smoothing_is_running_average(self):
        """Smoothed value should equal the mean of the buffer."""
        ext = ProsodyExtractor(buffer_size=5)
        chunk = self.waveform[16000:32000]
        for _ in range(5):
            result = ext.extract(chunk)
        expected_energy = float(np.mean(ext.energy_buffer))
        expected_energy_variance = float(np.mean(ext.energy_variance_buffer))
        self.assertAlmostEqual(result["energy"], expected_energy, places=10)
        self.assertAlmostEqual(result["energy_variance"], expected_energy_variance, places=10)

    # --- Real audio end-to-end ---

    def test_real_audio_multispeaker(self):
        ext = ProsodyExtractor(sr=self.sr)
        chunk_size = int(0.5 * self.sr)  # 500ms chunks
        results = []
        for start in range(0, len(self.waveform) - chunk_size, chunk_size):
            chunk = self.waveform[start : start + chunk_size]
            result = ext.extract(chunk)
            results.append(result)
        self.assertGreater(len(results), 0)
        voice_ratios = [r["voice_ratio"] for r in results]
        engagements = [r["engagement"] for r in results]
        self.assertTrue(
            any(vr > 0.5 for vr in voice_ratios),
            "Some chunks should detect speech",
        )
        self.assertTrue(
            any(vr == 0.0 for vr in voice_ratios),
            "Some chunks should be silence",
        )
        self.assertTrue(
            all(0.0 <= eng <= 1.0 for eng in engagements),
            "Engagement should always be in [0, 1]",
        )


if __name__ == "__main__":
    unittest.main()
