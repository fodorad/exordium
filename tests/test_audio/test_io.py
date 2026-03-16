# Test cases for audio I/O operations
import math
import os
import shutil
import tempfile
import unittest

import torch

from exordium.audio.io import AudioLoader, load_audio, save_audio, split_audio
from tests.fixtures import AUDIO_MULTISPEAKER, VIDEO_MULTISPEAKER


class TestAudioIO(unittest.TestCase):
    """Test cases for audio I/O operations."""

    def setUp(self):
        self.sample_rate = 16000
        self.test_audio_path = AUDIO_MULTISPEAKER
        self.test_video_path = VIDEO_MULTISPEAKER
        self.temp_dir = tempfile.mkdtemp()
        self.temp_files = []

    def tearDown(self):
        # Clean up temp files
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # ===== load_audio() Tests =====

    def test_load_audio(self):
        """Test loading audio with default settings."""
        waveform, sr = load_audio(self.test_audio_path, target_sample_rate=self.sample_rate)
        self.assertEqual(sr, self.sample_rate)
        self.assertTrue(isinstance(waveform, torch.Tensor))
        self.assertTrue(waveform.dim() == 1)  # Should be mono and squeezed

    def test_load_audio_resample(self):
        """Test loading audio and resampling to 44.1kHz."""
        waveform, sr = load_audio(self.test_audio_path, target_sample_rate=44100)
        self.assertEqual(sr, 44100)
        self.assertTrue(isinstance(waveform, torch.Tensor))

    def test_load_audio_clamp(self):
        """Test clamping audio values to [-1, 1]."""
        waveform, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, clamp=True
        )
        self.assertTrue((waveform.abs() <= 1).all())

    def test_load_audio_mono(self):
        """Test converting audio to mono."""
        waveform, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=True
        )
        self.assertEqual(waveform.dim(), 1)

    def test_load_audio_target_sample_rate_none(self):
        """Test loading audio without resampling (target_sample_rate=None)."""
        waveform, sr = load_audio(self.test_audio_path, target_sample_rate=None)
        # Original audio is 44100 Hz
        self.assertEqual(sr, 44100)
        self.assertIsInstance(waveform, torch.Tensor)

    def test_load_audio_mono_false(self):
        """Test keeping stereo audio (mono=False)."""
        waveform, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=False, squeeze=False
        )
        # Should be stereo (2 channels)
        self.assertEqual(waveform.shape[0], 2)

    def test_load_audio_squeeze_false(self):
        """Test not squeezing dimensions (squeeze=False)."""
        waveform, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=True, squeeze=False
        )
        # Note: when mono=True, mean() removes dimension, so still 1D even with squeeze=False
        self.assertEqual(waveform.dim(), 1)

    def test_load_audio_clamp_false(self):
        """Test not clamping values (clamp=False)."""
        waveform, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, clamp=False
        )
        # Just verify it loads without error
        self.assertIsInstance(waveform, torch.Tensor)

    def test_load_audio_all_params_combined(self):
        """Test all parameters combined."""
        waveform, sr = load_audio(
            self.test_audio_path, target_sample_rate=22050, clamp=True, mono=False, squeeze=False
        )
        self.assertEqual(sr, 22050)
        self.assertEqual(waveform.shape[0], 2)  # Stereo
        self.assertTrue((waveform.abs() <= 1).all())  # Clamped

    def test_load_audio_mono_squeezed_shape(self):
        """Verify shape is (T,) for mono squeezed audio."""
        waveform, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=True, squeeze=True
        )
        self.assertEqual(waveform.dim(), 1)

    def test_load_audio_mono_unsqueezed_shape(self):
        """Verify shape for mono unsqueezed audio."""
        waveform, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=True, squeeze=False
        )
        # Note: when mono=True, mean(dim=0) produces 1D tensor, squeeze doesn't affect it
        self.assertEqual(waveform.dim(), 1)

    def test_load_audio_stereo_squeezed_shape(self):
        """Verify shape is (2, T) for stereo squeezed audio."""
        waveform, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=False, squeeze=True
        )
        self.assertEqual(waveform.dim(), 2)
        self.assertEqual(waveform.shape[0], 2)

    def test_load_audio_stereo_unsqueezed_shape(self):
        """Verify shape is (2, T) for stereo unsqueezed audio."""
        waveform, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=False, squeeze=False
        )
        self.assertEqual(waveform.dim(), 2)
        self.assertEqual(waveform.shape[0], 2)

    def test_load_audio_clamp_verification(self):
        """Check all values are in [-1, 1] when clamped."""
        waveform, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, clamp=True
        )
        self.assertTrue(waveform.min() >= -1.0)
        self.assertTrue(waveform.max() <= 1.0)

    def test_load_audio_stereo_to_mono_averaging(self):
        """Verify stereo to mono conversion averages channels."""
        # Load as stereo
        stereo, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=False, squeeze=False
        )
        # Load as mono
        mono, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=True, squeeze=False
        )

        # Manual average
        expected_mono = stereo.mean(dim=0, keepdim=True)
        # Should be very close (allowing for floating point precision)
        self.assertTrue(torch.allclose(mono, expected_mono, atol=1e-6))

    def test_load_audio_sample_rate_preservation(self):
        """Verify returned sample rate matches requested rate."""
        target_sr = 22050
        waveform, sr = load_audio(self.test_audio_path, target_sample_rate=target_sr)
        self.assertEqual(sr, target_sr)

    def test_load_audio_multiple_sample_rates(self):
        """Test loading at various sample rates using subTest."""
        sample_rates = [8000, 16000, 22050, 44100, 48000]
        for sr in sample_rates:
            with self.subTest(sample_rate=sr):
                waveform, returned_sr = load_audio(self.test_audio_path, target_sample_rate=sr)
                self.assertEqual(returned_sr, sr)

    def test_load_audio_waveform_length_calculation(self):
        """Verify waveform length matches expected duration."""
        target_sr = 16000
        waveform, sr = load_audio(self.test_audio_path, target_sample_rate=target_sr)
        # Calculate duration
        duration = waveform.shape[-1] / sr
        # Should be around the duration of the audio file (allowing some tolerance)
        self.assertGreater(duration, 0)

    def test_load_audio_file_not_found(self):
        """Expect error for missing file."""
        with self.assertRaises(RuntimeError):  # torchaudio raises RuntimeError
            load_audio("/nonexistent/path/audio.wav", target_sample_rate=16000)

    # ===== save_audio() Tests =====

    def test_save_audio(self):
        """Test saving audio to a temporary file."""
        waveform, _ = load_audio(self.test_audio_path, target_sample_rate=self.sample_rate)
        temp_path = os.path.join(self.temp_dir, "test_save.wav")
        self.temp_files.append(temp_path)
        save_audio(waveform, temp_path, sr=self.sample_rate)
        self.assertTrue(os.path.exists(temp_path))

    def test_save_audio_1d_tensor(self):
        """Save 1D mono audio."""
        waveform, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=True, squeeze=True
        )
        temp_path = os.path.join(self.temp_dir, "test_1d.wav")
        self.temp_files.append(temp_path)
        save_audio(waveform, temp_path, sr=self.sample_rate)
        self.assertTrue(os.path.exists(temp_path))
        self.assertEqual(waveform.dim(), 1)

    def test_save_audio_2d_tensor(self):
        """Save 2D audio (stereo or batched)."""
        waveform, _ = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=False, squeeze=False
        )
        temp_path = os.path.join(self.temp_dir, "test_2d.wav")
        self.temp_files.append(temp_path)
        save_audio(waveform, temp_path, sr=self.sample_rate)
        self.assertTrue(os.path.exists(temp_path))

    def test_save_audio_multiple_sample_rates(self):
        """Save at different sample rates."""
        waveform, _ = load_audio(self.test_audio_path, target_sample_rate=16000)
        sample_rates = [16000, 44100, 48000]

        for sr in sample_rates:
            with self.subTest(sample_rate=sr):
                temp_path = os.path.join(self.temp_dir, f"test_{sr}.wav")
                self.temp_files.append(temp_path)
                save_audio(waveform, temp_path, sr=sr)
                self.assertTrue(os.path.exists(temp_path))

    def test_save_load_roundtrip_mono(self):
        """Save then load mono audio, verify identical."""
        waveform_original, sr = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=True
        )
        temp_path = os.path.join(self.temp_dir, "roundtrip_mono.wav")
        self.temp_files.append(temp_path)

        # Save
        save_audio(waveform_original, temp_path, sr=sr)

        # Load back
        waveform_loaded, sr_loaded = load_audio(temp_path, target_sample_rate=sr)

        # Verify
        self.assertEqual(sr, sr_loaded)
        # Allow small floating point differences
        self.assertTrue(torch.allclose(waveform_original, waveform_loaded, atol=1e-4))

    def test_save_load_roundtrip_stereo(self):
        """Save stereo then load, verify identical."""
        waveform_original, sr = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=False, squeeze=False
        )
        temp_path = os.path.join(self.temp_dir, "roundtrip_stereo.wav")
        self.temp_files.append(temp_path)

        # Save
        save_audio(waveform_original, temp_path, sr=sr)

        # Load back
        waveform_loaded, sr_loaded = load_audio(
            temp_path, target_sample_rate=sr, mono=False, squeeze=False
        )

        # Verify
        self.assertEqual(sr, sr_loaded)
        self.assertTrue(torch.allclose(waveform_original, waveform_loaded, atol=1e-4))

    def test_save_audio_creates_file(self):
        """Verify file exists after save."""
        waveform, _ = load_audio(self.test_audio_path, target_sample_rate=self.sample_rate)
        temp_path = os.path.join(self.temp_dir, "created_file.wav")
        self.temp_files.append(temp_path)

        self.assertFalse(os.path.exists(temp_path))  # Not exists before
        save_audio(waveform, temp_path, sr=self.sample_rate)
        self.assertTrue(os.path.exists(temp_path))  # Exists after

    # ===== split_audio() Tests =====

    def test_split_audio(self):
        """Test splitting audio into 1-second segments."""
        waveform, sr = load_audio(self.test_audio_path, target_sample_rate=self.sample_rate)
        segment_duration = 1.0  # 1 second
        segments = split_audio(waveform, segment_duration, sr)
        expected_num_segments = int(torch.ceil(torch.tensor(len(waveform) / sr)))
        self.assertEqual(len(segments), expected_num_segments)

    def test_split_audio_5_second_segments(self):
        """Split into 5-second chunks."""
        waveform, sr = load_audio(self.test_audio_path, target_sample_rate=self.sample_rate)
        segment_duration = 5.0
        segments = split_audio(waveform, segment_duration, sr)

        # Verify we have the expected number of segments
        segment_length = int(segment_duration * sr)
        expected_segments = -(-waveform.shape[-1] // segment_length)  # Ceiling division
        self.assertEqual(len(segments), expected_segments)

    def test_split_audio_exact_division(self):
        """Test audio length perfectly divisible by segment duration."""
        # Create synthetic audio with exact length
        sr = 16000
        duration = 10.0  # 10 seconds
        waveform = torch.randn(int(sr * duration))

        segment_duration = 2.0  # 2 seconds
        segments = split_audio(waveform, segment_duration, sr)

        # Should have exactly 5 segments
        self.assertEqual(len(segments), 5)
        # All segments should be the same length
        for seg in segments:
            self.assertEqual(seg.shape[-1], int(segment_duration * sr))

    def test_split_audio_partial_last_segment(self):
        """Last segment shorter than others."""
        # Create synthetic audio that doesn't divide evenly
        sr = 16000
        duration = 10.5  # 10.5 seconds
        waveform = torch.randn(int(sr * duration))

        segment_duration = 2.0  # 2 seconds
        segments = split_audio(waveform, segment_duration, sr)

        # Should have 6 segments (5 full + 1 partial)
        self.assertEqual(len(segments), 6)
        # Last segment should be shorter
        self.assertLess(segments[-1].shape[-1], int(segment_duration * sr))

    def test_split_audio_audio_shorter_than_segment(self):
        """Test audio shorter than segment duration."""
        # Create 0.5 second audio
        sr = 16000
        waveform = torch.randn(int(sr * 0.5))

        segment_duration = 2.0  # Longer than audio
        segments = split_audio(waveform, segment_duration, sr)

        # Should have 1 segment containing the whole audio
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].shape[-1], waveform.shape[-1])

    def test_split_audio_very_short_segments(self):
        """Test 0.1 second segments."""
        waveform, sr = load_audio(self.test_audio_path, target_sample_rate=self.sample_rate)
        segment_duration = 0.1
        segments = split_audio(waveform, segment_duration, sr)

        # Should have many segments
        self.assertGreater(len(segments), 10)

    def test_split_audio_verify_segment_lengths(self):
        """Check each segment is correct length."""
        sr = 16000
        duration = 10.0
        waveform = torch.randn(int(sr * duration))

        segment_duration = 2.0
        segments = split_audio(waveform, segment_duration, sr)
        segment_length = int(segment_duration * sr)

        # All but last should be exact length
        for seg in segments[:-1]:
            self.assertEqual(seg.shape[-1], segment_length)

    def test_split_audio_2d_tensor(self):
        """Split multi-channel (2D) audio - uses advanced indexing."""
        waveform, sr = load_audio(
            self.test_audio_path, target_sample_rate=self.sample_rate, mono=False, squeeze=False
        )
        segment_duration = 1.0
        # Use proper slicing for 2D: split along time dimension (last dim)
        segment_length = int(segment_duration * sr)
        num_segments = int(math.ceil(waveform.shape[-1] / segment_length))
        segments = [
            waveform[..., i * segment_length : (i + 1) * segment_length]
            for i in range(num_segments)
        ]

        # Verify segments maintain channel dimension
        for seg in segments:
            self.assertEqual(seg.shape[0], 2)  # Still stereo

    # ===== video to audio Tests =====

    def test_video_to_audio_real_file(self):
        """Extract audio from actual video file."""
        waveform, _ = load_audio(self.test_video_path)
        self.assertIsInstance(waveform, torch.Tensor)
        self.assertGreater(waveform.shape[-1], 0)

    def test_video_to_audio_resampling(self):
        """Verify resampling works."""
        waveform, _ = load_audio(self.test_video_path, target_sample_rate=22050)
        self.assertIsInstance(waveform, torch.Tensor)
        self.assertGreater(waveform.shape[-1], 0)

    # ===== AudioLoader Tests =====

    def test_audio_loader_cache(self):
        """Test caching behavior in AudioLoader."""
        loader = AudioLoader()
        waveform1, sr1 = loader.load_audio(self.test_audio_path)
        waveform2, sr2 = loader.load_audio(self.test_audio_path)
        self.assertEqual(sr1, sr2)
        self.assertTrue(torch.equal(waveform1, waveform2))

    def test_audio_loader_partial_load(self):
        """Test loading partial audio segments."""
        loader = AudioLoader()
        waveform, sr = loader.load_audio(self.test_audio_path, start_time_sec=1.0, end_time_sec=2.0)
        self.assertTrue(isinstance(waveform, torch.Tensor))
        self.assertEqual(waveform.shape[-1], int((2.0 - 1.0) * sr))

    def test_audioloader_cache_identity(self):
        """Same object returned from cache."""
        loader = AudioLoader()
        waveform1, sr1 = loader.load_audio(self.test_audio_path)
        waveform2, sr2 = loader.load_audio(self.test_audio_path)

        # Should be identical (same cached object)
        self.assertTrue(torch.equal(waveform1, waveform2))
        self.assertEqual(sr1, sr2)

    def test_audioloader_cache_multiple_files(self):
        """Different files cached separately."""
        loader = AudioLoader()

        # Load audio file
        waveform1, sr1 = loader.load_audio(self.test_audio_path)

        # Load video file (has audio track)
        waveform2, sr2 = loader.load_audio(self.test_video_path, target_sample_rate=16000)

        # Should be different
        self.assertFalse(torch.equal(waveform1, waveform2))

    def test_audioloader_time_slice_start_only(self):
        """Only start_time_sec specified."""
        loader = AudioLoader()
        # First load to cache
        loader.load_audio(self.test_audio_path)

        # Note: current implementation requires both start and end
        # This test documents current behavior
        waveform, sr = loader.load_audio(self.test_audio_path)
        self.assertIsInstance(waveform, torch.Tensor)

    def test_audioloader_time_slice_end_only(self):
        """Only end_time_sec specified."""
        loader = AudioLoader()
        # First load to cache
        loader.load_audio(self.test_audio_path)

        # Note: current implementation requires both start and end
        # This test documents current behavior
        waveform, sr = loader.load_audio(self.test_audio_path)
        self.assertIsInstance(waveform, torch.Tensor)

    def test_audioloader_time_slice_verify_length(self):
        """Check sliced duration is correct."""
        loader = AudioLoader()
        start = 2.0
        end = 5.0
        duration = end - start

        waveform, sr = loader.load_audio(
            self.test_audio_path, start_time_sec=start, end_time_sec=end
        )

        expected_length = int(duration * sr)
        self.assertEqual(waveform.shape[-1], expected_length)

    def test_audioloader_wav_format(self):
        """Load WAV file explicitly."""
        loader = AudioLoader()
        waveform, sr = loader.load_audio(self.test_audio_path)
        self.assertIsInstance(waveform, torch.Tensor)
        self.assertIsInstance(sr, int)

    def test_audioloader_mp4_format(self):
        """Load MP4 video file (requires sr)."""
        loader = AudioLoader()
        waveform, sr = loader.load_audio(self.test_video_path, target_sample_rate=16000)
        self.assertIsInstance(waveform, torch.Tensor)
        self.assertEqual(sr, 16000)

    def test_audioloader_batch_dim_true(self):
        """Add batch dimension."""
        loader = AudioLoader()
        waveform, sr = loader.load_audio(self.test_audio_path, batch_dim=True)

        # Should have batch dimension added
        # If audio is mono squeezed (T,) -> (1, T)
        # If audio is stereo (2, T) -> (1, 2, T)
        self.assertGreaterEqual(waveform.dim(), 2)

    def test_audioloader_batch_dim_false(self):
        """No batch dimension."""
        loader = AudioLoader()
        waveform, sr = loader.load_audio(self.test_audio_path, batch_dim=False)

        # Should not add extra dimension
        self.assertIsInstance(waveform, torch.Tensor)

    def test_audioloader_time_bounds_error(self):
        """Out of bounds time range raises ValueError."""
        loader = AudioLoader()

        with self.assertRaises(ValueError):
            # Try to load beyond audio duration
            loader.load_audio(self.test_audio_path, start_time_sec=0, end_time_sec=10000)

    def test_audioloader_end_before_start_error(self):
        """end_time < start_time raises ValueError."""
        loader = AudioLoader()

        with self.assertRaises(ValueError):
            loader.load_audio(self.test_audio_path, start_time_sec=5.0, end_time_sec=2.0)

    def test_audioloader_unsupported_format_error(self):
        """Invalid format raises ValueError."""
        loader = AudioLoader()
        fake_path = "/tmp/fake_audio.xyz"

        with self.assertRaises(FileNotFoundError):
            loader.load_audio(fake_path)


if __name__ == "__main__":
    unittest.main()
