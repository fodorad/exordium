import os
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio

from exordium.audio.spectrogram import (
    apply_preemphasis,
    compute_deltas,
    compute_melspec,
    compute_mfcc,
    preprocess_audio,
    process_audio_file,
    save_melspec_specshow,
    save_mfcc_specshow,
)
from tests.fixtures import AUDIO_MULTISPEAKER


class TestAudioProcessing(unittest.TestCase):
    """Tests for audio processing and spectrogram functions."""

    def setUp(self):
        self.test_audio_path = AUDIO_MULTISPEAKER
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    # ===== Helper Functions Tests =====

    def test_preprocess_audio_mono_conversion(self):
        """Test mono conversion."""
        waveform = torch.randn(2, 100)  # stereo
        processed, sr = preprocess_audio(waveform, 44100, 44100)
        self.assertEqual(processed.shape[0], 1)  # should be mono

    def test_preprocess_audio_resampling(self):
        """Test resampling."""
        waveform = torch.randn(1, 100)  # mono
        processed, sr = preprocess_audio(waveform, 48000, 44100)
        self.assertEqual(sr, 44100)

    def test_apply_preemphasis_empty(self):
        """Test with empty array."""
        y = np.array([])
        result = apply_preemphasis(y)
        self.assertEqual(len(result), 0)

    def test_apply_preemphasis_single_sample(self):
        """Test with single sample."""
        y = np.array([1.0])
        result = apply_preemphasis(y)
        self.assertEqual(result[0], 1.0)

    def test_apply_preemphasis_multiple_samples(self):
        """Test with multiple samples."""
        y = np.array([1.0, 2.0, 3.0])
        result = apply_preemphasis(y)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[1], 2.0 - 0.97, places=5)

    def test_compute_deltas_basic(self):
        """Test basic delta computation."""
        features = np.random.rand(100, 10)
        deltas = compute_deltas(features)
        self.assertEqual(deltas.shape, features.shape)

    def test_compute_deltas_return_all(self):
        """Test all deltas computation."""
        features = np.random.rand(100, 10)
        orig, delta, delta2 = compute_deltas(features, return_all=True)
        self.assertEqual(orig.shape, features.shape)
        self.assertEqual(delta.shape, features.shape)
        self.assertEqual(delta2.shape, features.shape)

    # ===== MFCC Tests =====

    def test_compute_mfcc_shape(self):
        """Test MFCC computation output shape."""
        waveform, sr = torchaudio.load(self.test_audio_path)
        y = waveform.squeeze().numpy()

        mfcc, mfcc_preemph = compute_mfcc(y, sr, n_mfcc=40)

        self.assertEqual(mfcc.shape[0], 40)
        self.assertEqual(mfcc_preemph.shape[0], 40)
        self.assertIsInstance(mfcc, np.ndarray)
        self.assertIsInstance(mfcc_preemph, np.ndarray)

    def test_compute_mfcc_different_n_mfcc(self):
        """Test MFCC with different coefficient counts."""
        waveform, sr = torchaudio.load(self.test_audio_path)
        y = waveform.squeeze().numpy()[:16000]  # Use first second

        for n_mfcc in [13, 20, 40]:
            with self.subTest(n_mfcc=n_mfcc):
                mfcc, _ = compute_mfcc(y, sr, n_mfcc=n_mfcc)
                self.assertEqual(mfcc.shape[0], n_mfcc)

    def test_compute_mfcc_save_fig(self):
        """Test MFCC computation with figure saving."""
        waveform, sr = torchaudio.load(self.test_audio_path)
        y = waveform.squeeze().numpy()[:16000]

        warnings.filterwarnings(
            "ignore", message=".*At least one mel filterbank has all zero values.*"
        )
        mfcc, mfcc_preemph = compute_mfcc(
            y, sr, n_mfcc=40, save_fig=True, output_path=self.temp_dir
        )

        fig_dir = Path(self.temp_dir) / "figures"
        self.assertTrue((fig_dir / "mfcc.png").exists())
        self.assertTrue((fig_dir / "mfcc_preemph.png").exists())

    # ===== Mel Spectrogram Tests =====

    def test_compute_melspec_shape(self):
        """Test mel spectrogram computation output shape."""
        waveform, sr = torchaudio.load(self.test_audio_path)
        y = waveform.squeeze().numpy()[:16000]

        warnings.filterwarnings(
            "ignore", message=".*At least one mel filterbank has all zero values.*"
        )
        melspec_db, melspec_preemph_db = compute_melspec(y, sr, n_mels=128)

        self.assertEqual(melspec_db.shape[0], 128)
        self.assertEqual(melspec_preemph_db.shape[0], 128)
        self.assertIsInstance(melspec_db, np.ndarray)

    def test_compute_melspec_different_n_mels(self):
        """Test mel spectrogram with different mel band counts."""
        waveform, sr = torchaudio.load(self.test_audio_path)
        y = waveform.squeeze().numpy()[:16000]

        warnings.filterwarnings(
            "ignore", message=".*At least one mel filterbank has all zero values.*"
        )
        for n_mels in [64, 128, 256]:
            with self.subTest(n_mels=n_mels):
                melspec_db, _ = compute_melspec(y, sr, n_mels=n_mels)
                self.assertEqual(melspec_db.shape[0], n_mels)

    def test_compute_melspec_save_fig(self):
        """Test mel spectrogram computation with figure saving."""
        waveform, sr = torchaudio.load(self.test_audio_path)
        y = waveform.squeeze().numpy()[:16000]

        warnings.filterwarnings(
            "ignore", message=".*At least one mel filterbank has all zero values.*"
        )
        melspec_db, melspec_preemph_db = compute_melspec(
            y, sr, save_fig=True, output_path=self.temp_dir
        )

        fig_dir = Path(self.temp_dir) / "figures"
        self.assertTrue((fig_dir / "melspec_dB.png").exists())
        self.assertTrue((fig_dir / "melspec_dB_preemph.png").exists())

    # ===== Visualization Tests =====

    def test_save_mfcc_specshow(self):
        """Test MFCC visualization saving."""
        data = np.random.rand(40, 100)
        output_path = str(Path(self.temp_dir) / "test_mfcc.png")

        save_mfcc_specshow(data, output_path, "Test MFCC")

        self.assertTrue(os.path.exists(output_path))

    def test_save_melspec_specshow_regular(self):
        """Test mel spectrogram visualization saving (regular)."""
        data = np.random.rand(128, 100)
        output_path = str(Path(self.temp_dir) / "test_melspec.png")

        save_melspec_specshow(data, output_path, "Test Melspec", is_delta=False)

        self.assertTrue(os.path.exists(output_path))

    def test_save_melspec_specshow_delta(self):
        """Test mel spectrogram visualization saving (delta)."""
        data = np.random.rand(128, 100)
        output_path = str(Path(self.temp_dir) / "test_melspec_delta.png")

        save_melspec_specshow(data, output_path, "Test Melspec Delta", is_delta=True)

        self.assertTrue(os.path.exists(output_path))

    # ===== Integration Test =====

    def test_process_audio_file_integration(self):
        """Test complete audio processing pipeline."""
        output = process_audio_file(
            self.test_audio_path,
            self.temp_dir,
            sample_rate=16000,
            save_fig=False,
            save_npy=True,
            n_mfcc=40,
            n_mels=128,
            f_max=8000,
        )

        # Check output dictionary structure
        self.assertIn("mfcc", output)
        self.assertIn("mfcc_preemph", output)
        self.assertIn("melspec_db", output)
        self.assertIn("melspec_preemph_db", output)
        self.assertIn("mfcc_deltas", output)
        self.assertIn("mfcc_preemph_deltas", output)
        self.assertIn("melspec_deltas", output)
        self.assertIn("melspec_preemph_deltas", output)

        # Check shapes
        self.assertEqual(output["mfcc"].shape[0], 40)
        self.assertEqual(output["melspec_db"].shape[0], 128)

        # Check saved numpy files
        output_dir = Path(self.temp_dir)
        self.assertTrue((output_dir / "mfcc.npy").exists())
        self.assertTrue((output_dir / "melspec_db.npy").exists())
        self.assertTrue((output_dir / "mfcc_deltas.npy").exists())

    def test_process_audio_file_with_figures(self):
        """Test audio processing with figure generation."""
        process_audio_file(
            self.test_audio_path,
            self.temp_dir,
            sample_rate=16000,
            save_fig=True,
            save_npy=False,
            n_mfcc=20,
            n_mels=64,
        )

        # Check figures were created
        fig_dir = Path(self.temp_dir) / "figures"
        self.assertTrue(fig_dir.exists())
        self.assertTrue((fig_dir / "mfcc.png").exists())
        self.assertTrue((fig_dir / "mfcc_preemph.png").exists())
        self.assertTrue((fig_dir / "melspec_dB.png").exists())
        self.assertTrue((fig_dir / "melspec_dB_preemph.png").exists())


if __name__ == "__main__":
    unittest.main()
