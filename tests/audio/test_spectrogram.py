"""Tests for exordium.audio.spectrogram feature extraction functions."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from exordium.audio.spectrogram import (
    apply_preemphasis,
    compute_deltas,
    compute_melspec,
    compute_mfcc,
    preprocess_audio,
)
from tests.fixtures import AUDIO_MULTISPEAKER


class TestPreprocessAudio(unittest.TestCase):
    def test_stereo_to_mono(self):
        waveform = torch.randn(2, 16000)
        out, sr = preprocess_audio(waveform, 16000, 16000)
        self.assertEqual(out.shape[0], 1)

    def test_resampling(self):
        waveform = torch.randn(1, 16000)
        out, sr = preprocess_audio(waveform, 16000, 8000)
        self.assertEqual(sr, 8000)
        self.assertEqual(out.shape[0], 1)

    def test_mono_passthrough(self):
        waveform = torch.randn(1, 16000)
        out, sr = preprocess_audio(waveform, 16000, 16000)
        self.assertEqual(out.shape, waveform.shape)


class TestApplyPreemphasis(unittest.TestCase):
    def test_output_shape(self):
        y = torch.randn(16000)
        out = apply_preemphasis(y)
        self.assertEqual(out.shape, y.shape)

    def test_empty_tensor(self):
        y = torch.tensor([])
        out = apply_preemphasis(y)
        self.assertEqual(out.numel(), 0)

    def test_dtype_preserved(self):
        y = torch.randn(1000).float()
        out = apply_preemphasis(y)
        self.assertEqual(out.dtype, torch.float32)


class TestComputeMfcc(unittest.TestCase):
    def setUp(self):
        self.y = torch.randn(16000)
        self.sr = 16000

    def test_output_shape(self):
        mfcc, mfcc_pre = compute_mfcc(self.y, self.sr, n_mfcc=40)
        self.assertEqual(mfcc.shape[0], 40)
        self.assertEqual(mfcc_pre.shape[0], 40)

    def test_output_dtype(self):
        mfcc, _ = compute_mfcc(self.y, self.sr)
        self.assertEqual(mfcc.dtype, np.float32)

    def test_stereo_input_handled(self):
        y_stereo = torch.randn(2, 16000)
        mfcc, _ = compute_mfcc(y_stereo, self.sr)
        self.assertEqual(mfcc.shape[0], 40)

    def test_time_dim_positive(self):
        mfcc, _ = compute_mfcc(self.y, self.sr)
        self.assertGreater(mfcc.shape[1], 0)


class TestComputeMelspec(unittest.TestCase):
    def setUp(self):
        self.y = torch.randn(16000)
        self.sr = 16000

    def test_output_shape(self):
        mel, mel_pre = compute_melspec(self.y, self.sr, n_mels=128, n_fft=1024)
        self.assertEqual(mel.shape[0], 128)
        self.assertEqual(mel_pre.shape[0], 128)

    def test_output_dtype(self):
        mel, _ = compute_melspec(self.y, self.sr)
        self.assertEqual(mel.dtype, np.float32)

    def test_time_dim_positive(self):
        mel, _ = compute_melspec(self.y, self.sr)
        self.assertGreater(mel.shape[1], 0)


class TestComputeDeltas(unittest.TestCase):
    def test_delta_shape(self):
        features = np.random.randn(40, 100).astype(np.float32)
        delta = compute_deltas(features)
        self.assertEqual(delta.shape, features.shape)

    def test_return_all(self):
        features = np.random.randn(40, 100).astype(np.float32)
        feats, delta, delta2 = compute_deltas(features, return_all=True)
        self.assertEqual(feats.shape, features.shape)
        self.assertEqual(delta.shape, features.shape)
        self.assertEqual(delta2.shape, features.shape)


class TestComputeMfccWithSave(unittest.TestCase):
    def test_compute_mfcc_with_save_fig(self):
        y = torch.zeros(16000)
        with tempfile.TemporaryDirectory() as tmp:
            mfcc, mfcc_pre = compute_mfcc(y, 16000, save_fig=True, output_path=tmp)
            self.assertIsInstance(mfcc, np.ndarray)
            self.assertTrue(Path(tmp, "figures", "mfcc.png").exists())


class TestComputeMelspecWithSave(unittest.TestCase):
    def test_compute_melspec_with_save_fig(self):
        y = torch.zeros(22050)
        with tempfile.TemporaryDirectory() as tmp:
            mel, mel_pre = compute_melspec(y, 22050, save_fig=True, output_path=tmp)
            self.assertIsInstance(mel, np.ndarray)


class TestProcessAudioFile(unittest.TestCase):
    def test_process_audio_file_returns_dict(self):
        from exordium.audio.spectrogram import process_audio_file

        with tempfile.TemporaryDirectory() as tmp:
            result = process_audio_file(
                AUDIO_MULTISPEAKER, tmp, sample_rate=16000, save_fig=False, save_npy=True
            )
            self.assertIsInstance(result, dict)
            self.assertIn("input_path", result)


class TestSaveMfccMelspecShow(unittest.TestCase):
    def test_save_mfcc_specshow(self):
        from exordium.audio.spectrogram import save_mfcc_specshow

        y = torch.zeros(8000)
        mfcc, _ = compute_mfcc(y, 16000, n_mfcc=20)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            p = Path(f.name)
        try:
            save_mfcc_specshow(mfcc, str(p), "Test MFCC")
            self.assertTrue(p.exists())
        finally:
            p.unlink(missing_ok=True)

    def test_save_melspec_specshow(self):
        from exordium.audio.spectrogram import save_melspec_specshow

        y = torch.zeros(22050)
        mel, _ = compute_melspec(y, 22050, n_mels=64)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            p = Path(f.name)
        try:
            save_melspec_specshow(mel, str(p), "Test Melspec")
            self.assertTrue(p.exists())
        finally:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
