"""Tests for Emotion2vecWrapper speech emotion feature extractor."""

import pathlib
import tempfile
import unittest

import numpy as np
import torch

from exordium.audio.emotion2vec import (
    EMOTION2VEC_FEATURE_DIM,
    EMOTION2VEC_SAMPLE_RATE,
    Emotion2vecWrapper,
)
from tests.fixtures import AUDIO_MULTISPEAKER


class TestEmotion2vecWrapperInit(unittest.TestCase):
    """Quick sanity checks on module constants (no model download)."""

    def test_sample_rate_is_16k(self):
        self.assertEqual(EMOTION2VEC_SAMPLE_RATE, 16000)

    def test_feature_dim_is_768(self):
        self.assertEqual(EMOTION2VEC_FEATURE_DIM, 768)


class TestEmotion2vecWrapper(unittest.TestCase):
    """Inference tests — loads the model once for the entire suite."""

    @classmethod
    def setUpClass(cls):
        cls.model = Emotion2vecWrapper(device_id=None)

    # ------------------------------------------------------------------
    # __call__ — different input shapes
    # ------------------------------------------------------------------

    def test_call_1d_tensor(self):
        waveform = torch.randn(16000)  # 1 second
        out = self.model(waveform)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[1], EMOTION2VEC_FEATURE_DIM)
        self.assertGreater(out.shape[0], 0)

    def test_call_2d_tensor(self):
        waveform = torch.randn(2, 16000)
        out = self.model(waveform)
        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[2], EMOTION2VEC_FEATURE_DIM)

    def test_call_numpy(self):
        waveform = np.random.randn(16000).astype(np.float32)
        out = self.model(waveform)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[1], EMOTION2VEC_FEATURE_DIM)

    def test_call_invalid_ndim_raises(self):
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 1, 16000))

    # ------------------------------------------------------------------
    # Frame rate sanity — ~50 Hz
    # ------------------------------------------------------------------

    def test_frame_rate_approximately_50hz(self):
        # 1 second of audio should yield approximately 49-50 frames
        waveform = torch.randn(16000)
        out = self.model(waveform)
        self.assertGreaterEqual(out.shape[0], 40)
        self.assertLessEqual(out.shape[0], 60)

    def test_longer_audio_more_frames(self):
        out_1s = self.model(torch.randn(16000))
        out_2s = self.model(torch.randn(32000))
        self.assertGreater(out_2s.shape[0], out_1s.shape[0])

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_deterministic_inference(self):
        waveform = torch.randn(16000)
        out1 = self.model(waveform)
        out2 = self.model(waveform)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    # ------------------------------------------------------------------
    # Output is non-trivial
    # ------------------------------------------------------------------

    def test_output_is_not_all_zeros(self):
        waveform = torch.randn(16000)
        out = self.model(waveform)
        self.assertTrue((out.abs().sum(dim=-1) > 0).all())

    # ------------------------------------------------------------------
    # audio_to_feature
    # ------------------------------------------------------------------

    def test_audio_to_feature_from_path(self):
        result = self.model.audio_to_feature(AUDIO_MULTISPEAKER)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[1], EMOTION2VEC_FEATURE_DIM)
        self.assertGreater(result.shape[0], 0)

    def test_audio_to_feature_from_tensor(self):
        waveform = torch.randn(16000)
        result = self.model.audio_to_feature(waveform)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[1], EMOTION2VEC_FEATURE_DIM)

    def test_audio_to_feature_from_numpy(self):
        waveform = np.random.randn(16000).astype(np.float32)
        result = self.model.audio_to_feature(waveform)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], EMOTION2VEC_FEATURE_DIM)

    def test_audio_to_feature_caching(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = pathlib.Path(tmp_dir) / "emotion2vec_features.npy"
            result1 = self.model.audio_to_feature(
                AUDIO_MULTISPEAKER,
                output_path=out_path,
            )
            result2 = self.model.audio_to_feature(
                AUDIO_MULTISPEAKER,
                output_path=out_path,
            )
            self.assertTrue(np.allclose(result1, result2))

    # ------------------------------------------------------------------
    # batch_audio_to_features
    # ------------------------------------------------------------------

    def test_batch_audio_to_features(self):
        results = self.model.batch_audio_to_features(
            [AUDIO_MULTISPEAKER, AUDIO_MULTISPEAKER],
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for arr in results:
            self.assertIsInstance(arr, np.ndarray)
            self.assertEqual(arr.ndim, 2)
            self.assertEqual(arr.shape[1], EMOTION2VEC_FEATURE_DIM)

    def test_batch_variable_length(self):
        w1 = torch.randn(16000)  # 1 second
        w2 = torch.randn(32000)  # 2 seconds
        results = self.model.batch_audio_to_features([w1, w2])
        self.assertEqual(len(results), 2)
        # Shorter audio should produce fewer frames
        self.assertLess(results[0].shape[0], results[1].shape[0])

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def test_inference_returns_tensor(self):
        waveform = torch.randn(16000)
        out = self.model.inference(waveform)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[1], EMOTION2VEC_FEATURE_DIM)


if __name__ == "__main__":
    unittest.main()
