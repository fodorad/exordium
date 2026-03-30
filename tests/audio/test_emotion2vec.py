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
from tests.fixtures import AUDIO_MULTISPEAKER, head_ok


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

    def test_call_1d_tensor(self):
        out = self.model(torch.randn(16000))
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[1], EMOTION2VEC_FEATURE_DIM)

    def test_call_2d_tensor(self):
        out = self.model(torch.randn(2, 16000))
        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[2], EMOTION2VEC_FEATURE_DIM)

    def test_call_numpy(self):
        out = self.model(np.random.randn(16000).astype(np.float32))
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[1], EMOTION2VEC_FEATURE_DIM)

    def test_call_invalid_ndim_raises(self):
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 1, 16000))

    def test_frame_rate_approximately_50hz(self):
        out = self.model(torch.randn(16000))
        self.assertGreaterEqual(out.shape[0], 40)
        self.assertLessEqual(out.shape[0], 60)

    def test_longer_audio_more_frames(self):
        out_1s = self.model(torch.randn(16000))
        out_2s = self.model(torch.randn(32000))
        self.assertGreater(out_2s.shape[0], out_1s.shape[0])

    def test_deterministic_inference(self):
        waveform = torch.randn(16000)
        self.assertTrue(torch.allclose(self.model(waveform), self.model(waveform), atol=1e-6))

    def test_output_is_not_all_zeros(self):
        out = self.model(torch.randn(16000))
        self.assertTrue((out.abs().sum(dim=-1) > 0).all())

    def test_audio_to_feature_from_path(self):
        result = self.model.audio_to_feature(AUDIO_MULTISPEAKER)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[1], EMOTION2VEC_FEATURE_DIM)

    def test_audio_to_feature_caching(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = pathlib.Path(tmp_dir) / "em2v.npy"
            r1 = self.model.audio_to_feature(AUDIO_MULTISPEAKER, output_path=out_path)
            r2 = self.model.audio_to_feature(AUDIO_MULTISPEAKER, output_path=out_path)
            self.assertTrue(np.allclose(r1, r2))

    def test_batch_audio_to_features(self):
        results = self.model.batch_audio_to_features([AUDIO_MULTISPEAKER, AUDIO_MULTISPEAKER])
        self.assertEqual(len(results), 2)
        for arr in results:
            self.assertEqual(arr.shape[1], EMOTION2VEC_FEATURE_DIM)

    def test_batch_variable_length(self):
        results = self.model.batch_audio_to_features([torch.randn(16000), torch.randn(32000)])
        self.assertLess(results[0].shape[0], results[1].shape[0])

    def test_inference_returns_tensor(self):
        out = self.model.inference(torch.randn(16000))
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape[1], EMOTION2VEC_FEATURE_DIM)


class TestEmotion2vecWeightAvailability(unittest.TestCase):
    def test_emotion2vec_plus_seed_url(self):
        url = "https://huggingface.co/emotion2vec/emotion2vec_plus_seed/resolve/main/model.pt"
        self.assertTrue(head_ok(url), f"Not reachable: {url}")


if __name__ == "__main__":
    unittest.main()
