import unittest
import warnings

import numpy as np
import torch

from exordium.audio.base import AudioModelWrapper
from exordium.audio.io import AudioLoader, load_audio
from exordium.audio.wav2vec2 import Wav2vec2Wrapper
from tests.fixtures import AUDIO_MULTISPEAKER


class TestWav2vec2Wrapper(unittest.TestCase):
    """Tests for Wav2vec2Wrapper class."""

    @classmethod
    def setUpClass(cls):
        cls.test_audio_path = AUDIO_MULTISPEAKER
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                cls.wav2vec = Wav2vec2Wrapper()
            except Exception as e:
                cls.wav2vec = None
                print(f"Warning: Could not initialize Wav2Vec: {e}")
        cls.audio_loader = AudioLoader()

    # --- Initialization ---

    def test_inherits_audio_model_wrapper(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        self.assertIsInstance(self.wav2vec, AudioModelWrapper)

    def test_initialization(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        wrapper = Wav2vec2Wrapper()
        self.assertIsInstance(wrapper, Wav2vec2Wrapper)
        self.assertIsNotNone(wrapper.preprocessor)
        self.assertIsNotNone(wrapper.model)

    def test_device_is_cpu(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        self.assertEqual(str(self.wav2vec.device), "cpu")

    def test_sample_rate_constant(self):
        self.assertEqual(Wav2vec2Wrapper.SAMPLE_RATE, 16000)

    # --- __call__ ---

    def test_call_with_1d_numpy(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        waveform = np.random.rand(16000).astype(np.float32)
        features = self.wav2vec(waveform)
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[1], 768)

    def test_call_with_torch_tensor(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        waveform = torch.rand(16000)
        features = self.wav2vec(waveform)
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape[1], 768)

    def test_call_with_2d_batch(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        waveform = np.random.rand(2, 16000).astype(np.float32)
        features = self.wav2vec(waveform)
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.ndim, 3)
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[2], 768)

    def test_call_3d_raises(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        with self.assertRaises(ValueError):
            self.wav2vec(np.random.rand(1, 2, 16000))

    # --- inference() ---

    def test_inference_shape(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        audio, _ = load_audio(
            self.test_audio_path, target_sample_rate=16000, mono=True, squeeze=True
        )
        features = self.wav2vec.inference(audio)
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[1], 768)

    def test_inference_returns_tensor(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        features = self.wav2vec.inference(torch.rand(16000))
        self.assertIsInstance(features, torch.Tensor)

    def test_inference_consistent_hidden_dim(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        waveform = np.random.rand(16000).astype(np.float32)
        self.assertEqual(
            self.wav2vec(waveform).shape[1],
            self.wav2vec.inference(torch.tensor(waveform)).shape[1],
        )

    # --- audio_to_feature: path ---

    def test_audio_to_feature_from_path(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        features = self.wav2vec.audio_to_feature(self.test_audio_path)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[1], 768)

    # --- audio_to_feature: waveform ---

    def test_audio_to_feature_from_numpy(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        waveform = np.random.rand(16000).astype(np.float32)
        features = self.wav2vec.audio_to_feature(waveform)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[1], 768)

    def test_audio_to_feature_from_tensor(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        waveform = torch.rand(16000)
        features = self.wav2vec.audio_to_feature(waveform)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[1], 768)

    def test_audio_to_feature_from_2d_numpy(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        # 2D (C, T) — base class selects first channel
        waveform = np.random.rand(2, 16000).astype(np.float32)
        features = self.wav2vec.audio_to_feature(waveform)
        self.assertEqual(features.shape[1], 768)

    # --- batch_audio_to_features ---

    def test_batch_audio_to_features_from_paths(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        results = self.wav2vec.batch_audio_to_features([self.test_audio_path, self.test_audio_path])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, np.ndarray)
            self.assertEqual(r.ndim, 2)
            self.assertEqual(r.shape[1], 768)

    def test_batch_audio_to_features_from_waveforms(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        waveforms = [np.random.rand(16000).astype(np.float32) for _ in range(3)]
        results = self.wav2vec.batch_audio_to_features(waveforms)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, np.ndarray)
            self.assertEqual(r.shape[1], 768)

    def test_batch_audio_to_features_variable_length(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec2 model not available")
        waveforms = [
            np.random.rand(8000).astype(np.float32),
            np.random.rand(16000).astype(np.float32),
        ]
        results = self.wav2vec.batch_audio_to_features(waveforms)
        self.assertEqual(len(results), 2)
        # shorter input should have fewer output frames
        self.assertLess(results[0].shape[0], results[1].shape[0])

    # --- Integration ---

    def test_integration_load_and_wav2vec(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec model not available")
        waveform, _ = load_audio(
            self.test_audio_path, target_sample_rate=16000, mono=True, squeeze=True
        )
        features = self.wav2vec(waveform)
        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[1], 768)

    def test_integration_audioloader_pipeline(self):
        if self.wav2vec is None:
            self.skipTest("Wav2Vec model not available")
        waveform, _ = self.audio_loader.load_audio(
            self.test_audio_path, target_sample_rate=16000, mono=True, squeeze=True
        )
        features = self.wav2vec(waveform)
        self.assertIsInstance(features, torch.Tensor)
        self.assertEqual(features.shape[1], 768)


if __name__ == "__main__":
    unittest.main()
