import unittest

import numpy as np
import torch

from exordium.audio.base import AudioModelWrapper
from exordium.audio.wavlm import WavlmWrapper
from tests.fixtures import AUDIO_MULTISPEAKER


class TestWavlmWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wavlm = WavlmWrapper(device_id=-1)
        cls.test_audio_path = AUDIO_MULTISPEAKER

    # --- Initialization ---

    def test_inherits_audio_model_wrapper(self):
        self.assertIsInstance(self.wavlm, AudioModelWrapper)

    def test_initialization_valid_models(self):
        for model_name in ["base", "base+", "large"]:
            wrapper = WavlmWrapper(model_name=model_name)
            self.assertIsInstance(wrapper.model, torch.nn.Module)
            self.assertEqual(wrapper.model.eval(), wrapper.model)

    def test_initialization_invalid_model_raises(self):
        with self.assertRaises(ValueError):
            WavlmWrapper(model_name="invalid")

    def test_device_is_cpu(self):
        self.assertEqual(str(self.wavlm.device), "cpu")

    def test_device_placement_gpu(self):
        if torch.cuda.is_available():
            gpu_wrapper = WavlmWrapper(device_id=0)
            self.assertIn(str(gpu_wrapper.device)[-1], {"cuda:0", "mps:0"})

    # --- __call__ ---

    def test_call_with_numpy_1d(self):
        waveform = np.random.rand(16000).astype(np.float32)
        features = self.wavlm(waveform)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 12)
        for f in features:
            self.assertIsInstance(f, torch.Tensor)

    def test_call_with_torch_tensor_1d(self):
        waveform = torch.rand(16000)
        features = self.wavlm(waveform)
        self.assertEqual(len(features), 12)
        for f in features:
            self.assertIsInstance(f, torch.Tensor)

    def test_call_with_2d_batch(self):
        waveform = np.random.rand(2, 16000).astype(np.float32)
        features = self.wavlm(waveform)
        self.assertEqual(len(features), 12)
        for f in features:
            self.assertIsInstance(f, torch.Tensor)

    def test_call_feature_hidden_dim(self):
        waveform = np.random.rand(16000).astype(np.float32)
        features = self.wavlm(waveform)
        self.assertEqual(features[0].shape[-1], 768)

    def test_call_invalid_shape_raises(self):
        with self.assertRaises(ValueError):
            self.wavlm(torch.rand(1, 2, 16000))

    # --- inference() ---

    def test_inference_returns_list(self):
        features = self.wavlm.inference(np.random.rand(16000).astype(np.float32))
        self.assertIsInstance(features, list)

    def test_inference_layer_count(self):
        features = self.wavlm.inference(np.random.rand(16000).astype(np.float32))
        self.assertEqual(len(features), 12)

    def test_inference_returns_tensors(self):
        for f in self.wavlm.inference(torch.rand(16000)):
            self.assertIsInstance(f, torch.Tensor)

    def test_inference_hidden_dim(self):
        features = self.wavlm.inference(np.random.rand(16000).astype(np.float32))
        self.assertEqual(features[0].shape[-1], 768)

    def test_inference_consistent_with_call(self):
        waveform = np.random.rand(16000).astype(np.float32)
        call_features = self.wavlm(waveform)
        inf_features = self.wavlm.inference(waveform)
        self.assertEqual(len(call_features), len(inf_features))
        self.assertEqual(call_features[0].shape[-1], inf_features[0].shape[-1])

    # --- audio_to_feature: path ---

    def test_audio_to_feature_from_path_returns_list(self):
        features = self.wavlm.audio_to_feature(self.test_audio_path)
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)

    def test_audio_to_feature_from_path_numpy_arrays(self):
        for feature in self.wavlm.audio_to_feature(self.test_audio_path):
            self.assertIsInstance(feature, np.ndarray)

    def test_audio_to_feature_from_path_shape(self):
        for feature in self.wavlm.audio_to_feature(self.test_audio_path):
            self.assertEqual(feature.shape[0], 3060)
            self.assertEqual(feature.shape[1], 768)

    # --- audio_to_feature: waveform ---

    def test_audio_to_feature_from_numpy(self):
        waveform = np.random.rand(16000).astype(np.float32)
        features = self.wavlm.audio_to_feature(waveform)
        self.assertIsInstance(features, list)
        self.assertEqual(features[0].shape[-1], 768)

    def test_audio_to_feature_from_tensor(self):
        waveform = torch.rand(16000)
        features = self.wavlm.audio_to_feature(waveform)
        self.assertIsInstance(features, list)
        self.assertEqual(features[0].shape[-1], 768)

    def test_audio_to_feature_from_2d_numpy(self):
        # 2D (C, T) — base class selects first channel
        waveform = np.random.rand(2, 16000).astype(np.float32)
        features = self.wavlm.audio_to_feature(waveform)
        self.assertEqual(features[0].shape[-1], 768)

    # --- batch_audio_to_features ---

    def test_batch_audio_to_features_from_paths(self):
        results = self.wavlm.batch_audio_to_features([self.test_audio_path, self.test_audio_path])
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)
        for per_file in results:
            self.assertIsInstance(per_file, list)
            self.assertEqual(len(per_file), 12)
            for layer in per_file:
                self.assertIsInstance(layer, np.ndarray)
                self.assertEqual(layer.shape[-1], 768)

    def test_batch_audio_to_features_from_waveforms(self):
        waveforms = [np.random.rand(16000).astype(np.float32) for _ in range(3)]
        results = self.wavlm.batch_audio_to_features(waveforms)
        self.assertEqual(len(results), 3)
        for per_file in results:
            self.assertEqual(len(per_file), 12)
            self.assertEqual(per_file[0].shape[-1], 768)

    def test_batch_audio_to_features_layer_count(self):
        waveforms = [
            np.random.rand(16000).astype(np.float32),
            np.random.rand(8000).astype(np.float32),
        ]
        results = self.wavlm.batch_audio_to_features(waveforms)
        for per_file in results:
            self.assertEqual(len(per_file), 12)

    def test_batch_audio_to_features_variable_length_trimmed(self):
        # Shorter input should have fewer output frames than longer input
        waveforms = [
            np.random.rand(8000).astype(np.float32),
            np.random.rand(16000).astype(np.float32),
        ]
        results = self.wavlm.batch_audio_to_features(waveforms)
        self.assertLess(results[0][0].shape[0], results[1][0].shape[0])

    def test_batch_audio_to_features_matches_single(self):
        # Single-file batch should match audio_to_feature frame count
        waveform = np.random.rand(16000).astype(np.float32)
        single = self.wavlm.audio_to_feature(waveform)
        batch = self.wavlm.batch_audio_to_features([waveform])
        self.assertEqual(single[0].shape[0], batch[0][0].shape[0])


if __name__ == "__main__":
    unittest.main()
