"""Tests for audio feature extraction wrappers: CLAP, WavLM, Wav2Vec2."""

import unittest

import numpy as np
import torch

from exordium.audio.clap import CLAP_MODEL_ID, CLAP_SAMPLE_RATE
from exordium.audio.wav2vec2 import SUPPORTED_MODELS, Wav2vec2Wrapper
from exordium.audio.wavlm import WAVLM_SAMPLE_RATE, WavlmWrapper, _MODEL_IDS
from tests.fixtures import AUDIO_MULTISPEAKER, head_ok, hf_repo_exists


class TestWavlmWrapperInit(unittest.TestCase):
    """Validation that runs before model loading — no download required."""

    def test_invalid_model_name_raises(self):
        with self.assertRaises(ValueError):
            WavlmWrapper(model_name="invalid")

    def test_sample_rate_constant(self):
        self.assertEqual(WAVLM_SAMPLE_RATE, 16000)

    def test_model_ids_keys(self):
        self.assertIn("base", _MODEL_IDS)
        self.assertIn("base+", _MODEL_IDS)
        self.assertIn("large", _MODEL_IDS)

    def test_model_ids_values_are_hf_strings(self):
        for key, hf_id in _MODEL_IDS.items():
            self.assertIsInstance(hf_id, str, msg=f"_MODEL_IDS[{key!r}] is not a string")
            self.assertIn("microsoft/wavlm", hf_id)


class TestClapConstants(unittest.TestCase):
    def test_sample_rate_is_48k(self):
        self.assertEqual(CLAP_SAMPLE_RATE, 48000)

    def test_model_id_is_string(self):
        self.assertIsInstance(CLAP_MODEL_ID, str)
        self.assertGreater(len(CLAP_MODEL_ID), 0)


class TestWav2vec2WrapperInvalidModel(unittest.TestCase):
    def test_invalid_model_name_raises(self):
        with self.assertRaises(ValueError):
            Wav2vec2Wrapper(device_id=None, model_name="nonexistent")

    def test_supported_models_contains_expected(self):
        self.assertIn("base-960h", SUPPORTED_MODELS)
        self.assertIn("emotion-iemocap", SUPPORTED_MODELS)


class TestClapWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.audio.clap import ClapWrapper

        cls.model = ClapWrapper(device_id=None)

    def test_audio_to_feature_from_path(self):
        result = self.model.audio_to_feature(AUDIO_MULTISPEAKER, sample_rate=16000)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.ndim, 2)

    def test_audio_to_feature_from_tensor(self):
        audio = torch.zeros(16000)
        result = self.model.audio_to_feature(audio, sample_rate=16000)
        self.assertIsInstance(result, torch.Tensor)

    def test_batch_audio_to_features(self):
        result = self.model.batch_audio_to_features(
            [torch.zeros(16000), torch.zeros(16000)], sample_rate=16000
        )
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 2)


class TestWavlmWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = WavlmWrapper(device_id=None)

    def test_audio_to_feature_from_tensor(self):
        result = self.model.audio_to_feature(torch.zeros(16000), sample_rate=16000)
        self.assertIsInstance(result, (list, torch.Tensor, np.ndarray))

    def test_audio_to_feature_from_path(self):
        result = self.model.audio_to_feature(AUDIO_MULTISPEAKER, sample_rate=16000)
        self.assertIsNotNone(result)

    def test_batch_audio_to_features(self):
        result = self.model.batch_audio_to_features([AUDIO_MULTISPEAKER, AUDIO_MULTISPEAKER])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_inference_returns_list_of_tensors(self):
        result = self.model.inference(torch.zeros(16000))
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], torch.Tensor)


class TestWav2vec2Wrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Wav2vec2Wrapper(device_id=None, model_name="base-960h")

    def test_audio_to_feature_from_tensor(self):
        result = self.model.audio_to_feature(torch.zeros(16000), sample_rate=16000)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2)

    def test_audio_to_feature_from_path(self):
        result = self.model.audio_to_feature(AUDIO_MULTISPEAKER, sample_rate=16000)
        self.assertIsInstance(result, np.ndarray)

    def test_batch_audio_to_features(self):
        result = self.model.batch_audio_to_features([torch.zeros(16000), torch.zeros(8000)])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_inference_returns_tensor(self):
        result = self.model.inference(torch.zeros(16000))
        self.assertIsInstance(result, torch.Tensor)


class TestWav2vec2WrapperEmotion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Wav2vec2Wrapper(device_id=None, model_name="emotion-iemocap")
        cls.base_model = Wav2vec2Wrapper(device_id=None, model_name="base-960h")

    def test_output_shape(self):
        result = self.model(torch.zeros(16000))
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape[1], 768)

    def test_audio_to_feature_shape(self):
        result = self.model.audio_to_feature(torch.zeros(16000), sample_rate=16000)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], 768)

    def test_invalid_ndim_raises(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 1, 16000))

    def test_different_weights_from_base(self):
        audio = torch.randn(16000)
        self.assertFalse(torch.allclose(self.base_model(audio), self.model(audio)))


class TestAudioWeightAvailability(unittest.TestCase):
    def test_clap_repo(self):
        self.assertTrue(hf_repo_exists("laion/larger_clap_music_and_speech"))

    def test_wavlm_base_repo(self):
        self.assertTrue(hf_repo_exists("microsoft/wavlm-base"))

    def test_wavlm_base_plus_repo(self):
        self.assertTrue(hf_repo_exists("microsoft/wavlm-base-plus"))

    def test_wavlm_large_repo(self):
        self.assertTrue(hf_repo_exists("microsoft/wavlm-large"))

    def test_wav2vec2_base_960h_repo(self):
        self.assertTrue(hf_repo_exists("facebook/wav2vec2-base-960h"))

    def test_wav2vec2_base_repo(self):
        self.assertTrue(hf_repo_exists("facebook/wav2vec2-base"))

    def test_wav2vec2_iemocap_ckpt(self):
        url = "https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP/resolve/main/wav2vec2.ckpt"
        self.assertTrue(head_ok(url), f"Not reachable: {url}")


if __name__ == "__main__":
    unittest.main()
