"""Tests for audio feature extraction wrappers: CLAP, WavLM, Wav2Vec2."""

import unittest

import numpy as np
import torch

from tests.fixtures import AUDIO_MULTISPEAKER


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


class TestWavlmWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.audio.wavlm import WavlmWrapper

        cls.model = WavlmWrapper(device_id=None)

    def test_audio_to_feature_from_tensor(self):
        audio = torch.zeros(16000)
        result = self.model.audio_to_feature(audio, sample_rate=16000)
        self.assertIsInstance(result, (list, torch.Tensor, np.ndarray))

    def test_audio_to_feature_from_path(self):
        result = self.model.audio_to_feature(AUDIO_MULTISPEAKER, sample_rate=16000)
        self.assertIsNotNone(result)


class TestWav2vec2Wrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.audio.wav2vec2 import Wav2vec2Wrapper

        cls.model = Wav2vec2Wrapper(device_id=None)

    def test_audio_to_feature_from_tensor(self):
        audio = torch.zeros(16000)
        result = self.model.audio_to_feature(audio, sample_rate=16000)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2)

    def test_audio_to_feature_from_path(self):
        result = self.model.audio_to_feature(AUDIO_MULTISPEAKER, sample_rate=16000)
        self.assertIsInstance(result, np.ndarray)


class TestWavlmWrapperBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.audio.wavlm import WavlmWrapper

        cls.model = WavlmWrapper(device_id=None)

    def test_batch_audio_to_features_two_files(self):
        result = self.model.batch_audio_to_features([AUDIO_MULTISPEAKER, AUDIO_MULTISPEAKER])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)

    def test_inference_returns_list_of_tensors(self):
        audio = torch.zeros(16000)
        result = self.model.inference(audio)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIsInstance(result[0], torch.Tensor)


class TestWav2vec2WrapperBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.audio.wav2vec2 import Wav2vec2Wrapper

        cls.model = Wav2vec2Wrapper(device_id=None)

    def test_batch_audio_to_features(self):
        audio1 = torch.zeros(16000)
        audio2 = torch.zeros(8000)
        result = self.model.batch_audio_to_features([audio1, audio2])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], np.ndarray)

    def test_inference_returns_tensor(self):
        audio = torch.zeros(16000)
        result = self.model.inference(audio)
        self.assertIsInstance(result, torch.Tensor)


class TestClapWrapperBatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.audio.clap import ClapWrapper

        cls.model = ClapWrapper(device_id=None)

    def test_batch_audio_to_features(self):
        audio1 = torch.zeros(16000)
        audio2 = torch.zeros(16000)
        result = self.model.batch_audio_to_features([audio1, audio2], sample_rate=16000)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
