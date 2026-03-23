"""Tests for exordium.audio.smile.OpensmileWrapper."""

import unittest

import numpy as np
import torch

from tests.fixtures import AUDIO_MULTISPEAKER


class TestOpensmileWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.audio.smile import OpensmileWrapper

        cls.model = OpensmileWrapper()

    def test_audio_to_feature_from_path(self):
        result = self.model.audio_to_feature(AUDIO_MULTISPEAKER, sample_rate=16000)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2)
        self.assertGreater(result.shape[1], 0)

    def test_audio_to_feature_from_tensor(self):
        audio = torch.zeros(16000)
        result = self.model.audio_to_feature(audio, sample_rate=16000)
        self.assertIsInstance(result, np.ndarray)

    def test_two_audio_to_features(self):
        for audio in [torch.zeros(16000), torch.zeros(8000)]:
            result = self.model.audio_to_feature(audio, sample_rate=16000)
            self.assertIsInstance(result, np.ndarray)


if __name__ == "__main__":
    unittest.main()
