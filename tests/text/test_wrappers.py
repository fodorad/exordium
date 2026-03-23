"""Tests for text feature extraction and speech-to-text wrappers."""

import unittest

import numpy as np
import torch

from tests.fixtures import AUDIO_MULTISPEAKER


class TestBertWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.text.bert import BertWrapper

        cls.model = BertWrapper(device_id=None)

    def test_call_returns_tensor(self):
        out = self.model("hello world")
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 3)

    def test_predict_returns_numpy(self):
        out = self.model.predict("hello world")
        self.assertIsInstance(out, np.ndarray)

    def test_batch_input(self):
        out = self.model(["hello world", "test sentence"])
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape[0], 2)


class TestRobertaWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.text.roberta import RobertaWrapper

        cls.model = RobertaWrapper(device_id=None)

    def test_call_returns_tensor(self):
        out = self.model("hello world")
        self.assertIsInstance(out, torch.Tensor)

    def test_predict_returns_numpy(self):
        out = self.model.predict("hello world")
        self.assertIsInstance(out, np.ndarray)


class TestXmlRobertaWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.text.xml_roberta import XmlRobertaWrapper

        cls.model = XmlRobertaWrapper(device_id=None)

    def test_call_returns_tensor(self):
        out = self.model("hello world")
        self.assertIsInstance(out, torch.Tensor)


class TestWhisperWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.text.whisper import WhisperWrapper

        cls.model = WhisperWrapper(device_id=None)

    def test_transcribe_from_path(self):
        out = self.model(AUDIO_MULTISPEAKER)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
