"""Tests for text feature extraction and speech-to-text wrappers."""

import unittest

import numpy as np
import torch

from exordium.text.base import TextModelWrapper
from tests.fixtures import AUDIO_MULTISPEAKER, ModelTestCase, hf_repo_exists


class TestTextMeanPool(unittest.TestCase):
    """TextModelWrapper._mean_pool — static method, no model required."""

    def test_output_shape(self):
        hidden = torch.randn(2, 8, 768)
        mask = torch.ones(2, 8, dtype=torch.long)
        out = TextModelWrapper._mean_pool(hidden, mask)
        self.assertEqual(out.shape, (2, 768))

    def test_all_tokens_attended_equals_mean(self):
        hidden = torch.randn(3, 6, 64)
        mask = torch.ones(3, 6, dtype=torch.long)
        out = TextModelWrapper._mean_pool(hidden, mask)
        self.assertTrue(torch.allclose(out, hidden.mean(dim=1), atol=1e-5))

    def test_padding_excluded_from_mean(self):
        hidden = torch.zeros(1, 4, 16)
        hidden[0, 0] = 1.0
        hidden[0, 1:] = 999.0
        mask = torch.zeros(1, 4, dtype=torch.long)
        mask[0, 0] = 1
        out = TextModelWrapper._mean_pool(hidden, mask)
        self.assertTrue(torch.allclose(out, torch.ones(1, 16), atol=1e-5))

    def test_batch_independence(self):
        hidden = torch.randn(4, 10, 32)
        mask = torch.ones(4, 10, dtype=torch.long)
        batched = TextModelWrapper._mean_pool(hidden, mask)
        for i in range(4):
            single = TextModelWrapper._mean_pool(hidden[i : i + 1], mask[i : i + 1])
            self.assertTrue(torch.allclose(batched[i], single[0], atol=1e-5))


class TestBertWrapper(ModelTestCase):
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


class TestRobertaWrapper(ModelTestCase):
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


class TestXmlRobertaWrapper(ModelTestCase):
    @classmethod
    def setUpClass(cls):
        from exordium.text.xml_roberta import XmlRobertaWrapper

        cls.model = XmlRobertaWrapper(device_id=None)

    def test_call_returns_tensor(self):
        out = self.model("hello world")
        self.assertIsInstance(out, torch.Tensor)

    def test_output_is_sentence_level(self):
        out = self.model("hello world")
        self.assertEqual(out.ndim, 2)


class TestWhisperWrapper(ModelTestCase):
    """The multispeaker fixture is ~61 s — i.e. longer than Whisper's 30 s window."""

    @classmethod
    def setUpClass(cls):
        from exordium.audio.io import load_audio
        from exordium.text.whisper import WhisperWrapper

        cls.model = WhisperWrapper(device_id=None)
        wave, sr = load_audio(AUDIO_MULTISPEAKER, target_sample_rate=16000)
        cls.waveform = wave
        cls.sample_rate = sr

    def test_transcribe_from_path(self):
        out = self.model(AUDIO_MULTISPEAKER)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_preprocess_short_audio_uses_padded_30s_window(self):
        short = self.waveform[..., : 10 * self.sample_rate]
        inputs = self.model.preprocess(short)
        self.assertEqual(inputs["input_features"].shape[-1], 3000)
        self.assertNotIn("attention_mask", inputs)

    def test_preprocess_long_audio_is_not_truncated(self):
        # Regression: the feature extractor's defaults cropped audio to 30 s.
        inputs = self.model.preprocess(AUDIO_MULTISPEAKER)
        self.assertGreater(inputs["input_features"].shape[-1], 3000)
        self.assertIn("attention_mask", inputs)
        self.assertFalse(inputs["attention_mask"].is_floating_point())

    def test_transcribe_long_audio_decodes_past_30s(self):
        # A 30 s-truncated decode yields ~138 words; the full 61 s yields ~250+.
        text = self.model(AUDIO_MULTISPEAKER, beam_size=1)
        self.assertGreater(len(text.split()), 180)

    def test_transcribe_segments_cover_the_full_audio(self):
        segments = self.model.transcribe_segments(AUDIO_MULTISPEAKER, beam_size=1)
        self.assertGreater(len(segments), 1)
        self.assertGreater(segments[-1].end, 45.0)
        starts = [s.start for s in segments]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(all(s.text.strip() for s in segments))

    def test_transcribe_segments_short_audio(self):
        short = self.waveform[..., : 10 * self.sample_rate]
        segments = self.model.transcribe_segments(short, beam_size=1)
        self.assertGreater(len(segments), 0)

    def test_stream_short_audio(self):
        short = self.waveform[..., : 10 * self.sample_rate]
        text = "".join(self.model.stream(short, beam_size=1))
        self.assertGreater(len(text.split()), 5)
        self.assertNotIn("<|", text)  # no timestamp tokens leak through

    def test_stream_long_audio_covers_the_whole_file(self):
        # Regression: streaming a >30 s file first hung forever (the worker
        # thread raised and the streamer never ended), then silently stopped
        # after 30 s because the streamer only sees long-form's first chunk.
        text = "".join(self.model.stream(AUDIO_MULTISPEAKER, beam_size=1))
        self.assertGreater(len(text.split()), 180)
        self.assertNotIn("<|", text)


class TestTextWeightAvailability(unittest.TestCase):
    def test_distil_whisper_repo(self):
        self.assertTrue(hf_repo_exists("distil-whisper/distil-large-v3"))

    def test_bert_base_repo(self):
        self.assertTrue(hf_repo_exists("google-bert/bert-base-uncased"))

    def test_roberta_base_repo(self):
        self.assertTrue(hf_repo_exists("FacebookAI/roberta-base"))

    def test_xlm_roberta_base_repo(self):
        self.assertTrue(hf_repo_exists("FacebookAI/xlm-roberta-base"))


if __name__ == "__main__":
    unittest.main()
