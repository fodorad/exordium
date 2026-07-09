"""Tests for torchaudio forced alignment and Whisper word timestamps."""

import unittest

import numpy as np

from exordium.text.alignment import (
    TorchaudioForcedAligner,
    WhisperWordTimestamper,
    normalize_words,
)
from exordium.text.base import Word
from exordium.text.transcript_align import find_segment
from tests.fixtures import AUDIO_MULTISPEAKER, ModelTestCase


class TestNormalizeWords(unittest.TestCase):
    def test_lowercases_and_splits(self):
        self.assertEqual(normalize_words("Hey Guys What"), ["hey", "guys", "what"])

    def test_strips_punctuation_and_diacritics(self):
        self.assertEqual(normalize_words("Café, résumé!"), ["cafe", "resume"])

    def test_empty_transcript_is_empty_list(self):
        self.assertEqual(normalize_words("  --- ... "), [])


def _assert_valid_word_stream(test: unittest.TestCase, words: list[Word], max_time: float):
    test.assertGreater(len(words), 0)
    for w in words:
        test.assertIsInstance(w, Word)
        test.assertLessEqual(w.start, w.end)
        test.assertGreaterEqual(w.start, 0.0)
        test.assertLessEqual(w.end, max_time + 1.0)
        test.assertGreaterEqual(w.score, 0.0)
    # Non-decreasing start times.
    starts = [w.start for w in words]
    test.assertEqual(starts, sorted(starts))


class TestTorchaudioForcedAligner(ModelTestCase):
    @classmethod
    def setUpClass(cls):
        cls.aligner = TorchaudioForcedAligner(device_id=None)
        # Duration of the fixture (~61s) for the time-bound assertions.
        from exordium.audio.io import load_audio

        wave, sr = load_audio(AUDIO_MULTISPEAKER, target_sample_rate=16000)
        cls.duration = wave.shape[-1] / sr

    def test_align_returns_valid_word_stream(self):
        text = "Hey guys, what do you guys want to eat for lunch?"
        words = self.aligner.align(AUDIO_MULTISPEAKER, text)
        _assert_valid_word_stream(self, words, self.duration)
        self.assertEqual(len(words), len(normalize_words(text)))

    def test_empty_transcript_returns_empty(self):
        self.assertEqual(self.aligner.align(AUDIO_MULTISPEAKER, "..."), [])

    def test_accepts_numpy_input(self):
        from exordium.audio.io import load_audio

        wave, _ = load_audio(AUDIO_MULTISPEAKER, target_sample_rate=16000)
        words = self.aligner.align(wave.numpy().astype(np.float32), "hey guys")
        self.assertEqual([w.text for w in words], ["hey", "guys"])


class TestWhisperWordTimestamper(ModelTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = WhisperWordTimestamper(device_id=None)
        from exordium.audio.io import load_audio

        wave, sr = load_audio(AUDIO_MULTISPEAKER, target_sample_rate=16000)
        cls.duration = wave.shape[-1] / sr

    def test_transcribe_words_from_path(self):
        words = self.model.transcribe_words(AUDIO_MULTISPEAKER)
        _assert_valid_word_stream(self, words, self.duration)

    def test_stream_is_searchable_by_fuzzy_matcher(self):
        words = self.model.transcribe_words(AUDIO_MULTISPEAKER)
        match = find_segment("what do you guys want to eat for lunch", words)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertGreaterEqual(match.score, 80.0)
        self.assertLess(match.start, match.end)


if __name__ == "__main__":
    unittest.main()
