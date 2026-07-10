"""Tests for torchaudio forced alignment and Whisper word timestamps."""

import unittest

import numpy as np

from exordium.text.alignment import (
    TorchaudioForcedAligner,
    WhisperWordTimestamper,
    normalize_words,
)
from exordium.text.base import Segment, Word
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

    def test_align_segments_offsets_words_onto_full_timeline(self):
        segments = [
            Segment(text="hey guys", start=0.0, end=2.0),
            Segment(text="what do you guys want to eat", start=2.0, end=6.0),
        ]
        words = self.aligner.align_segments(AUDIO_MULTISPEAKER, segments)
        self.assertGreater(len(words), 2)
        # Words from the second segment must sit after its start, not at 0.
        self.assertGreaterEqual(words[-1].start, 2.0)
        starts = [w.start for w in words]
        self.assertEqual(starts, sorted(starts))

    def test_align_segments_skips_blank_segments(self):
        segments = [
            Segment(text="   ", start=0.0, end=2.0),
            Segment(text="hey", start=0.0, end=2.0),
        ]
        words = self.aligner.align_segments(AUDIO_MULTISPEAKER, segments)
        self.assertEqual([w.text for w in words], ["hey"])


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

    def test_long_audio_words_reach_beyond_whispers_30s_window(self):
        # Regression: Whisper cropped to 30 s, so the aligner smeared a
        # half-transcript across the full 61 s audio and everything past the
        # window was unrecoverable.
        words = self.model.transcribe_words(AUDIO_MULTISPEAKER)
        self.assertGreater(len(words), 180)
        self.assertGreater(words[-1].end, 45.0)
        self.assertLessEqual(words[-1].end, self.duration + 1.0)

    def test_text_near_end_of_long_audio_is_findable(self):
        words = self.model.transcribe_words(AUDIO_MULTISPEAKER)
        match = find_segment("check us out at my taste base", words)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertGreater(match.start, 45.0)


if __name__ == "__main__":
    unittest.main()
