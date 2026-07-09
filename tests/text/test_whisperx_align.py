"""Tests for the whisperX forced aligner (ships with the ``text`` extra)."""

import unittest

from exordium.text import whisperx_align
from exordium.text.base import Word
from exordium.text.transcript_align import find_segment
from tests.fixtures import AUDIO_MULTISPEAKER, ModelTestCase

WHISPERX_AVAILABLE = whisperx_align._WHISPERX_AVAILABLE


class TestWhisperxImportGuard(unittest.TestCase):
    @unittest.skipIf(WHISPERX_AVAILABLE, "whisperX is installed; guard not exercised.")
    def test_raises_without_extra(self):
        with self.assertRaises(ImportError):
            whisperx_align.WhisperxForcedAligner(device_id=None)


@unittest.skipUnless(WHISPERX_AVAILABLE, "Requires whisperX (the text extra).")
class TestWhisperxForcedAligner(ModelTestCase):
    @classmethod
    def setUpClass(cls):
        cls.aligner = whisperx_align.WhisperxForcedAligner(language="en", device_id=None)

    def test_align_returns_valid_word_stream(self):
        text = "Hey guys, what do you guys want to eat for lunch?"
        words = self.aligner.align(AUDIO_MULTISPEAKER, text)
        self.assertGreater(len(words), 0)
        for w in words:
            self.assertIsInstance(w, Word)
            self.assertLessEqual(w.start, w.end)
            self.assertGreaterEqual(w.start, 0.0)
        starts = [w.start for w in words]
        self.assertEqual(starts, sorted(starts))

    def test_empty_transcript_returns_empty(self):
        self.assertEqual(self.aligner.align(AUDIO_MULTISPEAKER, "   "), [])

    def test_stream_is_searchable_by_fuzzy_matcher(self):
        words = self.aligner.align(
            AUDIO_MULTISPEAKER, "Hey guys, what do you guys want to eat for lunch?"
        )
        match = find_segment("what do you guys want to eat", words)
        self.assertIsNotNone(match)


if __name__ == "__main__":
    unittest.main()
