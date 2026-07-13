"""Tests for the whisperX forced aligner (ships with the ``text`` extra)."""

import unittest

import numpy as np

from exordium.text import whisperx_align
from exordium.text.base import MIN_ALIGN_SAMPLES, Segment, Word
from exordium.text.transcript_align import find_segment
from tests.fixtures import (
    AUDIO_MULTISPEAKER,
    ModelTestCase,
    best_anchored_word,
    logging_enabled,
)

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

    def test_align_segments_returns_full_timeline_words(self):
        # whisperX's native batched segment path (used for long recordings).
        segments = [
            Segment(text="hey guys", start=0.0, end=2.0),
            Segment(text="what do you guys want to eat", start=2.0, end=6.0),
        ]
        words = self.aligner.align_segments(AUDIO_MULTISPEAKER, segments)
        self.assertGreater(len(words), 2)
        self.assertGreaterEqual(words[-1].start, 2.0)
        starts = [w.start for w in words]
        self.assertEqual(starts, sorted(starts))

    def test_align_segments_empty_returns_empty(self):
        self.assertEqual(self.aligner.align_segments(AUDIO_MULTISPEAKER, []), [])

    def test_align_segments_recovers_degenerate_micro_segments(self):
        # Regression: Whisper long-form emits real text with a near-zero duration.
        # Any slice under MIN_ALIGN_SAMPLES collapses to a single emission frame, and
        # whisperX rescales timestamps by `duration * ... / (trellis.size(0) - 1)` —
        # a one-row trellis divides by zero and aborts the entire recording.
        #
        # The "I" segments are the teeth of this test: a 400-sample floor (wav2vec2's
        # own minimum, and the obvious guess for the guard) still lets them through,
        # because a single-character text is short enough for whisperX's backtrack to
        # *succeed* on one frame and reach the division.
        segments = [
            Segment(text="hey guys", start=0.0, end=2.0),
            Segment(text="that", start=2.100, end=2.120),  # 320 samples
            Segment(text="I", start=2.200, end=2.230),  # 480 samples: past a 400 floor
            Segment(text="I", start=2.300, end=2.340),  # 640 samples: past a 400 floor
            Segment(text="what do you guys want to eat", start=2.5, end=6.0),
        ]
        words = self.aligner.align_segments(AUDIO_MULTISPEAKER, segments)
        # Every degenerate word comes back timed, not dropped.
        text = [w.text.strip().lower().strip(".,!?") for w in words]
        self.assertIn("that", text)
        self.assertEqual(text.count("i"), 2)
        starts = [w.start for w in words]
        self.assertEqual(starts, sorted(starts))

    def test_recovered_degenerate_word_matches_ground_truth_timing(self):
        # Widening must return the word *correctly* timed, not merely return it. Align
        # the full sentence for ground truth, then re-align one word as a degenerate
        # 0.02s micro-segment and check we land back on it.
        target = best_anchored_word(
            self.aligner.align(
                AUDIO_MULTISPEAKER, "Hey guys, what do you guys want to eat for lunch?"
            )
        )
        midpoint = (target.start + target.end) / 2
        degenerate = [Segment(text=target.text, start=midpoint, end=midpoint + 0.02)]

        words = self.aligner.align_segments(AUDIO_MULTISPEAKER, degenerate)
        self.assertEqual(len(words), 1)
        # Tolerance is looser than the torchaudio equivalent because whisperX's
        # character-level timings are coarser, but still far tighter than the widened
        # window: this asserts the word was *located*, not merely returned.
        self.assertAlmostEqual(words[0].start, target.start, delta=0.25)

    def test_align_segments_all_unalignable_returns_empty(self):
        # Audio too short to carry any of the text: every segment is dropped, so the
        # payload is empty and whisperX is never called.
        waveform = np.zeros(MIN_ALIGN_SAMPLES, dtype=np.float32)
        segments = [Segment(text="far too much text for this audio", start=0.0, end=0.02)]
        with logging_enabled(), self.assertLogs("exordium.text.base", level="WARNING"):
            self.assertEqual(self.aligner.align_segments(waveform, segments), [])

    def test_align_on_audio_too_short_for_its_text_returns_empty(self):
        waveform = np.zeros(MIN_ALIGN_SAMPLES, dtype=np.float32)
        with logging_enabled(), self.assertLogs("exordium.text.whisperx_align", "WARNING"):
            self.assertEqual(self.aligner.align(waveform, "that"), [])


if __name__ == "__main__":
    unittest.main()
