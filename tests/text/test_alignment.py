"""Tests for torchaudio forced alignment and Whisper word timestamps."""

import unittest

import numpy as np

from exordium.text.alignment import (
    TorchaudioForcedAligner,
    WhisperWordTimestamper,
    normalize_words,
)
from exordium.text.base import MIN_ALIGN_SAMPLES, Segment, Word
from exordium.text.transcript_align import find_segment
from tests.fixtures import (
    AUDIO_MULTISPEAKER,
    ModelTestCase,
    best_anchored_word,
    logging_enabled,
)


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

    def test_align_segments_recovers_degenerate_micro_segment(self):
        # Regression: Whisper long-form emits real text spanning ~0.02s. The slice was
        # below MMS_FA's conv floor, which raised and killed the whole recording. The
        # word must not merely survive the crash — it must come back timed.
        segments = [
            Segment(text="hey guys", start=0.0, end=2.0),
            Segment(text="that", start=2.100, end=2.120),  # degenerate: 320 samples
            Segment(text="what do you guys want to eat", start=2.5, end=6.0),
        ]
        words = self.aligner.align_segments(AUDIO_MULTISPEAKER, segments)
        self.assertIn("that", [w.text for w in words])
        starts = [w.start for w in words]
        self.assertEqual(starts, sorted(starts))

    def test_recovered_degenerate_word_matches_ground_truth_timing(self):
        # The point of widening is not just "no crash" — the word must come back timed
        # *correctly*. Align the full sentence for ground truth, then re-align one of its
        # words as a degenerate 0.02s micro-segment and check we land back on it.
        target = best_anchored_word(
            self.aligner.align(
                AUDIO_MULTISPEAKER, "Hey guys, what do you guys want to eat for lunch?"
            )
        )
        midpoint = (target.start + target.end) / 2
        degenerate = [Segment(text=target.text, start=midpoint, end=midpoint + 0.02)]

        words = self.aligner.align_segments(AUDIO_MULTISPEAKER, degenerate)
        self.assertEqual([w.text for w in words], [target.text])
        # Recovered to within ~2 emission frames of where the word really is. Without the
        # speech-realistic floor the window clips the word and this drifts by >100 ms.
        self.assertAlmostEqual(words[0].start, target.start, delta=0.1)
        self.assertAlmostEqual(words[0].end, target.end, delta=0.1)

    def test_min_align_samples_uses_the_models_own_tokenizer(self):
        # normalize_words strips punctuation, so it must not inflate the requirement.
        self.assertEqual(
            self.aligner.min_align_samples("that!"), self.aligner.min_align_samples("that")
        )
        # Longer text genuinely needs more frames, hence more audio.
        self.assertGreater(
            self.aligner.min_align_samples("that"), self.aligner.min_align_samples("i")
        )

    def test_min_align_samples_falls_back_for_untokenizable_text(self):
        # Punctuation-only text has no MMS_FA tokens; fall back to the two-frame floor
        # rather than computing a requirement from zero tokens.
        self.assertEqual(self.aligner.min_align_samples("..."), MIN_ALIGN_SAMPLES)

    def test_align_on_audio_too_short_for_its_text_returns_empty(self):
        # align() is the raw single-shot path: it cannot widen, so it declines.
        waveform = np.zeros(MIN_ALIGN_SAMPLES, dtype=np.float32)
        with logging_enabled(), self.assertLogs("exordium.text.alignment", level="WARNING"):
            self.assertEqual(self.aligner.align(waveform, "that"), [])


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
