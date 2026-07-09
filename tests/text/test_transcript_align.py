"""Tests for dataset-agnostic fuzzy transcript re-alignment (pure logic)."""

import unittest

from exordium.text.base import SegmentMatch, Word
from exordium.text.transcript_align import find_segment, find_segments, normalize_token


def _stream(sentence: str, step: float = 1.0, dur: float = 0.5) -> list[Word]:
    """Build a synthetic timed word stream: word i spans [i*step, i*step+dur]."""
    return [
        Word(text=tok, start=i * step, end=i * step + dur) for i, tok in enumerate(sentence.split())
    ]


class TestNormalizeToken(unittest.TestCase):
    def test_lowercase_and_strip_punctuation(self):
        self.assertEqual(normalize_token("Hello, WORLD!"), "hello world")

    def test_strips_diacritics(self):
        self.assertEqual(normalize_token("Héllo Wörld"), "hello world")

    def test_pure_punctuation_is_empty(self):
        self.assertEqual(normalize_token("--- ... !!!"), "")

    def test_keeps_apostrophes_and_digits(self):
        self.assertEqual(normalize_token("It's 100%"), "it's 100")


class TestFindSegment(unittest.TestCase):
    def setUp(self):
        self.words = _stream("Hey guys, what do you want to eat for lunch today ?")

    def test_exact_substring_match(self):
        m = find_segment("what do you want to eat", self.words)
        self.assertIsInstance(m, SegmentMatch)
        assert m is not None
        self.assertEqual(m.text, "what do you want to eat")
        self.assertEqual((m.word_start_idx, m.word_end_idx), (2, 7))
        self.assertAlmostEqual(m.start, 2.0)
        self.assertAlmostEqual(m.end, 7.5)
        self.assertGreaterEqual(m.score, 99.0)

    def test_recovers_times_from_matched_words(self):
        m = find_segment("for lunch today", self.words)
        assert m is not None
        self.assertAlmostEqual(m.start, self.words[m.word_start_idx].start)
        self.assertAlmostEqual(m.end, self.words[m.word_end_idx].end)

    def test_paraphrase_with_asr_noise_still_matches(self):
        # "wanna" vs "want to" — minor ASR-style difference, should stay above cutoff.
        m = find_segment("what do you wanna eat", self.words, score_cutoff=70.0)
        self.assertIsNotNone(m)

    def test_absent_text_returns_none(self):
        self.assertIsNone(
            find_segment("the quick brown fox jumps over", self.words, score_cutoff=80.0)
        )

    def test_empty_query_returns_none(self):
        self.assertIsNone(find_segment("", self.words))
        self.assertIsNone(find_segment("...", self.words))

    def test_empty_stream_returns_none(self):
        self.assertIsNone(find_segment("anything", []))

    def test_cutoff_is_respected(self):
        # A weak partial overlap should be rejected at a high cutoff.
        self.assertIsNone(
            find_segment("want to eat something else entirely", self.words, score_cutoff=95.0)
        )

    def test_punctuation_only_words_skipped_but_indices_correct(self):
        words = _stream("hello , world")  # index 1 normalizes to empty
        m = find_segment("hello world", words)
        assert m is not None
        self.assertEqual(m.word_start_idx, 0)
        self.assertEqual(m.word_end_idx, 2)


class TestFindSegments(unittest.TestCase):
    def test_batch_returns_one_result_per_query(self):
        words = _stream("Hey guys what do you want to eat for lunch today")
        results = find_segments(["for lunch today", "nonsense zzz qqq"], words)
        self.assertEqual(len(results), 2)
        assert results[0] is not None
        self.assertEqual(results[0].text, "for lunch today")
        self.assertIsNone(results[1])


if __name__ == "__main__":
    unittest.main()
