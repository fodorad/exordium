"""Tests for transcript metrics and segment-annotation validation (pure logic)."""

import unittest

import numpy as np

from exordium.text.base import Word
from exordium.text.evaluation import (
    SegmentValidation,
    TranscriptEvaluator,
    TranscriptMetrics,
    evaluate_transcript,
    is_acceptable,
    normalize_asr,
    validate_segment,
    validate_segments,
)


def _stream(sentence: str, step: float = 1.0, dur: float = 0.5) -> list[Word]:
    return [Word(t, i * step, i * step + dur) for i, t in enumerate(sentence.split())]


class TestEvaluateTranscript(unittest.TestCase):
    def test_identical_is_perfect(self):
        # Identical after normalization (casing/punctuation removed).
        m = evaluate_transcript("Hey guys, WHAT up!", "hey guys what up")
        self.assertIsInstance(m, TranscriptMetrics)
        self.assertEqual(m.wer, 0.0)
        self.assertEqual(m.cer, 0.0)
        self.assertEqual(m.similarity, 100.0)
        self.assertEqual(m.normalized_wer, 0.0)
        self.assertIsNone(m.semantic_similarity)

    def test_one_word_substitution(self):
        m = evaluate_transcript("what do you want to eat", "what do you need to eat")
        # 1 substitution (want→need) over 6 reference words.
        self.assertAlmostEqual(m.wer, 1 / 6, places=6)
        self.assertGreater(m.similarity, 80.0)

    def test_completely_different(self):
        m = evaluate_transcript("hello world", "totally unrelated phrase here")
        self.assertGreaterEqual(m.wer, 1.0)
        self.assertLess(m.similarity, 50.0)

    def test_empty_reference_does_not_divide_by_zero(self):
        m = evaluate_transcript("", "something")
        self.assertGreaterEqual(m.wer, 0.0)

    def test_normalization_removes_cosmetic_errors(self):
        # "ten thousand" and "10,000" are identical in meaning; the Whisper
        # normalizer canonicalizes both to the same token, the raw metric does not.
        m = evaluate_transcript(
            "I hit ten thousand subscribers",
            "I hit 10,000 subscribers",
        )
        self.assertGreater(m.wer, 0.0)  # raw counts the number format as errors
        self.assertEqual(m.normalized_wer, 0.0)  # Whisper normalizer cancels them
        self.assertGreaterEqual(m.normalized_similarity, m.similarity)

    def test_contraction_normalization(self):
        # Standard Whisper normalizer expands contractions on both sides.
        m = evaluate_transcript("do not stop", "don't stop")
        self.assertEqual(m.normalized_wer, 0.0)

    def test_semantic_similarity_with_embedder(self):
        # Dummy embedder: identical vectors → cosine 1.0.
        same = evaluate_transcript("a", "b", embedder=lambda _t: np.array([[1.0, 0.0], [1.0, 0.0]]))
        self.assertAlmostEqual(same.semantic_similarity, 1.0, places=6)
        # Orthogonal vectors → cosine 0.0.
        orth = evaluate_transcript("a", "b", embedder=lambda _t: np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.assertAlmostEqual(orth.semantic_similarity, 0.0, places=6)


class TestNormalizeAsr(unittest.TestCase):
    def test_english_canonicalizes_numbers(self):
        self.assertEqual(normalize_asr("ten thousand"), normalize_asr("10,000"))

    def test_basic_normalizer_lowercases_and_strips(self):
        self.assertEqual(normalize_asr("Hello, World!", kind="basic").split(), ["hello", "world"])

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            normalize_asr("x", kind="nope")


class TestIsAcceptable(unittest.TestCase):
    def _metrics(self, nwer, semantic=None):
        return TranscriptMetrics(0, 0, 0, nwer, 0, 0, semantic, "r", "h")

    def test_accept_on_low_wer(self):
        self.assertTrue(is_acceptable(self._metrics(0.10)))

    def test_reject_on_high_wer_without_semantic(self):
        self.assertFalse(is_acceptable(self._metrics(0.40)))

    def test_semantic_rescues_high_wer(self):
        self.assertTrue(is_acceptable(self._metrics(0.40, semantic=0.9)))

    def test_custom_thresholds(self):
        self.assertFalse(is_acceptable(self._metrics(0.18), max_normalized_wer=0.15))


class TestTranscriptEvaluatorPureLogic(unittest.TestCase):
    def test_semantic_disabled_loads_no_model(self):
        # semantic_model=None → no embedding model is ever built.
        evaluator = TranscriptEvaluator(semantic_model=None)
        m = evaluator.evaluate("hello world", "hello world")
        self.assertIsInstance(m, TranscriptMetrics)
        self.assertIsNone(m.semantic_similarity)
        self.assertEqual(m.normalized_wer, 0.0)


class TestValidateSegment(unittest.TestCase):
    def setUp(self):
        # "silence hey guys what do you want to eat for lunch today bye"
        self.words = _stream("silence hey guys what do you want to eat for lunch today bye")

    def test_accept_when_annotation_matches(self):
        # "what ... today" spans words 3..11 → start 3.0, end 11.5.
        v = validate_segment("what do you want to eat for lunch today", self.words, 3.0, 11.5)
        self.assertIsInstance(v, SegmentValidation)
        self.assertEqual(v.decision, "accept")
        self.assertAlmostEqual(v.recovered_start, 3.0)
        self.assertAlmostEqual(v.recovered_end, 11.5)

    def test_recut_when_annotation_shifted(self):
        v = validate_segment("what do you want to eat for lunch today", self.words, 0.0, 5.0)
        self.assertEqual(v.decision, "recut")
        self.assertAlmostEqual(v.recovered_start, 3.0)
        self.assertAlmostEqual(v.start_offset, 3.0)

    def test_drop_when_text_absent(self):
        v = validate_segment("the quick brown fox jumps over", self.words, 0.0, 3.0)
        self.assertEqual(v.decision, "drop")
        self.assertIsNone(v.recovered_start)
        self.assertIsNone(v.match)
        self.assertEqual(v.score, 0.0)

    def test_tolerance_controls_accept_vs_recut(self):
        # "for lunch today" spans words 9..11 → start 9.0, end 11.5.
        # Annotation is within 0.5s of both edges → accept.
        v = validate_segment("for lunch today", self.words, 9.4, 11.8, tolerance=0.5)
        self.assertEqual(v.decision, "accept")
        # The same annotation is a recut under a tight tolerance.
        v2 = validate_segment("for lunch today", self.words, 9.4, 11.8, tolerance=0.2)
        self.assertEqual(v2.decision, "recut")


class TestValidateSegments(unittest.TestCase):
    def test_batch_decisions(self):
        words = _stream("hey guys what do you want to eat for lunch today")
        results = validate_segments(
            [
                ("what do you want to eat", 2.0, 7.5),
                ("nonexistent phrase here", 0.0, 2.0),
            ],
            words,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].decision, "accept")
        self.assertEqual(results[1].decision, "drop")


if __name__ == "__main__":
    unittest.main()
