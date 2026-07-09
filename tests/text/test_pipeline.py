"""Tests for the SpeechAlignmentPipeline facade."""

import unittest

from exordium.text.base import Word
from exordium.text.evaluation import SegmentValidation, TranscriptMetrics
from exordium.text.pipeline import SpeechAlignmentPipeline
from tests.fixtures import AUDIO_MULTISPEAKER, ModelTestCase


class TestPipelineBackendSelection(unittest.TestCase):
    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            SpeechAlignmentPipeline(backend="does-not-exist", device_id=None)


class TestSpeechAlignmentPipeline(ModelTestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = SpeechAlignmentPipeline(backend="torchaudio", device_id=None)

    def test_shares_whisper_when_provided(self):
        # A second pipeline can reuse the first's Whisper (avoids double-loading).
        other = SpeechAlignmentPipeline(
            backend="torchaudio", whisper=self.pipe.whisper, device_id=None
        )
        self.assertIs(other.whisper, self.pipe.whisper)

    def test_transcribe_returns_text(self):
        text = self.pipe.transcribe(AUDIO_MULTISPEAKER)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_transcribe_words_returns_word_stream(self):
        words = self.pipe.transcribe_words(AUDIO_MULTISPEAKER)
        self.assertGreater(len(words), 0)
        self.assertTrue(all(isinstance(w, Word) for w in words))

    def test_evaluate_annotation_good_annotation_low_wer(self):
        # Use the pipeline's own prediction as a near-perfect annotation.
        prediction = self.pipe.transcribe(AUDIO_MULTISPEAKER)
        metrics = self.pipe.evaluate_annotation(AUDIO_MULTISPEAKER, prediction)
        self.assertIsInstance(metrics, TranscriptMetrics)
        self.assertLess(metrics.wer, 0.1)
        self.assertLess(metrics.normalized_wer, 0.1)
        self.assertGreater(metrics.similarity, 90.0)
        self.assertIsNone(metrics.semantic_similarity)

    def test_evaluate_annotation_semantic_opt_in(self):
        prediction = self.pipe.transcribe(AUDIO_MULTISPEAKER)
        metrics = self.pipe.evaluate_annotation(AUDIO_MULTISPEAKER, prediction, semantic=True)
        self.assertIsNotNone(metrics.semantic_similarity)
        # Identical text → near-perfect semantic agreement.
        self.assertGreater(metrics.semantic_similarity, 0.95)

    def test_validate_segments_locates_known_text(self):
        results = self.pipe.validate_segments(
            AUDIO_MULTISPEAKER,
            [
                ("what do you guys want to eat for lunch", 0.0, 3.0),
                ("this phrase is definitely not spoken", 50.0, 55.0),
            ],
        )
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SegmentValidation)
        self.assertIn(results[0].decision, {"accept", "recut"})
        self.assertIsNotNone(results[0].recovered_start)
        self.assertEqual(results[1].decision, "drop")


if __name__ == "__main__":
    unittest.main()
