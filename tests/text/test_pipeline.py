"""Tests for the SpeechAlignmentPipeline facade."""

import unittest

from exordium.text.pipeline import SpeechAlignmentPipeline
from tests.fixtures import (
    PRETRAINED,
    TEST_WHISPER_MODEL,
    ModelTestCase,
)


class TestPipelineBackendSelection(unittest.TestCase):
    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            SpeechAlignmentPipeline(backend="does-not-exist", device_id=None)


class TestSpeechAlignmentPipeline(ModelTestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = SpeechAlignmentPipeline(
            backend="torchaudio",
            whisper_model=TEST_WHISPER_MODEL,
            device_id=None,
            pretrained=PRETRAINED,
        )

    def test_shares_whisper_when_provided(self):
        # A second pipeline can reuse the first's Whisper (avoids double-loading).
        other = SpeechAlignmentPipeline(
            backend="torchaudio",
            whisper=self.pipe.whisper,
            device_id=None,
            pretrained=PRETRAINED,
        )
        self.assertIs(other.whisper, self.pipe.whisper)


if __name__ == "__main__":
    unittest.main()
