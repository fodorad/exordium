"""Tests for the shared text/alignment base helpers."""

import unittest

from exordium.text.base import (
    MIN_ALIGN_SAMPLES,
    WAV2VEC2_RECEPTIVE_FIELD_SAMPLES,
    WAV2VEC2_STRIDE_SAMPLES,
    WIDEN_WINDOW_SECONDS,
    Segment,
    prepare_segments,
    wav2vec2_min_samples,
)
from tests.fixtures import logging_enabled

SAMPLE_RATE = 16000


def chars(text: str) -> int:
    """Token count the way the base class estimates it."""
    return sum(1 for c in text if not c.isspace())


def min_samples(text: str) -> int:
    """The default ``min_samples_for`` callback used across these tests."""
    return wav2vec2_min_samples(chars(text))


class TestWav2vec2MinSamples(unittest.TestCase):
    """The frames-per-token bound that decides how much audio a text needs."""

    def test_emission_frames_cover_every_token(self):
        # The contract: floor((N - 400) / 320) + 1 >= num_tokens.
        for num_tokens in range(1, 30):
            n = wav2vec2_min_samples(num_tokens)
            frames = (n - WAV2VEC2_RECEPTIVE_FIELD_SAMPLES) // WAV2VEC2_STRIDE_SAMPLES + 1
            self.assertGreaterEqual(frames, num_tokens, f"{num_tokens} tokens got {frames} frames")

    def test_is_tight_not_merely_sufficient(self):
        # One stride less must be too few frames, or we are wasting audio.
        for num_tokens in range(3, 30):
            n = wav2vec2_min_samples(num_tokens) - WAV2VEC2_STRIDE_SAMPLES
            frames = (n - WAV2VEC2_RECEPTIVE_FIELD_SAMPLES) // WAV2VEC2_STRIDE_SAMPLES + 1
            self.assertLess(frames, num_tokens)

    def test_never_below_the_two_frame_floor(self):
        # A single frame divides by zero in whisperX regardless of token count.
        self.assertEqual(wav2vec2_min_samples(1), MIN_ALIGN_SAMPLES)
        self.assertEqual(wav2vec2_min_samples(0), MIN_ALIGN_SAMPLES)

    def test_grows_with_token_count(self):
        self.assertGreater(wav2vec2_min_samples(4), wav2vec2_min_samples(1))


class TestPrepareSegments(unittest.TestCase):
    """Widening degenerate Whisper micro-segments instead of discarding them."""

    def test_keeps_normal_segments_untouched(self):
        segments = [
            Segment(text="hey guys", start=0.0, end=2.0),
            Segment(text="what do you want to eat", start=2.0, end=6.0),
        ]
        self.assertEqual(prepare_segments(segments, min_samples, SAMPLE_RATE), segments)

    def test_widens_degenerate_micro_segment_instead_of_dropping_it(self):
        # Regression: Whisper long-form emits real text with a ~0.02s span. The word is
        # real and the audio around it exists, so it must survive with usable timings.
        segments = [Segment(text="that", start=123.880, end=123.900)]
        prepared = prepare_segments(segments, min_samples, SAMPLE_RATE, num_samples=200 * 16000)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0].text, "that")
        span = int(prepared[0].end * SAMPLE_RATE) - int(prepared[0].start * SAMPLE_RATE)
        self.assertGreaterEqual(span, min_samples("that"))

    def test_widened_window_clears_the_speech_floor_not_just_the_model_floor(self):
        # The geometric minimum for "that" is only ~85ms, but the word takes ~250ms to
        # say; a window that tight clips it and the timestamp comes back truncated.
        segments = [Segment(text="that", start=10.0, end=10.02)]
        prepared = prepare_segments(segments, min_samples, SAMPLE_RATE, num_samples=20 * 16000)
        span = prepared[0].end - prepared[0].start
        self.assertGreaterEqual(span, WIDEN_WINDOW_SECONDS)

    def test_healthy_short_segments_are_left_alone(self):
        # The speech floor must not reshape spans Whisper got right; only degenerate
        # segments (below the model's own requirement) are widened.
        segments = [Segment(text="hey", start=1.0, end=1.2)]
        self.assertEqual(prepare_segments(segments, min_samples, SAMPLE_RATE), segments)

    def test_widened_window_stays_centred_on_the_original(self):
        segments = [Segment(text="I", start=10.0, end=10.02)]
        prepared = prepare_segments(segments, min_samples, SAMPLE_RATE, num_samples=20 * 16000)
        # The word is still where Whisper heard it, just with room to breathe.
        self.assertLess(prepared[0].start, 10.0)
        self.assertGreater(prepared[0].end, 10.02)
        self.assertAlmostEqual((prepared[0].start + prepared[0].end) / 2, 10.01, places=2)

    def test_widening_is_clamped_to_the_audio(self):
        num_samples = 5 * SAMPLE_RATE
        for segment in (
            Segment(text="that", start=0.0, end=0.02),  # at the very start
            Segment(text="that", start=4.99, end=5.0),  # at the very end
        ):
            prepared = prepare_segments([segment], min_samples, SAMPLE_RATE, num_samples)
            self.assertEqual(len(prepared), 1)
            self.assertGreaterEqual(prepared[0].start, 0.0)
            self.assertLessEqual(prepared[0].end, num_samples / SAMPLE_RATE)
            span = int(prepared[0].end * SAMPLE_RATE) - int(prepared[0].start * SAMPLE_RATE)
            self.assertGreaterEqual(span, min_samples("that"))

    def test_drops_only_when_the_audio_itself_is_too_short(self):
        # Widening cannot conjure audio that does not exist.
        segments = [Segment(text="a much longer sentence than this audio", start=0.0, end=0.02)]
        prepared = prepare_segments([*segments], min_samples, SAMPLE_RATE, num_samples=800)
        self.assertEqual(prepared, [])

    def test_drops_segment_starting_past_the_end_of_the_audio(self):
        # Whisper can emit an end (or start) beyond the true duration. Widening such a
        # segment would silently align it to the last window of the recording and hand
        # back a confidently-wrong timestamp, so it must be dropped instead.
        num_samples = 5 * SAMPLE_RATE
        segments = [Segment(text="ghost", start=6.0, end=6.02)]
        self.assertEqual(prepare_segments(segments, min_samples, SAMPLE_RATE, num_samples), [])

    def test_kept_segment_overrunning_the_audio_is_clamped(self):
        # Left unclamped, whisperX rescales timestamps by a nominal duration longer than
        # the audio it actually slices, inflating its frame ratio and stretching the word
        # times past the end of the recording.
        num_samples = 5 * SAMPLE_RATE
        segments = [Segment(text="overrun", start=4.0, end=9.0)]
        prepared = prepare_segments(segments, min_samples, SAMPLE_RATE, num_samples)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0].end, num_samples / SAMPLE_RATE)
        self.assertEqual(prepared[0].start, 4.0)

    def test_drops_blank_text_segments(self):
        good = Segment(text="hey", start=0.0, end=2.0)
        segments = [Segment(text="   ", start=0.0, end=2.0), good]
        self.assertEqual(prepare_segments(segments, min_samples, SAMPLE_RATE), [good])

    def test_segment_running_past_end_of_audio_is_widened_backwards(self):
        num_samples = SAMPLE_RATE  # 1 second of audio
        segments = [Segment(text="trailing", start=0.99, end=5.0)]
        prepared = prepare_segments(segments, min_samples, SAMPLE_RATE, num_samples)
        self.assertEqual(len(prepared), 1)
        self.assertLessEqual(prepared[0].end, 1.0)
        self.assertLess(prepared[0].start, 0.99)

    def test_logs_widened_and_dropped_segments(self):
        with logging_enabled(), self.assertLogs("exordium.text.base") as logs:
            prepare_segments(
                [Segment(text="that", start=1.0, end=1.02)],
                min_samples,
                SAMPLE_RATE,
                num_samples=10 * SAMPLE_RATE,
            )
        self.assertIn("Widened", logs.output[0])
        with logging_enabled(), self.assertLogs("exordium.text.base", level="WARNING") as logs:
            prepare_segments(
                [Segment(text="that", start=0.0, end=0.02)], min_samples, SAMPLE_RATE, 500
            )
        self.assertIn("Dropping", logs.output[0])

    def test_empty_input_returns_empty(self):
        self.assertEqual(prepare_segments([], min_samples, SAMPLE_RATE), [])


if __name__ == "__main__":
    unittest.main()
