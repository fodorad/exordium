"""Tests for StreamingFacePipeline: frames in, per-frame tracking updates out."""

import unittest

import torch

from exordium.video.core.detection import IouTracker
from exordium.video.core.streaming import StreamingTracker
from exordium.video.face.detector.base import FaceDetector
from exordium.video.face.streaming import StreamingFacePipeline

BIG = (10.0, 10.0, 130.0, 130.0)
SMALL = (600.0, 400.0, 640.0, 440.0)


class ScriptedDetector(FaceDetector):
    """Hand-written detector: reports fixed boxes per call, in order."""

    def __init__(self, script: list[list[tuple[float, float, float, float]]]):
        super().__init__()
        self.script = script
        self.calls = 0

    def run_detector(self, images):
        out = []
        for _ in images:
            boxes = self.script[min(self.calls, len(self.script) - 1)]
            self.calls += 1
            out.append(
                [(torch.tensor(b), torch.zeros(5, 2), 0.95) for b in boxes],
            )
        return out


def _frame(h: int = 480, w: int = 720) -> torch.Tensor:
    return torch.randint(0, 255, (3, h, w), dtype=torch.uint8)


def _pipeline(script, **kwargs) -> StreamingFacePipeline:
    return StreamingFacePipeline(ScriptedDetector(script), StreamingTracker(IouTracker(), **kwargs))


class TestDetectFrame(unittest.TestCase):
    def test_sets_the_frame_id_and_keeps_the_frame_as_source(self):
        frame = _frame()
        fd = ScriptedDetector([[BIG, SMALL]]).detect_frame(frame, frame_id=7)
        self.assertEqual(len(fd), 2)
        self.assertEqual(fd.frame_id, 7)
        self.assertIs(fd[0].source, frame)


class TestPush(unittest.TestCase):
    def test_counts_frames_when_no_id_is_given(self):
        pipeline = _pipeline([[BIG]])
        ids = [pipeline.push(_frame()).frame_id for _ in range(3)]
        self.assertEqual(ids, [0, 1, 2])

    def test_explicit_frame_ids_are_used(self):
        pipeline = _pipeline([[BIG]])
        self.assertEqual(pipeline.push(_frame(), frame_id=10).frame_id, 10)
        self.assertEqual(pipeline.push(_frame()).frame_id, 11)

    def test_tracks_and_leads_across_frames(self):
        pipeline = _pipeline([[BIG, SMALL]])
        updates = [pipeline.push(_frame()) for _ in range(10)]
        last = updates[-1]
        self.assertEqual(len(last.scores), 2)
        self.assertEqual(last.protagonist_detection.bb_xywh[2], 120)

    def test_frame_without_faces_still_advances(self):
        pipeline = _pipeline([[BIG], []])
        first = pipeline.push(_frame())
        empty = pipeline.push(_frame())
        self.assertEqual(empty.assignments, [])
        self.assertEqual(empty.protagonist_id, first.protagonist_id)

    def test_numpy_frames_are_accepted(self):
        pipeline = _pipeline([[BIG]])
        update = pipeline.push(_frame().permute(1, 2, 0).numpy())
        self.assertEqual(len(update.assignments), 1)


class TestProcess(unittest.TestCase):
    def test_yields_one_update_per_frame_lazily(self):
        pipeline = _pipeline([[BIG]])
        consumed = []

        def frames():
            for i in range(5):
                consumed.append(i)
                yield _frame()

        stream = pipeline.process(frames())
        first = next(stream)
        self.assertEqual(first.frame_id, 0)
        self.assertEqual(consumed, [0])  # nothing read ahead
        self.assertEqual(len(list(stream)), 4)

    def test_becomes_stable_on_a_steady_stream(self):
        pipeline = _pipeline(
            [[BIG, SMALL]], min_track_frames=5, min_stable_frames=8, min_margin=2.0
        )
        updates = list(pipeline.process(_frame() for _ in range(15)))
        self.assertFalse(updates[0].stable)
        self.assertTrue(updates[-1].stable)


class TestReset(unittest.TestCase):
    def test_restarts_frame_counter_and_tracks(self):
        pipeline = _pipeline([[BIG, SMALL]])
        for _ in range(5):
            pipeline.push(_frame())
        pipeline.reset()
        update = pipeline.push(_frame())
        self.assertEqual(update.frame_id, 0)
        self.assertEqual(len(update.scores), 2)
        self.assertEqual(sorted(t for t, _ in update.assignments), [0, 1])


if __name__ == "__main__":
    unittest.main()
